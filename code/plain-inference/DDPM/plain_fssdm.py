from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen.linear import Array

import plain as base
from flax_ddpm.script_utils import get_args


_APPROX3_EXP_COEFFS = jnp.asarray(
    [
        1.0,
        0.9736789799002139,
        0.4349844528710021,
        0.10637438059443326,
        0.013503439864618789,
        0.000682062067913879,
    ],
    dtype=jnp.float32,
)
_APPROX3_SILU_COEFFS = jnp.asarray(
    [-7.402215e-02, 2.5967005e-01, 1.0894996e-01, 1.089919e-02],
    dtype=jnp.float32,
)
_APPROX3_MISH_NEG_COEFFS = jnp.asarray(
    [-0.09822489, 0.26691618, 0.11584595, 0.01176954],
    dtype=jnp.float32,
)
_APPROX3_MISH_POSRES_COEFFS = jnp.asarray(
    [-0.09446990, 0.02936850, 0.02736322, 0.00344253],
    dtype=jnp.float32,
)
_APPROX3_SILU_T = 6.0
_APPROX3_MISH_T = 6.0


# Linear seed fitted offline on [1, 2] with fit_linear_seed_residual().
# This seed is then used on a mantissa-normalized denominator so the online
# reciprocal stays within the same bounded layer.
_PSLN_ALPHA = 1.0
_PSLN_A = 1.4081226750014004
_PSLN_B = -0.4766499506046734
_EPS = 1e-6


# -----------------------------
# 1) offline fit helper
# -----------------------------
def fit_linear_seed_residual(L: float, U: float, n_grid: int = 100000):
    """
    ?[L,U] ?seed r0=a+b*t?     residual |1 - t*(a+b*t)| €?
     dense-grid €?minimax / Remez?    """
    x = np.linspace(L, U, n_grid, dtype=np.float64)

    # English note: localized comment removed.
    A = np.stack([np.ones_like(x), x], axis=1)
    y = 1.0 / x
    coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coeffs.tolist()
    return float(a), float(b)


def eval_psln_error(alpha: float, a: float, b: float, L: float, U: float, n_grid: int = 100000):
    """
     t in [L,U] €?seed + 1 Newton ?    """
    t = np.linspace(L, U, n_grid, dtype=np.float64)
    true = 1.0 / t

    r0 = a + b * t
    r1 = r0 * (2.0 - t * r0)

    seed_rel = np.max(np.abs((r0 - true) / true))
    newton_rel = np.max(np.abs((r1 - true) / true))
    seed_res = np.max(np.abs(1.0 - t * r0))
    newton_res = np.max(np.abs(1.0 - t * r1))

    return {
        "seed_max_rel_err": float(seed_rel),
        "newton1_max_rel_err": float(newton_rel),
        "seed_max_residual": float(seed_res),
        "newton1_max_residual": float(newton_res),
    }


@jax.jit
def approx_exp_nr2_tensor(x: Array) -> Array:
    coeffs = _APPROX3_EXP_COEFFS.astype(x.dtype)
    u = 0.5 * x
    u2 = u * u
    u4 = u2 * u2
    y0 = coeffs[0] + coeffs[1] * u
    y1 = coeffs[2] + coeffs[3] * u
    y2 = coeffs[4] + coeffs[5] * u
    y = y0 + y1 * u2 + y2 * u4
    return y * y


# -----------------------------
# 2) online reciprocal
# -----------------------------
@jax.jit
def recip_psln_jax(
    denom: Array,
    alpha: float,
    a: float,
    b: float,
    eps: float = 1e-6,
) -> Array:
    """
    PSLN reciprocal:
        t = alpha * denom
        r0 = a + b*t
        r1 = r0 * (2 - t*r0)
        inv_denom = alpha * r1

    ㄧ?
      - b*t           : 1
      - t*r0          : 1
      - r0*(...)      : 1
       3 ′?    """
    denom = jnp.maximum(denom, jnp.asarray(eps, dtype=denom.dtype))

    alpha = jnp.asarray(alpha, dtype=denom.dtype)
    a = jnp.asarray(a, dtype=denom.dtype)
    b = jnp.asarray(b, dtype=denom.dtype)

    t = denom * alpha
    r0 = a + b * t
    r1 = r0 * (2.0 - t * r0)
    inv_denom = alpha * r1
    return inv_denom


@jax.jit
def reciprocal_psln_positive(denom: Array) -> Array:
    """
    Keep the provided PSLN kernel as the core reciprocal step, but normalize
    the positive denominator to a mantissa range [1, 2) first. This preserves
    the approximation family while making it usable for softmax denominators
    that span a much wider range than a single linear seed can cover.
    """
    work = jnp.maximum(denom, jnp.asarray(_EPS, dtype=denom.dtype))
    mantissa, exponent = jnp.frexp(work)  # work = mantissa * 2**exponent, mantissa in [0.5, 1)
    t = mantissa * jnp.asarray(2.0, dtype=work.dtype)  # t in [1, 2)
    inv_t = recip_psln_jax(t, _PSLN_ALPHA, _PSLN_A, _PSLN_B, eps=_EPS)
    factor = jnp.ldexp(jnp.ones_like(work), 1 - exponent)  # 1 / work = (1 / t) * 2**(1-exponent)
    return inv_t * factor


@partial(jax.jit, static_argnames=("axis",))
def _approx3_softmax_kernel(
    x: Array,
    where_mask: Array,
    initial_value: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
) -> Array:
    x_masked = jnp.where(where_mask, x, initial_value)
    x_max = jnp.max(x_masked, axis=axis, keepdims=True)
    shifted = x - x_max
    raw_dtype = shifted.dtype
    need_fp32 = raw_dtype in (jnp.float16, jnp.bfloat16)
    xf = shifted.astype(jnp.float32) if need_fp32 else shifted
    exp_x = approx_exp_nr2_tensor(xf)
    exp_x = jnp.where(where_mask, exp_x, jnp.zeros_like(exp_x))
    denom = jnp.sum(exp_x, axis=axis, keepdims=True)
    denom = jnp.clip(denom, a_min=jnp.asarray(1e-12, dtype=denom.dtype))
    inv_denom = reciprocal_psln_positive(denom)
    out = exp_x * inv_denom
    out = jnp.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(raw_dtype) if need_fp32 else out


def approx3_softmax(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
    initial: Optional[Array] = None,
) -> Array:
    where_mask = jnp.ones_like(x, dtype=jnp.bool_) if where is None else where
    initial_value = jnp.asarray(-jnp.inf if initial is None else initial, dtype=x.dtype)
    return _approx3_softmax_kernel(x, where_mask, initial_value, axis=axis)


def _mask_inside_radius(work_x: Array, threshold: float) -> Array:
    limit2 = jnp.asarray(threshold * threshold, dtype=work_x.dtype)
    return 1.0 - ((work_x * work_x) >= limit2).astype(work_x.dtype)


@jax.jit
def approx3_silu(x: Array) -> Array:
    raw_dtype = x.dtype
    work_x = x.astype(jnp.float32) if raw_dtype in (jnp.float16, jnp.bfloat16) else x
    coeffs = _APPROX3_SILU_COEFFS.astype(work_x.dtype)
    s = (work_x >= 0).astype(work_x.dtype)
    sx = s * work_x
    z = work_x - 2.0 * sx
    poly = ((coeffs[3] * z + coeffs[2]) * z + coeffs[1]) * z + coeffs[0]
    mask = _mask_inside_radius(work_x, _APPROX3_SILU_T)
    out = sx + mask * poly
    out = jnp.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(raw_dtype)


@jax.jit
def approx3_mish(x: Array) -> Array:
    raw_dtype = x.dtype
    work_x = x.astype(jnp.float32) if raw_dtype in (jnp.float16, jnp.bfloat16) else x
    neg_c = _APPROX3_MISH_NEG_COEFFS.astype(work_x.dtype)
    pos_c = _APPROX3_MISH_POSRES_COEFFS.astype(work_x.dtype)
    s = (work_x >= 0).astype(work_x.dtype)
    sx = s * work_x
    z = work_x - 2.0 * sx
    p_neg = ((neg_c[3] * z + neg_c[2]) * z + neg_c[1]) * z + neg_c[0]
    p_pos = ((pos_c[3] * z + pos_c[2]) * z + pos_c[1]) * z + pos_c[0]
    core = p_neg + s * (p_pos - p_neg)
    mask = _mask_inside_radius(work_x, _APPROX3_MISH_T)
    out = sx + mask * core
    out = jnp.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(raw_dtype)


@contextmanager
def approx3_softmax_context(enabled: bool = True):
    if not enabled:
        yield
        return
    raw_softmax = nn.softmax
    nn.softmax = approx3_softmax
    try:
        yield
    finally:
        nn.softmax = raw_softmax


@contextmanager
def approx3_silu_context(enabled: bool = True):
    if not enabled:
        yield
        return
    raw_silu = nn.silu
    nn.silu = approx3_silu
    try:
        yield
    finally:
        nn.silu = raw_silu


@contextmanager
def approx3_mish_context(enabled: bool = True):
    if not enabled:
        yield
        return
    raw_mish = getattr(nn, "mish", None)
    nn.mish = approx3_mish
    try:
        yield
    finally:
        if raw_mish is None:
            delattr(nn, "mish")
        else:
            nn.mish = raw_mish


def main(args: argparse.Namespace):
    with approx3_softmax_context(), approx3_silu_context(), approx3_mish_context():
        base.main(args)


if __name__ == "__main__":
    parser = base.get_sample_arg_parser()
    main(get_args(parser))


