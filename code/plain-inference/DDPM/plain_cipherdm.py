import argparse
from contextlib import contextmanager
from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.linear import Array
from jax import lax

import plain as base
from flax_ddpm.script_utils import get_args


_CHEXP_COEFFS = jnp.asarray(
    [
        0.14021878,
        0.27541278,
        0.22122865,
        0.14934221,
        0.0907736,
        0.04369614,
        0.02087868,
        0.00996535,
    ],
    dtype=jnp.float32,
)
_APPROX1_SILU_A_COEFFS = jnp.asarray(
    [-0.52212664, -0.16910363, -0.01420163],
    dtype=jnp.float32,
)
_APPROX1_SILU_B_COEFFS = jnp.asarray(
    [0.03453821, 0.49379432, 0.19784596, -0.00602401, 0.00008032],
    dtype=jnp.float32,
)
_APPROX1_MISH_A_COEFFS = jnp.asarray(
    [-0.55684445, -0.18375535, -0.01572019],
    dtype=jnp.float32,
)
_APPROX1_MISH_B_COEFFS = jnp.asarray(
    [0.07559242, 0.54902050, 0.20152583, -0.00735309, 0.00010786],
    dtype=jnp.float32,
)
_NEG_SIX = jnp.asarray(-6.0, dtype=jnp.float32)
_NEG_TWO = jnp.asarray(-2.0, dtype=jnp.float32)
_POS_SIX = jnp.asarray(6.0, dtype=jnp.float32)
_NEG_FOURTEEN = jnp.asarray(-14.0, dtype=jnp.float32)
_EPS = jnp.asarray(1e-6, dtype=jnp.float32)
GOLDSCHMIDT_ITERS = 3


@jax.jit
def transform_x(x: Array, x_min: float = -14.0, x_max: float = 0.0) -> Array:
    x_min = jnp.asarray(x_min, dtype=x.dtype)
    x_max = jnp.asarray(x_max, dtype=x.dtype)
    return 2.0 * (x - x_min) / (x_max - x_min) - 1.0


@jax.jit
def chexp(x: Array) -> Array:
    work_x = transform_x(x)
    coeffs = _CHEXP_COEFFS.astype(work_x.dtype)

    def body(i, state):
        idx = coeffs.shape[0] - 1 - i
        b_kplus1, b_kplus2 = state
        b_k = 2.0 * work_x * b_kplus1 - b_kplus2 + coeffs[idx]
        return b_k, b_kplus1

    init = (jnp.zeros_like(work_x), jnp.zeros_like(work_x))
    b1, b2 = lax.fori_loop(0, coeffs.shape[0] - 1, body, init)
    return coeffs[0] + work_x * b1 - b2


@jax.jit
def reciprocal_goldschmidt_normalized_approx(c: Array) -> Array:
    w = 2.9142 - 2.0 * c
    r0 = w
    e0 = 1.0 - c * w
    r1 = r0 * (1.0 + e0)
    e1 = e0 * e0
    r2 = r1 * (1.0 + e1)
    e2 = e1 * e1
    r3 = r2 * (1.0 + e2)
    return r3


@jax.jit
def reciprocal_goldschmidt_positive(b_abs: Array) -> Array:
    work = jnp.maximum(b_abs, jnp.asarray(_EPS, dtype=b_abs.dtype))
    c, exponent = jnp.frexp(work)
    factor = jnp.ldexp(jnp.ones_like(work), -exponent)
    r = reciprocal_goldschmidt_normalized_approx(c)
    return r * factor


@partial(jax.jit, static_argnames=("axis",))
def _approx1_softmax_kernel(
    x: Array,
    where_mask: Array,
    initial_value: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
) -> Array:
    x_masked = jnp.where(where_mask, x, initial_value)
    x_max = jnp.max(x_masked, axis=axis, keepdims=True)
    shifted = x - x_max
    valid = shifted > jnp.asarray(_NEG_FOURTEEN, dtype=shifted.dtype)
    active = jnp.logical_and(where_mask, valid)
    exp_x = jnp.where(active, chexp(shifted), jnp.zeros_like(shifted))
    denom = jnp.sum(exp_x, axis=axis, keepdims=True)
    inv_denom = reciprocal_goldschmidt_positive(denom)
    return exp_x * inv_denom


def approx1_softmax(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
    initial: Optional[Array] = None,
) -> Array:
    where_mask = jnp.ones_like(x, dtype=jnp.bool_) if where is None else where
    initial_value = jnp.asarray(-jnp.inf if initial is None else initial, dtype=x.dtype)
    return _approx1_softmax_kernel(x, where_mask, initial_value, axis=axis)


def _approx1_piecewise_poly(x: Array, a_coeffs: Array, b_coeffs: Array) -> Array:
    x2 = jnp.square(x)
    seg1 = (a_coeffs[2] * x + a_coeffs[1]) * x + a_coeffs[0]
    poly_even = ((b_coeffs[4] * x2 + b_coeffs[3]) * x2 + b_coeffs[2]) * x2 + b_coeffs[0]
    seg2 = poly_even + b_coeffs[1] * x

    below = x < jnp.asarray(_NEG_SIX, dtype=x.dtype)
    middle_low = jnp.logical_and(x >= jnp.asarray(_NEG_SIX, dtype=x.dtype), x < jnp.asarray(_NEG_TWO, dtype=x.dtype))
    above = x > jnp.asarray(_POS_SIX, dtype=x.dtype)

    out = jnp.where(middle_low, seg1, seg2)
    out = jnp.where(above, x, out)
    out = jnp.where(below, jnp.zeros_like(x), out)
    return out


@jax.jit
def approx1_silu(x: Array) -> Array:
    coeff_a = _APPROX1_SILU_A_COEFFS.astype(x.dtype)
    coeff_b = _APPROX1_SILU_B_COEFFS.astype(x.dtype)
    return _approx1_piecewise_poly(x, coeff_a, coeff_b)


@jax.jit
def approx1_mish(x: Array) -> Array:
    coeff_a = _APPROX1_MISH_A_COEFFS.astype(x.dtype)
    coeff_b = _APPROX1_MISH_B_COEFFS.astype(x.dtype)
    return _approx1_piecewise_poly(x, coeff_a, coeff_b)


@contextmanager
def approx1_softmax_context(enabled: bool = True):
    if not enabled:
        yield
        return
    raw_softmax = nn.softmax
    nn.softmax = approx1_softmax
    try:
        yield
    finally:
        nn.softmax = raw_softmax


@contextmanager
def approx1_silu_context(enabled: bool = True):
    if not enabled:
        yield
        return
    raw_silu = nn.silu
    nn.silu = approx1_silu
    try:
        yield
    finally:
        nn.silu = raw_silu


@contextmanager
def approx1_mish_context(enabled: bool = True):
    if not enabled:
        yield
        return
    raw_mish = getattr(nn, "mish", None)
    nn.mish = approx1_mish
    try:
        yield
    finally:
        if raw_mish is None:
            delattr(nn, "mish")
        else:
            nn.mish = raw_mish


def main(args: argparse.Namespace):
    with approx1_softmax_context(), approx1_silu_context(), approx1_mish_context():
        base.main(args)


if __name__ == "__main__":
    parser = base.get_sample_arg_parser()
    main(get_args(parser))
