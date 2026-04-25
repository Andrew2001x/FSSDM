import argparse
from contextlib import ExitStack, contextmanager
from functools import partial
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.linear import Array

import plain as base
import plain1jax as approx1
import plain3jax as approx3
from flax_ddpm.script_utils import get_args


Axis = Optional[Union[int, Tuple[int, ...]]]


def _prepare_softmax_inputs(
    x: Array,
    where: Optional[Array],
    initial: Optional[Array],
):
    where_mask = jnp.ones_like(x, dtype=jnp.bool_) if where is None else where
    initial_value = jnp.asarray(-jnp.inf if initial is None else initial, dtype=x.dtype)
    x_masked = jnp.where(where_mask, x, initial_value)
    x_max = jnp.max(x_masked, axis=-1, keepdims=True) if x.ndim == 1 else None
    return where_mask, initial_value, x_masked, x_max


def _make_softmax(
    *,
    exp_fn: Callable[[Array], Array],
    reciprocal_fn: Optional[Callable[[Array], Array]],
    exp_clip_floor: Optional[float] = None,
    inv_input_scale: Optional[float] = None,
    cast_fp32_for_exp: bool = False,
    sanitize_output: bool = False,
) -> Callable[[Array, Axis, Optional[Array], Optional[Array]], Array]:
    @partial(jax.jit, static_argnames=("axis",))
    def kernel(
        x: Array,
        where_mask: Array,
        initial_value: Array,
        axis: Axis = -1,
    ) -> Array:
        x_masked = jnp.where(where_mask, x, initial_value)
        x_max = jnp.max(x_masked, axis=axis, keepdims=True)
        shifted = x - x_max
        raw_dtype = shifted.dtype
        work_x = shifted.astype(jnp.float32) if cast_fp32_for_exp and raw_dtype in (jnp.float16, jnp.bfloat16) else shifted

        active_mask = where_mask
        if exp_clip_floor is not None:
            active_mask = jnp.logical_and(active_mask, work_x > jnp.asarray(exp_clip_floor, dtype=work_x.dtype))

        exp_x = exp_fn(work_x)
        exp_x = jnp.where(active_mask, exp_x, jnp.zeros_like(exp_x))
        denom = jnp.sum(exp_x, axis=axis, keepdims=True)

        if reciprocal_fn is None:
            denom = jnp.clip(denom, a_min=jnp.asarray(1e-12, dtype=denom.dtype))
            out = exp_x / denom
        else:
            recip_input = denom
            if inv_input_scale is not None:
                recip_input = recip_input * jnp.asarray(inv_input_scale, dtype=recip_input.dtype)
                inv_denom = reciprocal_fn(recip_input) * jnp.asarray(inv_input_scale, dtype=recip_input.dtype)
            else:
                inv_denom = reciprocal_fn(recip_input)
            out = exp_x * inv_denom

        if sanitize_output:
            out = jnp.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(raw_dtype) if work_x.dtype != raw_dtype else out

    def wrapped(
        x: Array,
        axis: Axis = -1,
        where: Optional[Array] = None,
        initial: Optional[Array] = None,
    ) -> Array:
        where_mask = jnp.ones_like(x, dtype=jnp.bool_) if where is None else where
        initial_value = jnp.asarray(-jnp.inf if initial is None else initial, dtype=x.dtype)
        return kernel(x, where_mask, initial_value, axis=axis)

    return wrapped


def _original_softmax(
    x: Array,
    axis: Axis = -1,
    where: Optional[Array] = None,
    initial: Optional[Array] = None,
) -> Array:
    return nn.softmax(x, axis=axis, where=where, initial=initial)


CASE_DESCRIPTIONS = {
    1: "approx1_silu + original_softmax",
    2: "original_silu + approx1_softmax(exp+reciprocal)",
    3: "original_silu + approx1_softmax(exp_only)",
    4: "original_silu + approx1_softmax(reciprocal_only)",
    5: "approx3_silu + original_softmax",
    6: "original_silu + approx3_softmax(exp+reciprocal)",
    7: "original_silu + approx3_softmax(exp_only)",
    8: "original_silu + approx3_softmax(reciprocal_only)",
}


def build_case(case_id: int):
    raw_silu = nn.silu
    raw_softmax = nn.softmax

    if case_id == 1:
        return approx1.approx1_silu, raw_softmax
    if case_id == 2:
        return raw_silu, approx1.approx1_softmax
    if case_id == 3:
        return raw_silu, _make_softmax(
            exp_fn=approx1.chexp,
            reciprocal_fn=None,
            exp_clip_floor=-14.0,
        )
    if case_id == 4:
        return raw_silu, _make_softmax(
            exp_fn=jnp.exp,
            reciprocal_fn=approx1.reciprocal_goldschmidt_positive,
        )
    if case_id == 5:
        return approx3.approx3_silu, raw_softmax
    if case_id == 6:
        return raw_silu, approx3.approx3_softmax
    if case_id == 7:
        return raw_silu, _make_softmax(
            exp_fn=approx3.approx_exp_nr2_tensor,
            reciprocal_fn=None,
            cast_fp32_for_exp=True,
            sanitize_output=True,
        )
    if case_id == 8:
        return raw_silu, _make_softmax(
            exp_fn=jnp.exp,
            reciprocal_fn=approx3.inv_newton2,
            inv_input_scale=approx3._APPROX3_INV_SCALE,
            cast_fp32_for_exp=True,
            sanitize_output=True,
        )
    raise ValueError(f"unsupported case_id: {case_id}")


@contextmanager
def patch_nn_functions(silu_fn, softmax_fn):
    raw_silu = nn.silu
    raw_softmax = nn.softmax
    nn.silu = silu_fn
    nn.softmax = softmax_fn
    try:
        yield
    finally:
        nn.silu = raw_silu
        nn.softmax = raw_softmax


def main(args: argparse.Namespace):
    silu_fn, softmax_fn = build_case(args.case)
    print(f"[CASE] {args.case}: {CASE_DESCRIPTIONS[args.case]}")
    with patch_nn_functions(silu_fn, softmax_fn):
        base.main(args)


if __name__ == "__main__":
    parser = base.get_sample_arg_parser()
    parser.add_argument("--case", type=int, required=True, choices=sorted(CASE_DESCRIPTIONS))
    main(get_args(parser))
