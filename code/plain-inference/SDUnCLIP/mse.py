# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 22,
    "legend.fontsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.facecolor": "#f4f4f4",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

torch.set_default_dtype(torch.float64)
_shape_t = Union[int, list[int], torch.Size, Tuple[int, ...]]

COLOR_BASE = "#111111"
COLOR_CIPHER = "#d62728"
COLOR_FSSDM = "#1f77b4"
COLOR_FSSDM2 = "#DAA520"

FONT_FAMILY = "serif"
FONT_WEIGHT = "normal"
AXIS_LABEL_WEIGHT = "bold"
TICK_WEIGHT = "bold"
LABEL_SIZE = 22
TICK_SIZE = 18

SAVE_FORMAT = "pdf"
SAVE_DPI = 1600


# ============================================================
# Exact SiLU
# ============================================================
def exact_silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


# ============================================================
# SiLU Approx1 (CipherDM)
# ============================================================
class SiLUApprox1(nn.Module):
    __constants__ = ["compute_fp32"]
    compute_fp32: bool

    def __init__(self, inplace: bool = False, compute_fp32: bool = False) -> None:
        super().__init__()
        self.compute_fp32 = compute_fp32

        self.register_buffer(
            "_a_coeffs",
            torch.tensor([-0.52212664, -0.16910363, -0.01420163], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_b_coeffs",
            torch.tensor([0.03453821, 0.49379432, 0.19784596, -0.00602401, 0.00008032], dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _poly2_horner(x: Tensor, c: Tensor) -> Tensor:
        return (c[2] * x + c[1]) * x + c[0]

    @staticmethod
    def _poly6_sparse(x: Tensor, c: Tensor) -> Tensor:
        x2 = x * x
        x4 = x2 * x2
        x6 = x2 * x4
        return c[0] + c[1] * x + c[2] * x2 + c[3] * x4 + c[4] * x6

    def forward(self, input: Tensor) -> Tensor:
        x = input

        mL = x < -6.0
        mA = (x >= -6.0) & (x < -2.0)
        mB = (x >= -2.0) & (x <= 6.0)

        x_calc = x.float() if (self.compute_fp32 and x.dtype in (torch.float16, torch.bfloat16)) else x

        a = self._a_coeffs.to(device=x_calc.device, dtype=x_calc.dtype)
        b = self._b_coeffs.to(device=x_calc.device, dtype=x_calc.dtype)

        segA = self._poly2_horner(x_calc, a)
        segB = self._poly6_sparse(x_calc, b)

        if segA.dtype != x.dtype:
            segA = segA.to(dtype=x.dtype)
            segB = segB.to(dtype=x.dtype)

        zeros = torch.zeros_like(x)

        out = torch.where(mL, zeros, x)
        out = torch.where(mA, segA, out)
        out = torch.where(mB, segB, out)
        return out


# ============================================================
# SiLU Approx2
# ============================================================
class SiLUApprox2(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = bool(inplace)
        self.register_buffer(
            "c_outer",
            torch.tensor(
                [5.90010486e-01, 4.21319936e-01, 1.26610092e-01, 1.97985686e-02, 1.59538912e-03, 5.25625056e-05],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "c_inner",
            torch.tensor(
                [4.99957471e-01, 2.49155471e-01, -4.01655846e-03, -2.84974728e-02, -6.76484741e-03, -4.34163661e-04],
                dtype=torch.float32,
            ),
            persistent=False,
        )

    @staticmethod
    def _poly5(x: Tensor, c: Tensor, x2: Tensor, x3: Tensor, x4: Tensor, x5: Tensor) -> Tensor:
        return c[0] + c[1] * x + c[2] * x2 + c[3] * x3 + c[4] * x4 + c[5] * x5

    def forward(self, x: Tensor) -> Tensor:
        raw_x = x
        xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        c_outer = self.c_outer.to(device=xf.device, dtype=xf.dtype)
        c_inner = self.c_inner.to(device=xf.device, dtype=xf.dtype)

        x2 = xf * xf
        x3 = x2 * xf
        x4 = x2 * x2
        x5 = x4 * xf

        p_outer = self._poly5(xf, c_outer, x2, x3, x4, x5)
        p_inner = self._poly5(xf, c_inner, x2, x3, x4, x5)
        p_outer_neg = c_outer[0] - c_outer[1] * xf + c_outer[2] * x2 - c_outer[3] * x3 + c_outer[4] * x4 - c_outer[5] * x5
        p_inner_neg = c_inner[0] - c_inner[1] * xf + c_inner[2] * x2 - c_inner[3] * x3 + c_inner[4] * x4 - c_inner[5] * x5

        gate = torch.zeros_like(xf)
        gate = torch.where((xf >= -7.0) & (xf < -2.2), p_outer, gate)
        gate = torch.where((xf >= -2.2) & (xf < 0.0), p_inner, gate)
        gate = torch.where((xf >= 0.0) & (xf < 2.2), 1.0 - p_inner_neg, gate)
        gate = torch.where((xf >= 2.2) & (xf < 7.0), 1.0 - p_outer_neg, gate)
        gate = torch.where(xf >= 7.0, torch.ones_like(gate), gate)
        gate = gate.clamp_(0.0, 1.0)

        out = raw_x.float() * gate if raw_x.dtype in (torch.float16, torch.bfloat16) else raw_x * gate.to(dtype=raw_x.dtype)
        out = out.to(dtype=raw_x.dtype)
        return out


# ============================================================
# SiLU Approx3
# ============================================================
SILU_OPTIMIZED_COEFFS = [-7.402215e-02, 2.5967005e-01, 1.0894996e-01, 1.089919e-02]
SILU_OPTIMIZED_T = 6.0


class SiLUApprox3(nn.Module):
    def __init__(self, coeffs=SILU_OPTIMIZED_COEFFS, T: float = SILU_OPTIMIZED_T, inplace: bool = False):
        super().__init__()
        self.T = float(T)
        self.inplace = bool(inplace)

        self.register_buffer(
            "coeffs",
            torch.tensor(coeffs, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _poly3(z: Tensor, c: Tensor) -> Tensor:
        z2 = z * z
        z3 = z2 * z
        return c[0] + c[1] * z + c[2] * z2 + c[3] * z3

    def forward(self, x: Tensor) -> Tensor:
        raw_x = x
        xf = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        c = self.coeffs.to(device=xf.device, dtype=xf.dtype)

        s = (xf >= 0).to(dtype=xf.dtype)
        sx = s * xf
        two_sx = 2.0 * sx
        z = xf + (-two_sx)

        x2 = xf * xf
        T_sq = self.T * self.T
        m = (1.0 - (x2 >= T_sq).to(dtype=xf.dtype))

        z_clamped = z.clamp(min=-self.T, max=0.0)
        pz = self._poly3(z_clamped, c)

        mpz = m * pz
        out = sx + mpz
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = out.to(dtype=raw_x.dtype)
        return out


# ============================================================
# Exact reference functions
# ============================================================
def exact_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x)


def exact_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = logits.amax(dim=dim, keepdim=True)
    x = logits - x_max
    exp_x = torch.exp(x)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp


def exact_softmax_pair_y(x: torch.Tensor) -> torch.Tensor:
    logits = torch.stack([x, torch.zeros_like(x)], dim=-1)
    y = exact_softmax(logits, dim=-1)
    return y[..., 0]


# ============================================================
# CipherDM exp / reciprocal / softmax
# ============================================================
def _transform_x(x: torch.Tensor, x_min: float = -14.0, x_max: float = 0.0) -> torch.Tensor:
    return 2.0 * (x - x_min) / (x_max - x_min) - 1.0


def cipher_exp(x: torch.Tensor) -> torch.Tensor:
    xt = _transform_x(x)
    t0 = torch.ones_like(xt)
    t1 = xt
    coeffs = [
        0.14021878, 0.27541278, 0.22122865, 0.14934221,
        0.09077360, 0.04369614, 0.02087868, 0.00996535
    ]

    ex = coeffs[0] * t0 + coeffs[1] * t1
    two_xt = 2.0 * xt
    t_nm2, t_nm1 = t0, t1

    for c in coeffs[2:]:
        t_n = two_xt * t_nm1 - t_nm2
        ex = ex + c * t_n
        t_nm2, t_nm1 = t_nm1, t_n

    ex = torch.where(x <= -14.0, torch.zeros_like(ex), ex)
    return ex


def reciprocal_goldschmidt_real(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    x_abs = torch.abs(x).clamp_min(eps)

    mantissa, exponent = torch.frexp(x_abs)
    factor = torch.ldexp(torch.ones_like(x_abs), -exponent)
    c = x_abs * factor

    r = 2.9142 - 2.0 * c
    e = 1.0 - c * r
    r = r * (1.0 + e)
    e = e * e
    r = r * (1.0 + e)

    return sign * r * factor


def cipher_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = logits.amax(dim=dim, keepdim=True)
    x = logits - x_max
    valid = x > -14.0
    exp_x = torch.where(valid, cipher_exp(x), torch.zeros_like(x))
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    inv_sum = reciprocal_goldschmidt_real(sum_exp)
    return exp_x * inv_sum


def cipher_softmax_pair_y(x: torch.Tensor) -> torch.Tensor:
    logits = torch.stack([x, torch.zeros_like(x)], dim=-1)
    y = cipher_softmax(logits, dim=-1)
    return y[..., 0]


# ============================================================
# FSSDM exp / reciprocal / softmax
# ============================================================
_PSLN_ALPHA = 1.0
_PSLN_A = 1.4081226750014004
_PSLN_B = -0.4766499506046734
_PSLN_EPS = 1e-6


def fssdm_exp(x: torch.Tensor) -> torch.Tensor:
    u = 0.5 * x
    c0 = 1.0
    c1 = 0.9736789799002139
    c2 = 0.4349844528710021
    c3 = 0.10637438059443326
    c4 = 0.013503439864618789
    c5 = 0.000682062067913879

    u2 = u * u
    u4 = u2 * u2
    y0 = c0 + c1 * u
    y1 = c2 + c3 * u
    y2 = c4 + c5 * u
    y = y0 + y1 * u2 + y2 * u4
    return y * y


def _recip_psln_core(t: torch.Tensor, alpha: float = _PSLN_ALPHA, a: float = _PSLN_A, b: float = _PSLN_B, eps: float = _PSLN_EPS) -> torch.Tensor:
    t = torch.clamp(t, min=eps)

    alpha_t = torch.as_tensor(alpha, dtype=t.dtype, device=t.device)
    a_t = torch.as_tensor(a, dtype=t.dtype, device=t.device)
    b_t = torch.as_tensor(b, dtype=t.dtype, device=t.device)

    r0 = a_t + b_t * t
    r1 = r0 * (2.0 - t * r0)
    return alpha_t * r1


def reciprocal_psln_positive_real(denom: torch.Tensor, eps: float = _PSLN_EPS) -> torch.Tensor:
    work = torch.clamp(denom, min=eps)
    mantissa, exponent = torch.frexp(work)
    t = mantissa * 2.0
    inv_t = _recip_psln_core(t, alpha=_PSLN_ALPHA, a=_PSLN_A, b=_PSLN_B, eps=eps)
    factor = torch.ldexp(torch.ones_like(work), 1 - exponent)
    return inv_t * factor


def fssdm_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = logits.amax(dim=dim, keepdim=True)
    x = logits - x_max
    exp_x = fssdm_exp(x)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    inv_sum = reciprocal_psln_positive_real(sum_exp)
    return exp_x * inv_sum


def fssdm_softmax_pair_y(x: torch.Tensor) -> torch.Tensor:
    logits = torch.stack([x, torch.zeros_like(x)], dim=-1)
    y = fssdm_softmax(logits, dim=-1)
    return y[..., 0]


# ============================================================
# Utilities
# ============================================================
def force_suffix(path: Path, suffix: str) -> Path:
    return path.with_suffix(suffix)


def style_axes(ax):
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE, width=1.2, length=6)

    for tick in ax.get_xticklabels():
        tick.set_fontfamily(FONT_FAMILY)
        tick.set_fontweight(TICK_WEIGHT)
        tick.set_fontsize(TICK_SIZE)

    for tick in ax.get_yticklabels():
        tick.set_fontfamily(FONT_FAMILY)
        tick.set_fontweight(TICK_WEIGHT)
        tick.set_fontsize(TICK_SIZE)

    ax.xaxis.get_offset_text().set_fontfamily(FONT_FAMILY)
    ax.xaxis.get_offset_text().set_fontweight(TICK_WEIGHT)
    ax.xaxis.get_offset_text().set_fontsize(TICK_SIZE)

    ax.yaxis.get_offset_text().set_fontfamily(FONT_FAMILY)
    ax.yaxis.get_offset_text().set_fontweight(TICK_WEIGHT)
    ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)


# ============================================================
# Plotting: exp
# ============================================================
def plot_exp_compare(out_dir: Path):
    x = torch.linspace(-14.0, 0.0, 10000)

    base = exact_exp(x)
    cipher = cipher_exp(x)
    fssdm = fssdm_exp(x)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    ax.plot(x.numpy(), base.numpy(), label="Base", color=COLOR_BASE, linestyle="-", linewidth=2.4, alpha=0.98)
    ax.plot(x.numpy(), cipher.numpy(), label="CipherDM", color=COLOR_CIPHER, linestyle="--", linewidth=2.4, alpha=0.98)
    ax.plot(x.numpy(), fssdm.numpy(), label="FSSDM", color=COLOR_FSSDM, linestyle="-.", linewidth=2.4, alpha=0.98)

    ax.set_xlabel("Input to exp", fontfamily=FONT_FAMILY, fontweight=AXIS_LABEL_WEIGHT, fontsize=LABEL_SIZE)
    ax.set_ylabel("Output", fontfamily=FONT_FAMILY, fontweight=AXIS_LABEL_WEIGHT, fontsize=LABEL_SIZE)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    style_axes(ax)

    legend = ax.legend(
        loc="upper left", frameon=True, fancybox=True, framealpha=0.95, edgecolor="#bbbbbb",
        fontsize=20, borderpad=0.7, labelspacing=0.45, handlelength=2.4, handletextpad=0.8,
        prop={"family": FONT_FAMILY, "weight": FONT_WEIGHT, "size": 20},
    )
    legend.get_frame().set_linewidth(1.6)

    fig.tight_layout()
    fig.savefig(force_suffix(out_dir / "exp_curve_compare.png", ".pdf"), format=SAVE_FORMAT, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    err_cipher = torch.abs(cipher - base)
    err_fssdm = torch.abs(fssdm - base)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    ax.plot(x.numpy(), err_cipher.numpy(), label=f"CipherDM (MAE={err_cipher.mean().item():.3e})", color=COLOR_CIPHER, linestyle="-", linewidth=2.4, alpha=0.98)
    ax.plot(x.numpy(), err_fssdm.numpy(), label=f"FSSDM (MAE={err_fssdm.mean().item():.3e})", color=COLOR_FSSDM, linestyle="-", linewidth=2.4, alpha=0.98)

    ax.set_xlabel("Input to exp", fontfamily=FONT_FAMILY, fontweight=AXIS_LABEL_WEIGHT, fontsize=LABEL_SIZE)
    ax.set_ylabel("Absolute Error", fontfamily=FONT_FAMILY, fontweight=AXIS_LABEL_WEIGHT, fontsize=LABEL_SIZE)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    style_axes(ax)

    legend = ax.legend(
        loc="upper left", frameon=True, fancybox=True, framealpha=0.95, edgecolor="#bbbbbb",
        fontsize=18, borderpad=0.65, labelspacing=0.4, handlelength=2.2, handletextpad=0.75,
        prop={"family": FONT_FAMILY, "weight": FONT_WEIGHT, "size": 18},
    )
    legend.get_frame().set_linewidth(1.6)

    fig.tight_layout()
    fig.savefig(force_suffix(out_dir / "exp_error_compare.png", ".pdf"), format=SAVE_FORMAT, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ============================================================
# Plotting: softmax as y(x)=softmax([x,0])_1
# ============================================================
def plot_softmax_compare(out_dir: Path):
    x = torch.linspace(-14.0, 14.0, 10000)

    base_y = exact_softmax_pair_y(x)
    cipher_y = cipher_softmax_pair_y(x)
    fssdm_y = fssdm_softmax_pair_y(x)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    ax.plot(x.numpy(), base_y.numpy(), label="Base", color=COLOR_BASE, linestyle="-", linewidth=2.4, alpha=0.98)
    ax.plot(x.numpy(), cipher_y.numpy(), label="CipherDM", color=COLOR_CIPHER, linestyle="--", linewidth=2.4, alpha=0.98)
    ax.plot(x.numpy(), fssdm_y.numpy(), label="FSSDM", color=COLOR_FSSDM, linestyle="-.", linewidth=2.4, alpha=0.98)

    ax.set_xlabel("Pre-softmax input x in [x, 0]", fontfamily=FONT_FAMILY, fontweight=AXIS_LABEL_WEIGHT, fontsize=LABEL_SIZE)
    ax.set_ylabel("Output y", fontfamily=FONT_FAMILY, fontweight=AXIS_LABEL_WEIGHT, fontsize=LABEL_SIZE)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    style_axes(ax)

    legend = ax.legend(
        loc="best", frameon=True, fancybox=True, framealpha=0.95, edgecolor="#bbbbbb",
        fontsize=19, borderpad=0.65, labelspacing=0.42, handlelength=2.3, handletextpad=0.75,
        prop={"family": FONT_FAMILY, "weight": FONT_WEIGHT, "size": 19},
    )
    legend.get_frame().set_linewidth(1.6)

    fig.tight_layout()
    fig.savefig(force_suffix(out_dir / "softmax_curve_compare_pair.png", ".pdf"), format=SAVE_FORMAT, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    err_cipher = torch.abs(cipher_y - base_y)
    err_fssdm = torch.abs(fssdm_y - base_y)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    ax.plot(x.numpy(), err_cipher.numpy(), label=f"CipherDM (MAE={err_cipher.mean().item():.3e})", color=COLOR_CIPHER, linestyle="-", linewidth=2.4, alpha=0.98)
    ax.plot(x.numpy(), err_fssdm.numpy(), label=f"FSSDM (MAE={err_fssdm.mean().item():.3e})", color=COLOR_FSSDM, linestyle="-", linewidth=2.4, alpha=0.98)

    ax.set_xlabel("Input", fontfamily=FONT_FAMILY, fontweight=AXIS_LABEL_WEIGHT, fontsize=LABEL_SIZE)
    ax.set_ylabel("Absolute Error", fontfamily=FONT_FAMILY, fontweight=AXIS_LABEL_WEIGHT, fontsize=LABEL_SIZE)

    ax.yaxis.set_major_locator(MultipleLocator(0.5))

    style_axes(ax)

    legend = ax.legend(
        loc="best", frameon=True, fancybox=True, framealpha=0.95, edgecolor="#bbbbbb",
        fontsize=15, borderpad=0.55, labelspacing=0.38, handlelength=2.0, handletextpad=0.7,
        prop={"family": FONT_FAMILY, "weight": FONT_WEIGHT, "size": 15},
    )
    legend.get_frame().set_linewidth(1.6)

    fig.tight_layout()
    fig.savefig(force_suffix(out_dir / "softmax_error_compare_pair.png", ".pdf"), format=SAVE_FORMAT, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ============================================================
# Plotting: SiLU
# ============================================================
def plot_silu_compare(out_dir: Path):
    x = torch.linspace(-8.0, 8.0, 10000)

    base = exact_silu(x)
    silu1 = SiLUApprox1()(x)
    silu2 = SiLUApprox2()(x)
    silu3 = SiLUApprox3()(x)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    ax.plot(x.numpy(), base.numpy(), label="Base", color=COLOR_BASE, linestyle="-", linewidth=2.4, alpha=0.98)
    ax.plot(x.numpy(), silu1.numpy(), label="Approx1 (CipherDM)", color=COLOR_CIPHER, linestyle="--", linewidth=2.4, alpha=0.98)
    ax.plot(x.numpy(), silu2.numpy(), label="Approx2 (FSSSiLU1)", color=COLOR_FSSDM, linestyle="-.", linewidth=2.4, alpha=0.98)
    ax.plot(x.numpy(), silu3.numpy(), label="Approx3 (FSSSiLU2)", color=COLOR_FSSDM2, linestyle=":", linewidth=2.4, alpha=0.98)

    ax.set_xlabel("Input", fontfamily=FONT_FAMILY, fontweight=AXIS_LABEL_WEIGHT, fontsize=LABEL_SIZE)
    ax.set_ylabel("Output", fontfamily=FONT_FAMILY, fontweight=AXIS_LABEL_WEIGHT, fontsize=LABEL_SIZE)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    style_axes(ax)

    legend = ax.legend(
        loc="upper left", frameon=True, fancybox=True, framealpha=0.95, edgecolor="#bbbbbb",
        fontsize=20, borderpad=0.7, labelspacing=0.45, handlelength=2.4, handletextpad=0.8,
        prop={"family": FONT_FAMILY, "weight": FONT_WEIGHT, "size": 20},
    )
    legend.get_frame().set_linewidth(1.6)

    fig.tight_layout()
    fig.savefig(force_suffix(out_dir / "silu_curve_compare.png", ".pdf"), format=SAVE_FORMAT, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    err_silu1 = torch.abs(silu1 - base)
    err_silu2 = torch.abs(silu2 - base)
    err_silu3 = torch.abs(silu3 - base)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))
    ax.plot(x.numpy(), err_silu1.numpy(), label=f"CipherDM (MAE={err_silu1.mean().item():.3e})", color=COLOR_CIPHER, linestyle="-", linewidth=2.4, alpha=0.98)
    ax.plot(x.numpy(), err_silu2.numpy(), label=f"FSSSiLU1 (MAE={err_silu2.mean().item():.3e})", color=COLOR_FSSDM, linestyle="-", linewidth=2.4, alpha=0.98)
    ax.plot(x.numpy(), err_silu3.numpy(), label=f"FSSSiLU2 (MAE={err_silu3.mean().item():.3e})", color="#a349a4", linestyle="-", linewidth=2.4, alpha=0.98)

    ax.set_xlabel("Input", fontfamily=FONT_FAMILY, fontweight=AXIS_LABEL_WEIGHT, fontsize=LABEL_SIZE)
    ax.set_ylabel("Absolute Error", fontfamily=FONT_FAMILY, fontweight=AXIS_LABEL_WEIGHT, fontsize=LABEL_SIZE)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    style_axes(ax)

    legend = ax.legend(
        loc="upper left", frameon=True, fancybox=True, framealpha=0.95, edgecolor="#bbbbbb",
        fontsize=11, borderpad=0.68, labelspacing=0.44, handlelength=2.34, handletextpad=0.78,
        prop={"family": FONT_FAMILY, "weight": FONT_WEIGHT, "size": 11},
    )
    legend.get_frame().set_linewidth(1.6)

    fig.tight_layout()
    fig.savefig(force_suffix(out_dir / "silu_error_compare.png", ".pdf"), format=SAVE_FORMAT, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main(out_dir: str = "./compare_figs_softmax_pair_input"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_exp_compare(out)
    plot_softmax_compare(out)
    plot_silu_compare(out)

    print(f"All figures are saved to: {out.resolve()}")


if __name__ == "__main__":
    main()