# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import math
from collections import OrderedDict

import torch
from torch import Tensor, nn

from .integrations.hub_kernels import use_kernel_forward_from_hub
from .utils import logging
from .utils.import_utils import is_torchdynamo_compiling


logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("GeluTanh")
class GELUTanh(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://huggingface.co/papers/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self, use_gelu_tanh_python: bool = False):
        super().__init__()
        if use_gelu_tanh_python:
            self.act = self._gelu_tanh_python
        else:
            self.act = functools.partial(nn.functional.gelu, approximate="tanh")

    def _gelu_tanh_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


# Added for compatibility with autoawq which is archived now and imports PytorchGELUTanh from activations.py
PytorchGELUTanh = GELUTanh


@use_kernel_forward_from_hub("NewGELU")
class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://huggingface.co/papers/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


@use_kernel_forward_from_hub("GeLU")
class GELUActivation(nn.Module):
    """
    ュ?QuickGELU ㈠ = x * sigmoid(1.702 * x)
    ｇ㈠aw_x * gate?gateigmoid(1.702*raw_x)?    """

    def __init__(self, use_gelu_python: bool = False, **kwargs):
        super().__init__()
        self.use_gelu_python = bool(use_gelu_python)

        # 5th-order polynomial coefficients for sigmoid(x) approximation
        # on negative side:
        #   outer: x in [-7, -2.2)
        #   inner: x in [-2.2, 0)
        # coef order: c0..c5 (low->high): c0 + c1*x + ... + c5*x^5
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
    def _poly5(x, c, x2, x3, x4, x5):
        # P(x)=c0+c1*x+c2*x^2+c3*x^3+c4*x^4+c5*x^5
        return c[0] + c[1]*x + c[2]*x2 + c[3]*x3 + c[4]*x4 + c[5]*x5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_x = x
        xs = 1.702 * x  # scaled x for gate

        # ---- compute polynomial in float32 for stability (important for fp16/bf16) ----
        xs_f = xs.float() if xs.dtype in (torch.float16, torch.bfloat16) else xs

        cO = self.c_outer.to(device=xs.device, dtype=xs_f.dtype)
        cI = self.c_inner.to(device=xs.device, dtype=xs_f.dtype)

        # masks on scaled xs (use same boundaries as your original code)
        mL = xs_f < -7.0
        mA = (xs_f >= -7.0) & (xs_f < -2.2)
        mB = (xs_f >= -2.2) & (xs_f < 0.0)
        mC = (xs_f >= 0.0) & (xs_f < 2.2)
        mD = (xs_f >= 2.2) & (xs_f < 7.0)
        mR = xs_f >= 7.0

        x2 = xs_f * xs_f
        x3 = x2 * xs_f
        x4 = x2 * x2
        x5 = x4 * xs_f

        # negative-side polynomials
        P_outer = self._poly5(xs_f, cO, x2, x3, x4, x5)  # [-7,-2.2)
        P_inner = self._poly5(xs_f, cI, x2, x3, x4, x5)  # [-2.2,0)

        # English note: localized comment removed.
        # evaluate P(-x) efficiently by flipping odd terms
        # P(-x)=c0 - c1*x + c2*x^2 - c3*x^3 + c4*x^4 - c5*x^5
        P_outer_neg = cO[0] - cO[1]*xs_f + cO[2]*x2 - cO[3]*x3 + cO[4]*x4 - cO[5]*x5
        P_inner_neg = cI[0] - cI[1]*xs_f + cI[2]*x2 - cI[3]*x3 + cI[4]*x4 - cI[5]*x5

        segL = torch.zeros_like(xs_f)
        segA = P_outer
        segB = P_inner
        segC = 1.0 - P_inner_neg
        segD = 1.0 - P_outer_neg
        segR = torch.ones_like(xs_f)

        gate = (
            mL.to(xs_f.dtype) * segL
            + mA.to(xs_f.dtype) * segA
            + mB.to(xs_f.dtype) * segB
            + mC.to(xs_f.dtype) * segC
            + mD.to(xs_f.dtype) * segD
            + mR.to(xs_f.dtype) * segR
        )
        gate = gate.clamp_(0.0, 1.0)

        # multiply in float32 if needed, then cast back
        if raw_x.dtype in (torch.float16, torch.bfloat16):
            y = (raw_x.float() * gate).to(dtype=raw_x.dtype)
        else:
            y = raw_x * gate.to(dtype=raw_x.dtype)

        return y

@use_kernel_forward_from_hub("SiLU")
class SiLUActivation(nn.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.silu(input)


@use_kernel_forward_from_hub("FastGELU")
class FastGELUActivation(nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


@use_kernel_forward_from_hub("QuickGELU")
class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://huggingface.co/papers/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://huggingface.co/papers/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)


class AccurateGELUActivation(nn.Module):
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """

    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1 + torch.tanh(self.precomputed_constant * (input + 0.044715 * torch.pow(input, 3))))


class MishActivation(nn.Module):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://huggingface.co/papers/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def __init__(self):
        super().__init__()
        self.act = nn.functional.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        return input * torch.tanh(nn.functional.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, input: Tensor) -> Tensor:
        return input


class LaplaceActivation(nn.Module):
    """
    Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See
    https://huggingface.co/papers/2209.10655

    Inspired by squared relu, but with bounded range and gradient for better stability
    """

    def forward(self, input, mu=0.707107, sigma=0.282095):
        input = (input - mu).div(sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + torch.erf(input))


class ReLUSquaredActivation(nn.Module):
    """
    Applies the relu^2 activation introduced in https://huggingface.co/papers/2109.08668
    """

    def forward(self, input):
        relu_applied = nn.functional.relu(input)
        squared = torch.square(relu_applied)
        return squared


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


class XIELUActivation(nn.Module):
    """
    Applies the xIELU activation function introduced in https://arxiv.org/abs/2411.13010

    If the user has installed the nickjbrowning/XIELU wheel, we import xIELU CUDA
    Otherwise, we emit a single warning and use xIELU Python
    """

    def __init__(
        self,
        alpha_p_init=0.8,
        alpha_n_init=0.8,
        beta=0.5,
        eps=-1e-6,
        dtype=torch.bfloat16,
        with_vector_loads=False,
    ):
        super().__init__()
        self.alpha_p = nn.Parameter(torch.log(torch.expm1(torch.tensor(alpha_p_init, dtype=dtype))).unsqueeze(0))
        self.alpha_n = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(alpha_n_init - beta, dtype=dtype))).unsqueeze(0)
        )
        self.register_buffer("beta", torch.tensor(beta, dtype=dtype))
        self.register_buffer("eps", torch.tensor(eps, dtype=dtype))
        self.with_vector_loads = with_vector_loads
        # Temporary until xIELU CUDA fully implemented
        self._beta_scalar = float(self.beta.detach().cpu().float().item())
        self._eps_scalar = float(self.eps.detach().cpu().float().item())

        self._xielu_cuda_obj = None
        try:
            import xielu.ops  # noqa: F401

            self._xielu_cuda_obj = torch.classes.xielu.XIELU()
            msg = "Using experimental xIELU CUDA."
            try:
                from torch._dynamo import allow_in_graph

                self._xielu_cuda_fn = allow_in_graph(self._xielu_cuda)
                msg += " Enabled torch._dynamo for xIELU CUDA."
            except Exception as err:
                msg += f" Could not enable torch._dynamo for xIELU ({err}) - this may result in slower performance."
                self._xielu_cuda_fn = self._xielu_cuda
            logger.warning_once(msg)
        except Exception as err:
            logger.warning_once(
                "CUDA-fused xIELU not available (%s) ?falling back to a Python version.\n"
                "For CUDA xIELU (experimental), `pip install git+https://github.com/nickjbrowning/XIELU`",
                str(err),
            )

    def _xielu_python(self, x: Tensor) -> Tensor:
        alpha_p = nn.functional.softplus(self.alpha_p)
        alpha_n = self.beta + nn.functional.softplus(self.alpha_n)
        return torch.where(
            x > 0,
            alpha_p * x * x + self.beta * x,
            (torch.expm1(torch.min(x, self.eps)) - x) * alpha_n + self.beta * x,
        )

    def _xielu_cuda(self, x: Tensor) -> Tensor:
        """Firewall function to prevent torch.compile from seeing .item() calls"""
        original_shape = x.shape
        # CUDA kernel expects 3D tensors, reshape if needed
        while x.dim() < 3:
            x = x.unsqueeze(0)
        if x.dim() > 3:
            x = x.view(-1, 1, x.size(-1))
        if original_shape != x.shape:
            logger.warning_once(
                "Warning: xIELU input tensor expects 3 dimensions but got (shape: %s). Reshaping to (shape: %s).",
                original_shape,
                x.shape,
            )
        result = self._xielu_cuda_obj.forward(
            x,
            self.alpha_p.to(x.dtype),
            self.alpha_n.to(x.dtype),
            # Temporary until xIELU CUDA fully implemented -> self.{beta,eps}.item()
            self._beta_scalar,
            self._eps_scalar,
            self.with_vector_loads,
        )
        return result.view(original_shape)

    def forward(self, input: Tensor) -> Tensor:
        if self._xielu_cuda_obj is not None and input.is_cuda:
            if not is_torchdynamo_compiling():
                return self._xielu_cuda_fn(input)
            else:
                logger.warning_once("torch._dynamo is compiling, using Python version of xIELU.")
        return self._xielu_python(input)


ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": GELUTanh,
    "gelu_python_tanh": (GELUTanh, {"use_gelu_tanh_python": True}),
    "gelu_accurate": AccurateGELUActivation,
    "laplace": LaplaceActivation,
    "leaky_relu": nn.LeakyReLU,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": SiLUActivation,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "prelu": nn.PReLU,
    "xielu": XIELUActivation,
}
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")


# For backwards compatibility with: from activations import gelu_python
gelu_python = get_activation("gelu_python")
gelu_new = get_activation("gelu_new")
gelu = get_activation("gelu")
gelu_fast = get_activation("gelu_fast")
quick_gelu = get_activation("quick_gelu")
silu = get_activation("silu")
mish = get_activation("mish")
linear_act = get_activation("linear")


