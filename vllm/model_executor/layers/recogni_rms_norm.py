"""Custom normalization layers."""

from typing import Optional, Union

import recogni.torch
import torch
import torch.nn as nn

from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform


class RecogniRMSNorm(recogni.torch.nn.MACRMSNorm):
    """Root mean square normalization.
    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    name = "recogni_rms_norm"

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(eps=eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.residual_add = recogni.torch.math.Add()
        self._forward_method = self.dispatch_forward()

    # def _forward(
    #     self,
    #     x: torch.Tensor,
    #     residual: Optional[torch.Tensor] = None,
    # ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    #     """PyTorch-native implementation equivalent to forward()."""
    #     orig_dtype = x.dtype
    #     x = x.to(torch.float32)
    #     if residual is not None:
    #         x = x + residual.to(torch.float32)
    #         residual = x.to(orig_dtype)

    #     variance = x.pow(2).mean(dim=-1, keepdim=True)
    #     x = x * torch.rsqrt(variance + self.variance_epsilon)
    #     x = x.to(orig_dtype) * self.weight
    #     if residual is None:
    #         return x
    #     else:
    #         return x, residual
    def dispatch_forward(self):
        # NOTE(woosuk): Here we assume that vLLM was built for only one
        # specific backend. Currently, we do not support dynamic dispatching.
        compilation_config = get_current_vllm_config().compilation_config
        enabled = True
        if enabled:
            compilation_config.enabled_custom_ops.update([self.__class__.name])
        else:
            compilation_config.disabled_custom_ops.update([self.__class__.name])

        if not enabled:
            return self.forward_native

        if current_platform.is_rocm():
            raise NotImplementedError("HIP not supported")
        elif current_platform.is_cpu():
            raise NotImplementedError("CPU not supported")
        elif current_platform.is_hpu():
            raise NotImplementedError("HPU not supported")
        elif current_platform.is_tpu():
            raise NotImplementedError("TPU not supported")
        elif current_platform.is_xpu():
            raise NotImplementedError("XPU not supported")
        else:
            return self.forward_cuda

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # if residual is not None:
        #     ops.fused_add_rms_norm(
        #         x,
        #         residual,
        #         self.weight.data,
        #         self.variance_epsilon,
        #     )
        #     return x, residual

        # out = torch.empty_like(x)
        # ops.rms_norm(
        #     out,
        #     x,
        #     self.weight.data,
        #     self.variance_epsilon,
        # )
        if residual is not None:
            x = self.residual_add(x, residual)
            residual = x

        out = super().forward(x)

        if residual is None:
            return out
        else:
            return out, residual

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)
