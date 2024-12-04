"""Custom normalization layers."""

from typing import Optional, Tuple, Union

import recogni.torch
import torch
import torch.nn as nn


class RecogniRMSNorm(recogni.torch.nn.RMSNorm):
    """Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(eps=eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.recogni_rms_norm = recogni.torch.nn.RMSNorm(eps=eps)
        self.residual_add = recogni.torch.math.Add()

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

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
