from typing import Optional

import recogni.torch as rtorch
import torch
from recogni import torch_emu

from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.utils import set_weight_attrs


class RecogniConfig(QuantizationConfig):
    """Config class for AWQ.

    Reference: https://arxiv.org/abs/2306.00978
    """

    def __init__(self, modules_to_not_convert=None, *args, **kwargs) -> None:
        self.modules_to_not_convert = modules_to_not_convert or []

    def __repr__(self) -> str:
        return "RecogniConfig"

    def get_name(self) -> str:
        return "recogni"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, *args, **kwargs) -> "RecogniConfig":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[LinearMethodBase]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped_recogni(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            return RecogniFastLinear()
        return None


def is_layer_skipped_recogni(prefix: str, modules_to_not_convert: list[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


class RecogniFastLinear(LinearMethodBase):
    """Fast linear method for Recogni."""

    def __init__(self):
        self.dtype_in = rtorch.ops.dtypes.FP16(eb=-1)
        self.dtype_out = rtorch.ops.dtypes.FP16(eb=-1)
        self.weight_type = rtorch.ops.dtypes.LnsI5F10(
            eb=-1,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight = torch.nn.Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})

        weight_loader_fn = extra_weight_attrs["weight_loader"]

        # if weight_loader_fn is not None:
        #     weight_loader = _get_quantzied_weight_loader(
        #         extra_weight_attrs["weight_loader"], self.weight_type
        #     )
        #     extra_weight_attrs["weight_loader"] = weight_loader
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch_emu.ops.qonly.linear(
            input=x,
            weights=layer.weight,
            bias=bias,
            dtype_in=self.dtype_in.emu_eb_data_type,
            dtype_out=self.dtype_out.emu_eb_data_type,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        orig_device = layer.weight.device
        layer.weight = torch.nn.Parameter(
            rtorch.quantize(
                layer.weight.to(torch.float16).to("cuda"), self.weight_type
            ).to(orig_device),
        )
