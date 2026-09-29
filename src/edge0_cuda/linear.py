import torch

from .quant import unpack_int4_u32, dequantize_affine_int4
from .lora import LoRALinear


class QuantizedLinear:
    """
    INT4 affine quantized linear layer.

    Stored checkpoint weight:
        [out_features, packed_in_features]

    Logical weight after unpacking:
        [out_features, in_features]
    """

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        biases: torch.Tensor,
        group_size: int = 64,
    ):
        self.packed_weight = packed_weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size

        self._weight = None

    def weight(self) -> torch.Tensor:
        if self._weight is None:
            self._weight = dequantize_affine_int4(
                self.packed_weight,
                self.scales,
                self.biases,
                group_size=self.group_size,
            )

        return self._weight

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.weight()

        return x.float() @ weight.float().t()


class QuantizedLoRALinear:
    """
    INT4 base linear + unmerged Recover-LoRA.

    Output:

        base(x) + (alpha / rank) * LoRA(x)
    """

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        biases: torch.Tensor,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        group_size: int = 64,
        alpha: float = 32.0,
        rank: int = 16,
    ):
        self.base = QuantizedLinear(
            packed_weight=packed_weight,
            scales=scales,
            biases=biases,
            group_size=group_size,
        )

        self.lora = LoRALinear(
            base=self.base,
            lora_a=lora_a,
            lora_b=lora_b,
            alpha=alpha,
            rank=rank,
        )

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.lora(x)

    def weight(self) -> torch.Tensor:
        return self.base.weight()
