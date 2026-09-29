import torch


class LoRALinear:
    """
    Unmerged Recover-LoRA wrapper.

    Edge0 semantics:

        y = base(x) + (alpha / r) * ((x @ A.T) @ B.T)

    The base weight remains untouched.
    """

    def __init__(
        self,
        base,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        alpha: float = 32.0,
        rank: int = 16,
    ):
        if lora_a.ndim != 2:
            raise ValueError(
                f"LoRA A must be 2D, got {lora_a.shape}"
            )

        if lora_b.ndim != 2:
            raise ValueError(
                f"LoRA B must be 2D, got {lora_b.shape}"
            )

        if lora_a.shape[0] != rank:
            raise ValueError(
                f"LoRA A rank mismatch: "
                f"{lora_a.shape[0]} != {rank}"
            )

        if lora_b.shape[1] != rank:
            raise ValueError(
                f"LoRA B rank mismatch: "
                f"{lora_b.shape[1]} != {rank}"
            )

        self.base = base
        self.lora_A = lora_a
        self.lora_B = lora_b

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        y = self.base(x)

        adapter_dtype = self.lora_A.dtype

        if x.dtype == adapter_dtype:
            adapter_input = x
        else:
            adapter_input = x.to(adapter_dtype)

        delta = (
            adapter_input @ self.lora_A.t()
        ) @ self.lora_B.t()

        return y + (
            self.scale * delta
        ).to(y.dtype)
