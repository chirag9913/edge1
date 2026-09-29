import torch
import torch.nn.functional as F

from .linear import QuantizedLinear


class SharedExpert:
    """
    Edge0 shared SwiGLU expert.

    Architecture:

        gate_proj(x)
             \
              -> SiLU(gate) * up_proj(x)
             /
        down_proj
    """

    def __init__(
        self,
        gate_proj,
        up_proj,
        down_proj,
    ):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    @torch.no_grad()
    def __call__(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        hidden = F.silu(gate.float()) * up.float()

        return self.down_proj(hidden)
