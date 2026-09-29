import sys
sys.path.insert(0, r".\src")

import torch

from edge0_cuda.model import Edge0Model
from edge0_cuda.layer_factory import layer_factory

MODEL_DIR = r".\models\edge0-8b"

print("=" * 60)
print("EDGE0 FULL 24-LAYER FORWARD")
print("=" * 60)

model = Edge0Model(
    model_path=MODEL_DIR,
    layer_factory=layer_factory,
    cache_size=4,
)

input_ids = torch.tensor([[100, 200, 300, 400]], dtype=torch.long)

print("INPUT:", input_ids.shape)

with torch.no_grad():
    hidden = model.forward(input_ids)

print("OUTPUT:", hidden.shape)
print("DTYPE:", hidden.dtype)
print("FINITE:", torch.isfinite(hidden).all().item())
print("MIN:", hidden.min().item())
print("MAX:", hidden.max().item())
print("MEAN:", hidden.mean().item())
print("STD:", hidden.std().item())

if hidden.shape != (1, 4, 1536):
    raise RuntimeError(f"Unexpected output shape: {hidden.shape}")

if not torch.isfinite(hidden).all():
    raise RuntimeError("Non-finite values detected")

print("=" * 60)
print("FULL 24-LAYER FORWARD PASSED")
print("=" * 60)
