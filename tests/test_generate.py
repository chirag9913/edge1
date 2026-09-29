import sys
sys.path.insert(0, r".\src")

import torch

from edge0_cuda.model import Edge0Model
from edge0_cuda.layer_factory import layer_factory

model = Edge0Model(
    r".\models\edge0-8b",
    layer_factory,
    cache_size=4,
)

tokens = torch.tensor(
    [[100, 200, 300, 400]],
    dtype=torch.long,
)

print("GENERATING...")

output = model.generate(
    tokens,
    max_new_tokens=4,
    temperature=0.0,
)

print("OUTPUT:", output)
print("SHAPE:", output.shape)
print("FINITE:", torch.isfinite(output.float()).all().item())
