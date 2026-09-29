import torch

from edge0_cuda.model import Edge0Model
from edge0_cuda.layer0 import build_layer0

model = Edge0Model(
    model_path=r".\models\edge0-8b",
    layer_factory=lambda **kwargs: None,
)

model.load()

layer = build_layer0(model.store)

input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

hidden = model.embedding(input_ids)

print("INPUT:", hidden.shape, hidden.dtype)

output = layer(hidden)

print("OUTPUT:", output.shape, output.dtype)
print("FINITE:", torch.isfinite(output).all().item())
print("MIN:", output.min().item())
print("MAX:", output.max().item())
print("MEAN:", output.mean().item())
print("STD:", output.std().item())

model.close()
