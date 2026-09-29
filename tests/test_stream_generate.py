import sys
import time

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

print("STREAM:")

for token_id in model.generate_stream(
    tokens,
    max_new_tokens=8,
    temperature=0.0,
):
    print(token_id, end=" ", flush=True)
    time.sleep(0.05)

print()
