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
    [[100, 200, 300, 400, 500]],
    dtype=torch.long,
)

print("FULL 5-TOKEN")

full_logits = model.logits(tokens)
full_last = full_logits[:, -1]

print("FULL:", full_last.shape)

print("CACHED")

prefill_logits, states = model.prefill(tokens[:, :4])

decode_logits, states = model.decode(
    tokens[:, 4:5],
    states,
)

cached_last = decode_logits[:, -1]

diff = (full_last - cached_last).abs()

print("CACHED:", cached_last.shape)
print("MAX DIFF:", diff.max().item())
print("MEAN DIFF:", diff.mean().item())
print(
    "COSINE:",
    torch.nn.functional.cosine_similarity(
        full_last,
        cached_last,
        dim=-1,
    ).item(),
)

print(
    "FINITE:",
    torch.isfinite(cached_last).all().item(),
)

if diff.max().item() < 2e-3:
    print("MLA/KDA CACHE VALIDATION PASSED")
else:
    print("MLA/KDA CACHE VALIDATION FAILED")
