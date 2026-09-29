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

layer = model._build_layer(1)
attention = layer.attention

tokens = torch.tensor(
    [[100, 200, 300, 400, 500]],
    dtype=torch.long,
)

with torch.no_grad():
    hidden = model.embedding(tokens)
    x = layer.input_layernorm(hidden)

    full_out = attention(
        x,
        return_state=False,
    )

    prefill_out, state = attention(
        x[:, :4],
        return_state=True,
    )

    decode_out, state2 = attention(
        x[:, 4:5],
        initial_state=state,
        return_state=True,
    )

    full_last = full_out[:, -1]
    cached_last = decode_out[:, -1]

    diff = (full_last - cached_last).abs()

    print("FULL:", full_last.shape)
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

    print("STATE POSITION:", state2["kda"].shape)

    if diff.max().item() < 1e-3:
        print("KIMI DELTA CACHE PASSED")
    else:
        print("KIMI DELTA CACHE FAILED")
