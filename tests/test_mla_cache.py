import sys
sys.path.insert(0, r".\src")

import torch

from edge0_cuda.model import Edge0Model
from edge0_cuda.layer_factory import rotary_cos_sin, layer_factory

model = Edge0Model(
    r".\models\edge0-8b",
    layer_factory,
    cache_size=4,
)

layer = model._build_layer(3)
mla = layer.attention

tokens = torch.tensor(
    [[100, 200, 300, 400, 500]],
    dtype=torch.long,
)

with torch.no_grad():
    hidden = model.embedding(tokens)

    x = layer.input_layernorm(hidden)

    x4 = x[:, :4]
    x5 = x[:, 4:5]

    cos4, sin4 = rotary_cos_sin(
        4,
        device=x.device,
        dtype=x.dtype,
        start_position=0,
    )

    full_cos, full_sin = rotary_cos_sin(
        5,
        device=x.device,
        dtype=x.dtype,
        start_position=0,
    )

    print("FULL MLA")

    full_out = mla(
        x,
        full_cos,
        full_sin,
    )

    print("FULL OUTPUT:", full_out.shape)

    print("CACHED MLA")

    cached_out, state = mla(
        x4,
        cos4,
        sin4,
        initial_state=None,
        return_state=True,
    )

    cos1, sin1 = rotary_cos_sin(
        1,
        device=x.device,
        dtype=x.dtype,
        start_position=4,
    )

    decode_out, state2 = mla(
        x5,
        cos1,
        sin1,
        initial_state=state,
        return_state=True,
    )

    full_last = full_out[:, -1]
    cached_last = decode_out[:, -1]

    diff = (full_last - cached_last).abs()

    print("CACHED OUTPUT:", decode_out.shape)
    print("CACHE POSITION:", state2["position"])
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

    if diff.max().item() < 1e-3:
        print("MLA CACHE PASSED")
    else:
        print("MLA CACHE FAILED")
