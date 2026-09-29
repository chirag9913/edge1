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

prompt = torch.tensor(
    [[100, 200, 300, 400]],
    dtype=torch.long,
)

print("PREFILL")

logits, states = model.prefill(prompt)

print("LOGITS:", logits.shape)
print("STATE LAYERS:", sorted(states.keys()))

for layer_idx in [1, 2, 3, 7, 11, 23]:
    state = states[layer_idx]

    if state is None:
        print(f"Layer {layer_idx}: NO STATE")
        continue

    if "kda" in state:
        print(
            f"Layer {layer_idx}: KimiDelta",
            state["kda"].shape,
            "q_conv:",
            state["q_conv"].shape,
        )
    elif "key" in state:
        print(
            f"Layer {layer_idx}: MLA",
            "key:",
            state["key"].shape,
            "value:",
            state["value"].shape,
            "position:",
            state["position"],
        )
    else:
        raise RuntimeError(
            f"Unknown state format for layer {layer_idx}: {state.keys()}"
        )

next_token = logits[:, -1].argmax(dim=-1)

print("NEXT TOKEN:", next_token.tolist())

print("DECODE")

next_logits, next_states = model.decode(
    next_token[:, None],
    states,
)

print("NEXT LOGITS:", next_logits.shape)
print(
    "FINITE:",
    torch.isfinite(next_logits).all().item(),
)

print("STATE UPDATE OK")
