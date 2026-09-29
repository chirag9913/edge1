import torch

from edge0_cuda.mmap_store import SafetensorsMmap
from edge0_cuda.router import Edge0Router
from edge0_cuda.moe import StreamingMoE


store = SafetensorsMmap(
    "models/edge0-8b"
)


# Load real layer-1 router.
router_weight = torch.frombuffer(
    bytearray(
        store.read_raw(
            "model.layers.1.mlp.gate.weight"
        )
    ),
    dtype=torch.bfloat16,
).reshape(128, 1536)


expert_bias = torch.frombuffer(
    bytearray(
        store.read_raw(
            "model.layers.1.mlp.gate.expert_bias"
        )
    ),
    dtype=torch.bfloat16,
).reshape(128)


router = Edge0Router(
    router_weight,
    expert_bias,
)


moe = StreamingMoE(
    store=store,
    router=router,
    layer_idx=1,
)


torch.manual_seed(12345)

x = torch.randn(
    4,
    1536,
)


print()
print("=" * 60)
print("STREAMING MOE TEST")
print("=" * 60)

print()
print("Input:", x.shape)


output, idx, weights, logits = moe(x)


print()
print("Router:")
print("  experts:", idx.shape)
print("  weights:", weights.shape)
print("  logits :", logits.shape)


print()
print("MoE:")
print("  output:", output.shape)
print("  mean:  ", output.mean().item())
print("  std:   ", output.std().item())
print("  min:   ", output.min().item())
print("  max:   ", output.max().item())


print()
print("Selected experts:")
print(idx)


print()
print("=" * 60)
print("STREAMING MOE OK")
print("=" * 60)
