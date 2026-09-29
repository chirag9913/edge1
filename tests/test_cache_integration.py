import torch

from edge0_cuda.mmap_store import SafetensorsMmap
from edge0_cuda.router import Edge0Router
from edge0_cuda.moe import StreamingMoE


store = SafetensorsMmap(
    "models/edge0-8b"
)

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
    cache_size=32,
)

torch.manual_seed(12345)

x = torch.randn(
    4,
    1536,
)

print()
print("=" * 60)
print("EXPERT CACHE INTEGRATION TEST")
print("=" * 60)

print()
print("Initial cache:")
print(moe.cache_stats())

print()
print("Running first MoE call...")
moe(x)

print("After first call:")
print("  stats:", moe.cache_stats())
print("  cached:", moe.cached_experts())

print()
print("Running second MoE call...")
moe(x)

print("After second call:")
print("  stats:", moe.cache_stats())
print("  cached:", moe.cached_experts())

print()
print("=" * 60)

if moe.cache_stats()["hits"] > 0:
    print("EXPERT CACHE INTEGRATION OK")
else:
    raise RuntimeError(
        "No cache hits detected"
    )

print("=" * 60)

