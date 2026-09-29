import torch

from edge0_cuda.mmap_store import SafetensorsMmap
from edge0_cuda.expert import Int4Expert


MODEL_DIR = "models/edge0-8b"


store = SafetensorsMmap(MODEL_DIR)

layer = 1
expert_id = 0

prefix = f"model.layers.{layer}.mlp.experts."


def read_u32(name, shape):
    raw = store.read_dim0_slice(
        prefix + name,
        expert_id,
    )

    return torch.frombuffer(
        bytearray(raw),
        dtype=torch.uint32,
    ).reshape(shape)


def read_bf16(name, shape):
    raw = store.read_dim0_slice(
        prefix + name,
        expert_id,
    )

    return torch.frombuffer(
        bytearray(raw),
        dtype=torch.bfloat16,
    ).reshape(shape)


print()
print("=" * 60)
print("EDGE0 FULL INT4 EXPERT TEST")
print("=" * 60)


# ------------------------------------------------------------
# Gate projection
# ------------------------------------------------------------

gate_weight = read_u32(
    "gate_proj.weight",
    (512, 192),
)

gate_scales = read_bf16(
    "gate_proj.scales",
    (512, 24),
)

gate_biases = read_bf16(
    "gate_proj.biases",
    (512, 24),
)


# ------------------------------------------------------------
# Up projection
# ------------------------------------------------------------

up_weight = read_u32(
    "up_proj.weight",
    (512, 192),
)

up_scales = read_bf16(
    "up_proj.scales",
    (512, 24),
)

up_biases = read_bf16(
    "up_proj.biases",
    (512, 24),
)


# ------------------------------------------------------------
# Down projection
# ------------------------------------------------------------

down_weight = read_u32(
    "down_proj.weight",
    (1536, 64),
)

down_scales = read_bf16(
    "down_proj.scales",
    (1536, 8),
)

down_biases = read_bf16(
    "down_proj.biases",
    (1536, 8),
)


# ------------------------------------------------------------
# Dequantize
# ------------------------------------------------------------

gate = Int4Expert(
    gate_weight,
    gate_scales,
    gate_biases,
).dequantize()


up = Int4Expert(
    up_weight,
    up_scales,
    up_biases,
).dequantize()


down = Int4Expert(
    down_weight,
    down_scales,
    down_biases,
).dequantize()


print()
print("Dequantized weights:")
print("  gate:", gate.shape)
print("  up:  ", up.shape)
print("  down:", down.shape)


# ------------------------------------------------------------
# Execute expert
# ------------------------------------------------------------

x = torch.randn(
    2,
    1536,
    dtype=torch.float32,
)


gate_output = x @ gate.t()

up_output = x @ up.t()

hidden = (
    torch.nn.functional.silu(
        gate_output
    )
    * up_output
)

output = hidden @ down.t()


print()
print("Expert execution:")
print("  input: ", x.shape)
print("  gate:  ", gate_output.shape)
print("  up:    ", up_output.shape)
print("  hidden:", hidden.shape)
print("  output:", output.shape)


print()
print("Output statistics:")
print("  min :", output.min().item())
print("  max :", output.max().item())
print("  mean:", output.mean().item())
print("  std :", output.std().item())


print()
print("=" * 60)
print("FULL EXPERT OK")
print("=" * 60)
