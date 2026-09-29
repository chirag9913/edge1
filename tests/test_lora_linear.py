import sys

import torch

sys.path.insert(0, r".\src")

from edge0_cuda.mmap_store import SafetensorsMmap
from edge0_cuda.linear import QuantizedLoRALinear


MODEL_DIR = r".\models\edge0-8b"

BASE = "model.layers.3.attention.q_b_proj"
LORA_A = BASE + ".lora_A"
LORA_B = BASE + ".lora_B"


def read_u32(store, name, shape):
    raw = store.read_raw(name)

    return torch.frombuffer(
        bytearray(raw),
        dtype=torch.uint32,
    ).reshape(shape)


def read_bf16(store, name, shape):
    raw = store.read_raw(name)

    return torch.frombuffer(
        bytearray(raw),
        dtype=torch.bfloat16,
    ).reshape(shape)


def read_f16(store, name, shape):
    raw = store.read_raw(name)

    return torch.frombuffer(
        bytearray(raw),
        dtype=torch.float16,
    ).reshape(shape)


print()
print("=" * 80)
print("LAYER 3 q_b_proj + LoRA TEST")
print("=" * 80)
print()

store = SafetensorsMmap(MODEL_DIR)

packed_weight = read_u32(
    store,
    BASE + ".weight",
    (3072, 32),
)

scales = read_bf16(
    store,
    BASE + ".scales",
    (3072, 4),
)

biases = read_bf16(
    store,
    BASE + ".biases",
    (3072, 4),
)

lora_a = read_f16(
    store,
    LORA_A,
    (16, 256),
)

lora_b = read_f16(
    store,
    LORA_B,
    (3072, 16),
)

print("Checkpoint tensors:")
print("  packed weight :", packed_weight.shape)
print("  scales        :", scales.shape)
print("  biases        :", biases.shape)
print("  LoRA A        :", lora_a.shape)
print("  LoRA B        :", lora_b.shape)
print()

layer = QuantizedLoRALinear(
    packed_weight=packed_weight,
    scales=scales,
    biases=biases,
    lora_a=lora_a,
    lora_b=lora_b,
    group_size=64,
    alpha=32,
    rank=16,
)

print("LoRA scale:", layer.lora.scale)

x = torch.randn(
    2,
    256,
    dtype=torch.float32,
)

with torch.no_grad():
    output = layer(x)

print()
print("Input :", x.shape)
print("Output:", output.shape)
print()

print("Output statistics:")
print("  min :", float(output.min()))
print("  max :", float(output.max()))
print("  mean:", float(output.mean()))
print("  std :", float(output.std()))

assert output.shape == (2, 3072)
assert layer.lora.scale == 2.0

print()
print("=" * 80)
print("LAYER 3 q_b_proj + LoRA TEST PASSED")
print("=" * 80)

store.close()
