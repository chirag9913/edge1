import torch

from edge0_cuda.mmap_store import SafetensorsMmap
from edge0_cuda.router import Edge0Router
from edge0_cuda.moe import StreamingMoE


torch.manual_seed(12345)

store = SafetensorsMmap(
    "models/edge0-8b"
)


# ------------------------------------------------------------
# Real router
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Streaming implementation
# ------------------------------------------------------------

moe = StreamingMoE(
    store=store,
    router=router,
    layer_idx=1,
)


# Same deterministic input.
x = torch.randn(
    4,
    1536,
)


# Run our implementation.
our_output, our_idx, our_weights, _ = moe(x)


# ------------------------------------------------------------
# Reference MoE
#
# Uses the SAME router and the SAME expert execution,
# but performs the accumulation explicitly.
# ------------------------------------------------------------

reference_output = torch.zeros(
    4,
    1536,
    dtype=torch.float32,
)


for token in range(x.shape[0]):

    for slot in range(8):

        expert_idx = int(
            our_idx[token, slot].item()
        )

        weight = our_weights[
            token,
            slot,
        ]

        prefix = (
            f"model.layers.1.mlp.experts."
        )

        def read_u32(name, shape):
            raw = store.read_dim0_slice(
                prefix + name,
                expert_idx,
            )

            return torch.frombuffer(
                bytearray(raw),
                dtype=torch.uint32,
            ).reshape(shape)

        def read_bf16(name, shape):
            raw = store.read_dim0_slice(
                prefix + name,
                expert_idx,
            )

            return torch.frombuffer(
                bytearray(raw),
                dtype=torch.bfloat16,
            ).reshape(shape)

        from edge0_cuda.expert import Int4Expert

        gate = Int4Expert(
            read_u32(
                "gate_proj.weight",
                (512, 192),
            ),
            read_bf16(
                "gate_proj.scales",
                (512, 24),
            ),
            read_bf16(
                "gate_proj.biases",
                (512, 24),
            ),
        ).dequantize()

        up = Int4Expert(
            read_u32(
                "up_proj.weight",
                (512, 192),
            ),
            read_bf16(
                "up_proj.scales",
                (512, 24),
            ),
            read_bf16(
                "up_proj.biases",
                (512, 24),
            ),
        ).dequantize()

        down = Int4Expert(
            read_u32(
                "down_proj.weight",
                (1536, 64),
            ),
            read_bf16(
                "down_proj.scales",
                (1536, 8),
            ),
            read_bf16(
                "down_proj.biases",
                (1536, 8),
            ),
        ).dequantize()

        hidden = (
            torch.nn.functional.silu(
                x[token:token + 1] @ gate.t()
            )
            *
            (
                x[token:token + 1]
                @ up.t()
            )
        )

        expert_output = (
            hidden @ down.t()
        )

        reference_output[token] += (
            expert_output[0] * weight
        )


# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

difference = torch.abs(
    our_output - reference_output
)

max_difference = difference.max().item()
mean_difference = difference.mean().item()


print()
print("=" * 60)
print("STREAMING MOE VALIDATION")
print("=" * 60)

print()
print("Our output:       ", our_output.shape)
print("Reference output: ", reference_output.shape)

print()
print("Maximum difference:", max_difference)
print("Mean difference:   ", mean_difference)

print()

if torch.allclose(
    our_output,
    reference_output,
    rtol=1e-5,
    atol=5e-6,
):
    print("STREAMING MOE VALIDATION PASSED")
else:
    print("STREAMING MOE VALIDATION FAILED")
    raise RuntimeError(
        "Streaming MoE does not match reference"
    )

print("=" * 60)

