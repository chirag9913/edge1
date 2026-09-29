import torch

from edge0_cuda.mmap_store import SafetensorsMmap
from edge0_cuda.router import Edge0Router


MODEL_DIR = "models/edge0-8b"

store = SafetensorsMmap(MODEL_DIR)

layer = 1

weight_name = (
    f"model.layers.{layer}.mlp.gate.weight"
)

bias_name = (
    f"model.layers.{layer}.mlp.gate.expert_bias"
)


def read_bf16(name, shape):
    raw = store.read_raw(name)

    return torch.frombuffer(
        bytearray(raw),
        dtype=torch.bfloat16,
    ).reshape(shape)


router_weight = read_bf16(
    weight_name,
    (128, 1536),
)

expert_bias = read_bf16(
    bias_name,
    (128,),
)


router = Edge0Router(
    router_weight,
    expert_bias,
)


# Fixed input so results are reproducible.
torch.manual_seed(12345)

hidden_states = torch.randn(
    4,
    1536,
    dtype=torch.float32,
)


# ------------------------------------------------------------
# OUR ROUTER
# ------------------------------------------------------------

our_idx, our_weights, our_logits = router(
    hidden_states
)


# ------------------------------------------------------------
# REFERENCE IMPLEMENTATION
#
# Directly mirrors the BailingMoeV3Gate source
# you provided earlier.
# ------------------------------------------------------------

def reference_router(
    hidden_states,
    weight,
    expert_bias,
):
    hidden_states = hidden_states.view(
        -1,
        hidden_states.shape[-1],
    )

    logits = torch.nn.functional.linear(
        hidden_states.type(torch.float32),
        weight.type(torch.float32),
    )

    scores = torch.sigmoid(
        logits.float()
    ).type_as(logits)

    scores_for_routing = (
        scores + expert_bias
    )

    num_tokens = scores.shape[0]
    num_experts = 128
    n_group = 8
    topk_group = 4
    top_k = 8

    experts_per_group = (
        num_experts // n_group
    )

    group_scores = (
        scores_for_routing
        .view(
            num_tokens,
            n_group,
            -1,
        )
        .topk(
            2,
            dim=-1,
        )[0]
        .sum(dim=-1)
    )

    group_idx = torch.topk(
        group_scores,
        k=topk_group,
        dim=-1,
        sorted=False,
    )[1]

    group_mask = torch.zeros_like(
        group_scores
    )

    group_mask.scatter_(
        1,
        group_idx,
        1,
    )

    score_mask = (
        group_mask
        .unsqueeze(-1)
        .expand(
            num_tokens,
            n_group,
            experts_per_group,
        )
        .reshape(
            num_tokens,
            -1,
        )
    )

    masked_scores = scores_for_routing.masked_fill(
        ~score_mask.bool(),
        float("-inf"),
    )

    _, top_indices = torch.topk(
        masked_scores,
        k=top_k,
        dim=-1,
    )

    selected_scores = torch.gather(
        scores,
        dim=1,
        index=top_indices,
    )

    topk_weight = (
        selected_scores
        / (
            selected_scores.sum(
                dim=-1,
                keepdim=True,
            )
            + 1e-20
        )
    )

    topk_weight = (
        topk_weight * 2.5
    )

    return (
        top_indices,
        topk_weight,
        logits,
    )


ref_idx, ref_weights, ref_logits = reference_router(
    hidden_states,
    router_weight,
    expert_bias,
)


# ------------------------------------------------------------
# COMPARE
# ------------------------------------------------------------

idx_equal = torch.equal(
    our_idx,
    ref_idx,
)

weights_equal = torch.allclose(
    our_weights,
    ref_weights,
    rtol=0,
    atol=0,
)

logits_equal = torch.allclose(
    our_logits,
    ref_logits,
    rtol=0,
    atol=0,
)


print()
print("=" * 60)
print("EDGE0 ROUTER VALIDATION")
print("=" * 60)

print()
print("Expert indices:")
print("  equal:", idx_equal)

print()
print("Routing weights:")
print("  equal:", weights_equal)

print()
print("Router logits:")
print("  equal:", logits_equal)

print()
print("Our indices:")
print(our_idx)

print()
print("Reference indices:")
print(ref_idx)

print()
print("Maximum weight difference:")
print(
    torch.max(
        torch.abs(
            our_weights - ref_weights
        )
    ).item()
)

print()
print("Maximum logit difference:")
print(
    torch.max(
        torch.abs(
            our_logits - ref_logits
        )
    ).item()
)

print()

if not idx_equal:
    raise RuntimeError(
        "Router expert selection DOES NOT MATCH reference"
    )

if not weights_equal:
    raise RuntimeError(
        "Router weights DO NOT MATCH reference"
    )

if not logits_equal:
    raise RuntimeError(
        "Router logits DO NOT MATCH reference"
    )

print("=" * 60)
print("ROUTER VALIDATION PASSED")
print("=" * 60)
