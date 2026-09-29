import numpy as np
import torch

from .linear import QuantizedLinear, QuantizedLoRALinear
from .kimi_delta import KimiDeltaAttention
from .mla import MLA, RMSNorm
from .router import Edge0Router
from .moe import StreamingMoE
from .shared_expert import SharedExpert

HIDDEN = 1536
NUM_HEADS = 16
HEAD_DIM = 128
QK_HEAD_DIM = 192
QK_NOPE = 128
QK_ROPE = 64
V_HEAD_DIM = 128
KV_LORA_RANK = 512

NP_DTYPES = {
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "U8": np.uint8,
    "I8": np.int8,
    "U16": np.uint16,
    "I16": np.int16,
    "U32": np.uint32,
    "I32": np.int32,
    "U64": np.uint64,
    "I64": np.int64,
}


def read_tensor(store, name):
    if not store.has(name):
        raise KeyError(name)

    info = store.info(name)
    raw = store.read_raw(name)
    dtype_name = info["dtype"]

    if dtype_name == "BF16":
        array = np.frombuffer(raw, dtype=np.uint16).copy()
        tensor = torch.from_numpy(array.astype(np.int16)).view(torch.bfloat16)
    elif dtype_name in NP_DTYPES:
        array = np.frombuffer(raw, dtype=NP_DTYPES[dtype_name]).copy()
        if dtype_name == "U32":
            array = array.astype(np.int64)
        tensor = torch.from_numpy(array)
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    return tensor.reshape(info["shape"])


def make_linear(store, prefix):
    weight = read_tensor(store, prefix + ".weight")
    scales = read_tensor(store, prefix + ".scales")
    biases = read_tensor(store, prefix + ".biases")

    has_lora = (
        store.has(prefix + ".lora_A")
        and store.has(prefix + ".lora_B")
    )

    if has_lora:
        return QuantizedLoRALinear(
            weight,
            scales,
            biases,
            read_tensor(store, prefix + ".lora_A"),
            read_tensor(store, prefix + ".lora_B"),
            group_size=64,
            alpha=32.0,
            rank=16,
        )

    return QuantizedLinear(
        weight,
        scales,
        biases,
        group_size=64,
    )


def build_kimi_delta(store, layer_idx):
    prefix = f"model.layers.{layer_idx}.attention."

    return KimiDeltaAttention(
        q_proj=make_linear(store, prefix + "q_proj"),
        k_proj=make_linear(store, prefix + "k_proj"),
        v_proj=make_linear(store, prefix + "v_proj"),
        q_conv=read_tensor(store, prefix + "q_conv1d.weight"),
        k_conv=read_tensor(store, prefix + "k_conv1d.weight"),
        v_conv=read_tensor(store, prefix + "v_conv1d.weight"),
        f_proj=make_linear(store, prefix + "f_proj"),
        b_proj=make_linear(store, prefix + "b_proj"),
        g_proj=make_linear(store, prefix + "g_proj"),
        o_norm=read_tensor(store, prefix + "o_norm.weight"),
        o_proj=make_linear(store, prefix + "o_proj"),
        A_log=read_tensor(store, prefix + "A_log"),
        dt_bias=read_tensor(store, prefix + "dt_bias"),
        num_heads=16,
        head_dim=128,
        rms_norm_eps=1e-6,
    )


def build_mla(store, layer_idx):
    prefix = f"model.layers.{layer_idx}.attention."

    return MLA(
        q_a_proj=make_linear(store, prefix + "q_a_proj"),
        q_a_layernorm=RMSNorm(
            read_tensor(store, prefix + "q_a_layernorm.weight"),
            eps=1e-6,
        ),
        q_b_proj=make_linear(store, prefix + "q_b_proj"),
        kv_a_proj_with_mqa=make_linear(
            store,
            prefix + "kv_a_proj_with_mqa",
        ),
        kv_a_layernorm=RMSNorm(
            read_tensor(store, prefix + "kv_a_layernorm.weight"),
            eps=1e-6,
        ),
        kv_b_proj=make_linear(store, prefix + "kv_b_proj"),
        g_proj=make_linear(store, prefix + "g_proj"),
        dense=make_linear(store, prefix + "dense"),
        num_heads=16,
        qk_head_dim=192,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
    )


def build_moe(store, layer_idx, cache_size):
    prefix = f"model.layers.{layer_idx}.mlp.gate"

    router_weight = read_tensor(
        store,
        prefix + ".weight",
    )

    expert_bias = read_tensor(
        store,
        prefix + ".expert_bias",
    )

    router = Edge0Router(
        router_weight,
        expert_bias,
    )

    return StreamingMoE(
        store,
        router,
        layer_idx,
        cache_size=cache_size,
    )


def build_shared_expert(store, layer_idx):
    prefix = f"model.layers.{layer_idx}.mlp.shared_experts"

    return SharedExpert(
        make_linear(store, prefix + ".gate_proj"),
        make_linear(store, prefix + ".up_proj"),
        make_linear(store, prefix + ".down_proj"),
    )


def rotary_cos_sin(
    sequence_length,
    device,
    dtype,
    start_position=0,
):
    position = torch.arange(
        start_position,
        start_position + sequence_length,
        device=device,
        dtype=torch.float32,
    )

    inv_freq = 1.0 / (
        6000000 ** (
            torch.arange(
                0,
                64,
                2,
                device=device,
                dtype=torch.float32,
            ) / 64
        )
    )

    freqs = torch.outer(position, inv_freq)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)

    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    return cos[None, None], sin[None, None]


class Edge0MoELayer:
    def __init__(
        self,
        input_layernorm,
        attention,
        post_attention_layernorm,
        moe,
        shared_expert,
        mla=False,
    ):
        self.input_layernorm = input_layernorm
        self.attention = attention
        self.post_attention_layernorm = post_attention_layernorm
        self.moe = moe
        self.shared_expert = shared_expert
        self.mla = mla

    @torch.no_grad()
    def __call__(
        self,
        hidden_states,
        attention_state=None,
        return_state=False,
    ):
        residual = hidden_states

        x = self.input_layernorm(hidden_states)

        if self.mla:
            start_position = 0

            if attention_state is not None:
                start_position = int(
                    attention_state["position"]
                )

            cos, sin = rotary_cos_sin(
                x.shape[1],
                device=x.device,
                dtype=x.dtype,
                start_position=start_position,
            )

            if return_state:
                attn_output, next_state = self.attention(
                    x,
                    cos,
                    sin,
                    initial_state=attention_state,
                    return_state=True,
                )
            else:
                attn_output = self.attention(
                    x,
                    cos,
                    sin,
                    initial_state=attention_state,
                    return_state=False,
                )
                next_state = attention_state

        else:
            if return_state:
                attn_output, next_state = self.attention(
                    x,
                    initial_state=attention_state,
                    return_state=True,
                )
            else:
                attn_output = self.attention(
                    x,
                    initial_state=attention_state,
                    return_state=False,
                )
                next_state = attention_state

        hidden_states = residual + attn_output

        residual = hidden_states

        x = self.post_attention_layernorm(hidden_states)

        batch_size, sequence_length, hidden_size = x.shape

        moe_input = x.reshape(
            batch_size * sequence_length,
            hidden_size,
        )

        routed, topk_idx, topk_weight, router_logits = self.moe(
            moe_input
        )

        shared = self.shared_expert(moe_input)

        routed = routed.reshape(
            batch_size,
            sequence_length,
            hidden_size,
        )

        shared = shared.reshape(
            batch_size,
            sequence_length,
            hidden_size,
        )

        hidden_states = residual + routed + shared

        if return_state:
            return (
                hidden_states,
                next_state,
                topk_idx,
                topk_weight,
                router_logits,
            )

        return hidden_states


def build_layer(store, layer_idx, cache_size=4):
    if layer_idx < 1 or layer_idx > 23:
        raise ValueError(
            "MoE layer index must be between 1 and 23"
        )

    input_layernorm = RMSNorm(
        read_tensor(
            store,
            f"model.layers.{layer_idx}.input_layernorm.weight",
        ),
        eps=1e-6,
    )

    post_attention_layernorm = RMSNorm(
        read_tensor(
            store,
            f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        ),
        eps=1e-6,
    )

    if (layer_idx + 1) % 4 == 0:
        attention = build_mla(
            store,
            layer_idx,
        )
        mla = True
    else:
        attention = build_kimi_delta(
            store,
            layer_idx,
        )
        mla = False

    moe = build_moe(
        store,
        layer_idx,
        cache_size,
    )

    shared_expert = build_shared_expert(
        store,
        layer_idx,
    )

    return Edge0MoELayer(
        input_layernorm,
        attention,
        post_attention_layernorm,
        moe,
        shared_expert,
        mla=mla,
    )


def layer_factory(store, layer_idx, cache_size=4):
    return build_layer(
        store,
        layer_idx,
        cache_size,
    )
