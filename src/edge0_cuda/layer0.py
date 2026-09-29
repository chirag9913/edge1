import torch

from .kimi_delta import KimiDeltaAttention
from .linear import QuantizedLoRALinear
from .mla import RMSNorm


class DenseMLP:
    def __init__(self, gate_proj, up_proj, down_proj):
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    @torch.no_grad()
    def __call__(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = torch.nn.functional.silu(gate) * up
        return self.down_proj(hidden)


class Edge0Layer0:
    def __init__(self, input_layernorm, attention, post_attention_layernorm, mlp):
        self.input_layernorm = input_layernorm
        self.attention = attention
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp

    @torch.no_grad()
    def __call__(self, hidden_states, attention_state=None, return_state=False):
        residual = hidden_states

        x = self.input_layernorm(hidden_states)

        if return_state:
            attn_output, attention_state = self.attention(
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

        hidden_states = residual + attn_output

        residual = hidden_states

        x = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(x)

        hidden_states = residual + mlp_output

        if return_state:
            return hidden_states, attention_state

        return hidden_states


def _read(store, name, dtype, shape):
    raw = store.read_raw(name)
    return torch.frombuffer(
        bytearray(raw),
        dtype=dtype,
    ).clone().reshape(shape)


def _linear(store, prefix, in_features, out_features):
    return QuantizedLoRALinear(
        _read(
            store,
            prefix + ".weight",
            torch.uint32,
            (out_features, in_features // 8),
        ),
        _read(
            store,
            prefix + ".scales",
            torch.bfloat16,
            (out_features, in_features // 64),
        ),
        _read(
            store,
            prefix + ".biases",
            torch.bfloat16,
            (out_features, in_features // 64),
        ),
        _read(
            store,
            prefix + ".lora_A",
            torch.bfloat16,
            (16, in_features),
        ),
        _read(
            store,
            prefix + ".lora_B",
            torch.bfloat16,
            (out_features, 16),
        ),
    )


def build_layer0(store):
    prefix = "model.layers.0"

    input_layernorm = RMSNorm(
        _read(
            store,
            prefix + ".input_layernorm.weight",
            torch.bfloat16,
            (1536,),
        ),
        eps=1e-6,
    )

    attention = KimiDeltaAttention(
        q_proj=_linear(
            store,
            prefix + ".attention.q_proj",
            1536,
            2048,
        ),
        k_proj=_linear(
            store,
            prefix + ".attention.k_proj",
            1536,
            2048,
        ),
        v_proj=_linear(
            store,
            prefix + ".attention.v_proj",
            1536,
            2048,
        ),
        q_conv=_read(
            store,
            prefix + ".attention.q_conv1d.weight",
            torch.bfloat16,
            (2048, 1, 4),
        ),
        k_conv=_read(
            store,
            prefix + ".attention.k_conv1d.weight",
            torch.bfloat16,
            (2048, 1, 4),
        ),
        v_conv=_read(
            store,
            prefix + ".attention.v_conv1d.weight",
            torch.bfloat16,
            (2048, 1, 4),
        ),
        f_proj=_linear(
            store,
            prefix + ".attention.f_proj",
            1536,
            2048,
        ),
        b_proj=_linear(
            store,
            prefix + ".attention.b_proj",
            1536,
            16,
        ),
        g_proj=_linear(
            store,
            prefix + ".attention.g_proj",
            1536,
            2048,
        ),
        o_norm=_read(
            store,
            prefix + ".attention.o_norm.weight",
            torch.bfloat16,
            (128,),
        ),
        o_proj=_linear(
            store,
            prefix + ".attention.o_proj",
            2048,
            1536,
        ),
        A_log=_read(
            store,
            prefix + ".attention.A_log",
            torch.float32,
            (8,),
        ),
        dt_bias=_read(
            store,
            prefix + ".attention.dt_bias",
            torch.float32,
            (1024,),
        ),
        num_heads=16,
        head_dim=128,
        rms_norm_eps=1e-6,
    )

    post_attention_layernorm = RMSNorm(
        _read(
            store,
            prefix + ".post_attention_layernorm.weight",
            torch.bfloat16,
            (1536,),
        ),
        eps=1e-6,
    )

    mlp = DenseMLP(
        gate_proj=_linear(
            store,
            prefix + ".mlp.gate_proj",
            1536,
            4608,
        ),
        up_proj=_linear(
            store,
            prefix + ".mlp.up_proj",
            1536,
            4608,
        ),
        down_proj=_linear(
            store,
            prefix + ".mlp.down_proj",
            4608,
            1536,
        ),
    )

    return Edge0Layer0(
        input_layernorm,
        attention,
        post_attention_layernorm,
        mlp,
    )






