import torch
import torch.nn.functional as F

from .linear import QuantizedLinear, QuantizedLoRALinear


class RMSNorm:
    def __init__(self, weight, eps=1e-6):
        self.weight = weight.float()
        self.eps = eps

    @torch.no_grad()
    def __call__(self, x):
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        output = x_float * torch.rsqrt(variance + self.eps)
        return (output * self.weight).to(x.dtype)


def rotate_half_interleaved(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_interleaved(x, cos, sin):
    return x * cos + rotate_half_interleaved(x) * sin


class MLA:
    def __init__(
        self,
        q_a_proj,
        q_a_layernorm,
        q_b_proj,
        kv_a_proj_with_mqa,
        kv_a_layernorm,
        kv_b_proj,
        g_proj,
        dense,
        num_heads=16,
        qk_head_dim=192,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
    ):
        self.q_a_proj = q_a_proj
        self.q_a_layernorm = q_a_layernorm
        self.q_b_proj = q_b_proj
        self.kv_a_proj_with_mqa = kv_a_proj_with_mqa
        self.kv_a_layernorm = kv_a_layernorm
        self.kv_b_proj = kv_b_proj
        self.g_proj = g_proj
        self.dense = dense
        self.num_heads = num_heads
        self.qk_head_dim = qk_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.scale = qk_head_dim ** -0.5

    def _project(self, hidden_states):
        batch_size, seq_length, _ = hidden_states.shape

        q_states = self.q_a_proj(hidden_states)
        q_states = self.q_a_layernorm(q_states)
        q_states = self.q_b_proj(q_states)
        q_states = q_states.reshape(
            batch_size,
            seq_length,
            self.num_heads,
            self.qk_head_dim,
        )
        q_states = q_states.permute(0, 2, 1, 3)

        q_pass, q_rot = torch.split(
            q_states,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

        k_pass_latent, k_rot = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )

        k_pass_latent = self.kv_a_layernorm(k_pass_latent)

        kv_states = self.kv_b_proj(k_pass_latent)
        kv_states = kv_states.reshape(
            batch_size,
            seq_length,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        kv_states = kv_states.permute(0, 2, 1, 3)

        k_pass, value_states = torch.split(
            kv_states,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )

        k_rot = k_rot.reshape(
            batch_size,
            1,
            seq_length,
            self.qk_rope_head_dim,
        )

        return q_pass, q_rot, k_pass, k_rot, value_states

    @torch.no_grad()
    def __call__(
        self,
        hidden_states,
        cos,
        sin,
        initial_state=None,
        return_state=False,
    ):
        if hidden_states.ndim != 3:
            raise ValueError("MLA expects [batch, sequence, hidden]")

        batch_size, seq_length, _ = hidden_states.shape

        q_pass, q_rot, k_pass, k_rot, value_states = self._project(
            hidden_states
        )

        if initial_state is None:
            past_key = None
            past_value = None
            start_position = 0
        else:
            past_key = initial_state["key"]
            past_value = initial_state["value"]
            start_position = int(initial_state["position"])

        q_rot = apply_rotary_interleaved(q_rot, cos, sin)

        position_cos = cos
        position_sin = sin

        if start_position != 0:
            position_cos = cos
            position_sin = sin

        k_rot = apply_rotary_interleaved(
            k_rot,
            position_cos,
            position_sin,
        )

        k_rot = k_rot.expand(
            batch_size,
            self.num_heads,
            seq_length,
            self.qk_rope_head_dim,
        )

        query_states = torch.cat(
            [q_pass, q_rot],
            dim=-1,
        )

        current_key = torch.cat(
            [k_pass, k_rot],
            dim=-1,
        )

        if past_key is not None:
            key_states = torch.cat(
                [past_key, current_key],
                dim=2,
            )
            all_values = torch.cat(
                [past_value, value_states],
                dim=2,
            )
        else:
            key_states = current_key
            all_values = value_states

        attention_scores = torch.matmul(
            query_states.float(),
            key_states.float().transpose(-1, -2),
        )

        attention_scores = attention_scores * self.scale

        past_length = key_states.shape[2] - seq_length

        query_positions = torch.arange(
            past_length,
            past_length + seq_length,
            device=hidden_states.device,
        )

        key_positions = torch.arange(
            key_states.shape[2],
            device=hidden_states.device,
        )

        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)

        attention_scores = attention_scores.masked_fill(
            causal_mask.view(1, 1, seq_length, key_states.shape[2]),
            float("-inf"),
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1,
        )

        attn_output = torch.matmul(
            attention_weights,
            all_values.float(),
        )

        attn_output = attn_output.permute(
            0,
            2,
            1,
            3,
        )

        gate = self.g_proj(hidden_states)
        gate = torch.sigmoid(gate.float()).unsqueeze(-1)

        attn_output = attn_output * gate

        attn_output = attn_output.reshape(
            batch_size,
            seq_length,
            self.num_heads * self.v_head_dim,
        )

        output = self.dense(attn_output)

        next_state = {
            "key": key_states,
            "value": all_values,
            "position": past_length + seq_length,
        }

        if return_state:
            return output, next_state

        return output
