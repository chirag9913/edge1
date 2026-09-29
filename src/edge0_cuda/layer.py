import torch

from .mla import RMSNorm
from .kimi_delta import KimiDeltaAttention
from .moe import StreamingMoE
from .shared_expert import SharedExpert


class Edge0Layer1:
    """
    Edge0 Layer 1:

        input
          -> input RMSNorm
          -> KimiDelta attention
          -> residual
          -> post-attention RMSNorm
          -> routed MoE
          -> shared expert
          -> residual
    """

    def __init__(
        self,
        input_layernorm,
        attention,
        post_attention_layernorm,
        moe,
        shared_expert,
    ):
        self.input_layernorm = input_layernorm
        self.attention = attention
        self.post_attention_layernorm = post_attention_layernorm
        self.moe = moe
        self.shared_expert = shared_expert

    @torch.no_grad()
    def __call__(
        self,
        hidden_states,
        attention_state=None,
        return_state=False,
    ):
        # ----------------------------------------------------
        # Attention block
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # MoE block
        # ----------------------------------------------------

        residual = hidden_states

        x = self.post_attention_layernorm(hidden_states)

        # StreamingMoE and SharedExpert work on [tokens, hidden].
        # Flatten [batch, sequence, hidden] -> [tokens, hidden].
        batch_size, sequence_length, hidden_size = x.shape

        moe_input = x.reshape(
            batch_size * sequence_length,
            hidden_size,
        )

        routed, topk_idx, topk_weight, router_logits = self.moe(moe_input)

        shared = self.shared_expert(moe_input)

        # Back to [batch, sequence, hidden].
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
                attention_state,
                topk_idx,
                topk_weight,
                router_logits,
            )

        return hidden_states