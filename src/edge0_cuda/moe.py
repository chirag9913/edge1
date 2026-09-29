import torch

from .expert import Int4Expert
from .expert_cache import ExpertCache


class StreamingMoE:

    def __init__(
        self,
        store,
        router,
        layer_idx: int,
        cache_size: int = 4,
    ):
        self.store = store
        self.router = router
        self.layer_idx = layer_idx

        self.cache = ExpertCache(
            max_experts=cache_size
        )

    def _read_u32(
        self,
        name,
        expert_idx,
        shape,
    ):
        raw = self.store.read_dim0_slice(
            name,
            expert_idx,
        )

        return torch.frombuffer(
            bytearray(raw),
            dtype=torch.uint32,
        ).reshape(shape)

    def _read_bf16(
        self,
        name,
        expert_idx,
        shape,
    ):
        raw = self.store.read_dim0_slice(
            name,
            expert_idx,
        )

        return torch.frombuffer(
            bytearray(raw),
            dtype=torch.bfloat16,
        ).reshape(shape)

    def _load_expert(
        self,
        expert_idx: int,
    ):
        prefix = (
            f"model.layers."
            f"{self.layer_idx}"
            f".mlp.experts."
        )

        gate = Int4Expert(
            self._read_u32(
                prefix + "gate_proj.weight",
                expert_idx,
                (512, 192),
            ),
            self._read_bf16(
                prefix + "gate_proj.scales",
                expert_idx,
                (512, 24),
            ),
            self._read_bf16(
                prefix + "gate_proj.biases",
                expert_idx,
                (512, 24),
            ),
        ).dequantize()

        up = Int4Expert(
            self._read_u32(
                prefix + "up_proj.weight",
                expert_idx,
                (512, 192),
            ),
            self._read_bf16(
                prefix + "up_proj.scales",
                expert_idx,
                (512, 24),
            ),
            self._read_bf16(
                prefix + "up_proj.biases",
                expert_idx,
                (512, 24),
            ),
        ).dequantize()

        down = Int4Expert(
            self._read_u32(
                prefix + "down_proj.weight",
                expert_idx,
                (1536, 64),
            ),
            self._read_bf16(
                prefix + "down_proj.scales",
                expert_idx,
                (1536, 8),
            ),
            self._read_bf16(
                prefix + "down_proj.biases",
                expert_idx,
                (1536, 8),
            ),
        ).dequantize()

        return gate, up, down

    def _get_expert(
        self,
        expert_idx: int,
    ):
        expert = self.cache.get(
            expert_idx
        )

        if expert is not None:
            return expert

        expert = self._load_expert(
            expert_idx
        )

        self.cache.put(
            expert_idx,
            expert,
        )

        return expert

    @staticmethod
    def _run_expert(
        x,
        gate,
        up,
        down,
    ):
        gate_output = x @ gate.t()

        up_output = x @ up.t()

        hidden = (
            torch.nn.functional.silu(
                gate_output
            )
            * up_output
        )

        return hidden @ down.t()

    @torch.no_grad()
    def __call__(
        self,
        hidden_states: torch.Tensor,
    ):
        if hidden_states.ndim != 2:
            raise ValueError(
                "StreamingMoE expects "
                "[tokens, hidden_size]"
            )

        (
            topk_idx,
            topk_weight,
            router_logits,
        ) = self.router(
            hidden_states
        )

        num_tokens = hidden_states.shape[0]

        output = torch.zeros(
            num_tokens,
            1536,
            dtype=torch.float32,
        )

        unique_experts = torch.unique(
            topk_idx
        )

        for expert_tensor in unique_experts:

            expert_idx = int(
                expert_tensor.item()
            )

            token_positions, slot_positions = (
                torch.where(
                    topk_idx == expert_idx
                )
            )

            if token_positions.numel() == 0:
                continue

            expert_input = hidden_states[
                token_positions
            ]

            gate, up, down = (
                self._get_expert(
                    expert_idx
                )
            )

            expert_output = self._run_expert(
                expert_input,
                gate,
                up,
                down,
            )

            weights = topk_weight[
                token_positions,
                slot_positions,
            ].unsqueeze(-1)

            output.index_add_(
                0,
                token_positions,
                expert_output * weights,
            )

        return (
            output,
            topk_idx,
            topk_weight,
            router_logits,
        )

    def cache_stats(self):
        return self.cache.stats()

    def cached_experts(self):
        return self.cache.keys()

    def clear_cache(self):
        self.cache.clear()
