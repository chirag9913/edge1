import torch
import torch.nn.functional as F


class Edge0Router:
    """
    Exact BailingMoeV3 router implementation.

    Architecture:
        128 experts
        8 groups
        16 experts per group
        top 4 groups
        top 8 experts/token
        sigmoid scoring
        expert bias used only for routing selection
        routing weights use the original sigmoid scores
    """

    def __init__(
        self,
        weight: torch.Tensor,
        expert_bias: torch.Tensor,
        top_k: int = 8,
        num_experts: int = 128,
        n_group: int = 8,
        topk_group: int = 4,
        routed_scaling_factor: float = 2.5,
    ):
        self.weight = weight.float()
        self.expert_bias = expert_bias.float()

        self.top_k = top_k
        self.num_experts = num_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor

        if num_experts % n_group != 0:
            raise ValueError(
                "num_experts must be divisible by n_group"
            )

        if weight.shape != (num_experts, weight.shape[-1]):
            raise ValueError(
                f"Invalid router weight shape: {weight.shape}"
            )

        if expert_bias.numel() != num_experts:
            raise ValueError(
                f"Invalid expert bias shape: {expert_bias.shape}"
            )

    @torch.no_grad()
    def __call__(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states:
                [tokens, hidden_size]

        Returns:
            topk_idx:
                [tokens, 8]

            topk_weight:
                [tokens, 8]

            logits:
                [tokens, 128]
        """

        hidden_states = hidden_states.reshape(
            -1,
            hidden_states.shape[-1],
        )

        # Exact source behavior:
        #
        # F.linear(
        #     hidden_states.type(torch.float32),
        #     self.weight.type(torch.float32)
        # )
        logits = F.linear(
            hidden_states.float(),
            self.weight.float(),
        )

        # Exact source behavior.
        scores = torch.sigmoid(logits)

        # Expert bias is used ONLY for selecting experts.
        scores_for_routing = (
            scores + self.expert_bias
        )

        # ---------------------------------------------------------
        # Group-limited top-k
        # ---------------------------------------------------------

        experts_per_group = (
            self.num_experts // self.n_group
        )

        grouped = scores_for_routing.view(
            -1,
            self.n_group,
            experts_per_group,
        )

        # Top 2 experts from each group.
        group_scores = grouped.topk(
            2,
            dim=-1,
        )[0].sum(dim=-1)

        # Select the best 4 groups.
        group_idx = torch.topk(
            group_scores,
            k=self.topk_group,
            dim=-1,
            sorted=False,
        )[1]

        # Create group mask.
        group_mask = torch.zeros_like(
            group_scores
        )

        group_mask.scatter_(
            1,
            group_idx,
            1,
        )

        # Expand group mask back to expert dimension.
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                -1,
                self.n_group,
                experts_per_group,
            )
            .reshape(
                -1,
                self.num_experts,
            )
        )

        # Remove experts from unselected groups.
        masked_scores = scores_for_routing.masked_fill(
            ~score_mask.bool(),
            float("-inf"),
        )

        # Select final 8 experts.
        _, topk_idx = torch.topk(
            masked_scores,
            k=self.top_k,
            dim=-1,
        )

        # IMPORTANT:
        # Routing weights use ORIGINAL sigmoid scores,
        # not bias-adjusted scores.
        selected_scores = torch.gather(
            scores,
            dim=1,
            index=topk_idx,
        )

        # Normalize.
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

        # Exact scaling factor from config.
        topk_weight = (
            topk_weight
            * self.routed_scaling_factor
        )

        return (
            topk_idx,
            topk_weight,
            logits,
        )
