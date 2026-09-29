import torch


class Int4Expert:
    """
    Memory-efficient INT4 expert decoder.

    Edge0 checkpoint format:

        packed weights:
            U32

        gate/up:
            [128, 512, 192]

        down:
            [128, 1536, 64]

        logical INT4 width:
            packed_width * 8

        quantization:
            affine
            group size = 64
    """

    def __init__(
        self,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        biases: torch.Tensor,
        group_size: int = 64,
    ):
        self.packed_weight = packed_weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size

    @staticmethod
    def unpack_int4_u32(
        packed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Unpack eight INT4 values from every U32 value.
        """

        # CPU PyTorch does not reliably provide the required
        # unsigned right-shift operation, so use int64.
        packed = packed.to(torch.int64)

        values = []

        for shift in range(0, 32, 4):
            value = (
                packed >> shift
            ) & 0xF

            values.append(
                value.to(torch.uint8)
            )

        return torch.stack(
            values,
            dim=-1,
        ).reshape(
            *packed.shape[:-1],
            packed.shape[-1] * 8,
        )

    def dequantize(self) -> torch.Tensor:
        """
        Dequantize one expert weight matrix.

        Returns:
            float32 matrix
        """

        q = self.unpack_int4_u32(
            self.packed_weight
        ).float()

        out_features = q.shape[-2]
        in_features = q.shape[-1]

        if in_features % self.group_size != 0:
            raise ValueError(
                f"Input dimension {in_features} "
                f"is not divisible by group size "
                f"{self.group_size}"
            )

        groups = (
            in_features // self.group_size
        )

        scales = self.scales.float()
        biases = self.biases.float()

        if scales.shape[-1] != groups:
            raise ValueError(
                f"Scale groups {scales.shape[-1]} "
                f"!= expected {groups}"
            )

        if biases.shape[-1] != groups:
            raise ValueError(
                f"Bias groups {biases.shape[-1]} "
                f"!= expected {groups}"
            )

        scales = scales.repeat_interleave(
            self.group_size,
            dim=-1,
        )

        biases = biases.repeat_interleave(
            self.group_size,
            dim=-1,
        )

        return q * scales + biases

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute one dequantized expert.

        Weight layout:
            [out_features, in_features]

        Therefore:
            x @ weight.T
        """

        weight = self.dequantize()

        return torch.matmul(
            x,
            weight.t(),
        )
