import torch


def unpack_int4_u32(
    packed: torch.Tensor,
) -> torch.Tensor:
    """
    Unpack 8 unsigned 4-bit values from every uint32 value.

    Layout:
        uint32 -> 8 x uint4
    """

    packed = packed.to(torch.int64)

    values = []

    for shift in range(0, 32, 4):
        value = (packed >> shift) & 0xF
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


def dequantize_affine_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int = 64,
) -> torch.Tensor:
    """
    Dequantize affine INT4 weights.

    q = unpacked uint4

    weight = q * scale + bias
    """

    q = unpack_int4_u32(
        packed
    ).float()

    in_features = q.shape[-1]

    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) "
            f"must be divisible by "
            f"group_size ({group_size})"
        )

    groups = in_features // group_size

    if scales.shape[-1] != groups:
        raise ValueError(
            f"Invalid scale groups: "
            f"{scales.shape[-1]} != {groups}"
        )

    if biases.shape[-1] != groups:
        raise ValueError(
            f"Invalid bias groups: "
            f"{biases.shape[-1]} != {groups}"
        )

    scales = (
        scales.float()
        .repeat_interleave(
            group_size,
            dim=-1,
        )
    )

    biases = (
        biases.float()
        .repeat_interleave(
            group_size,
            dim=-1,
        )
    )

    return (
        q * scales
        + biases
    )
