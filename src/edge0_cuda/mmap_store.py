from pathlib import Path
import json
import struct
from typing import Dict, List


class SafetensorsMmap:
    """
    Low-memory safetensors reader.

    The checkpoint is never loaded completely into RAM.

    We first read the safetensors headers and build an index.
    Individual tensor byte ranges can then be read on demand.
    """

    DTYPE_SIZES = {
        "U8": 1,
        "I8": 1,
        "U16": 2,
        "I16": 2,
        "U32": 4,
        "I32": 4,
        "U64": 8,
        "I64": 8,
        "F16": 2,
        "BF16": 2,
        "F32": 4,
        "F64": 8,
    }

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory does not exist: {self.model_dir}"
            )

        self.files = sorted(
            self.model_dir.glob("*.safetensors")
        )

        if not self.files:
            raise FileNotFoundError(
                f"No .safetensors files found in {self.model_dir}"
            )

        self._index: Dict[str, dict] = {}

        self._build_index()

    def _build_index(self):
        print("Building safetensors index...")

        for path in self.files:
            print(f"Indexing: {path.name}")

            with path.open("rb") as f:
                raw_size = f.read(8)

                if len(raw_size) != 8:
                    raise ValueError(
                        f"Invalid safetensors header: {path}"
                    )

                header_size = struct.unpack(
                    "<Q",
                    raw_size,
                )[0]

                header_bytes = f.read(header_size)

                if len(header_bytes) != header_size:
                    raise ValueError(
                        f"Truncated safetensors header: {path}"
                    )

                header = json.loads(
                    header_bytes.decode("utf-8")
                )

            for name, info in header.items():
                if name == "__metadata__":
                    continue

                self._index[name] = {
                    "file": path,
                    "dtype": info["dtype"],
                    "shape": tuple(info["shape"]),
                    "offsets": tuple(info["data_offsets"]),
                    "header_size": header_size,
                }

        print()
        print(
            f"Indexed {len(self._index):,} tensors "
            f"across {len(self.files)} files."
        )

    def keys(self) -> List[str]:
        return list(self._index.keys())

    def has(self, name: str) -> bool:
        return name in self._index

    def info(self, name: str) -> dict:
        if name not in self._index:
            raise KeyError(
                f"Tensor not found: {name}"
            )

        return self._index[name]

    def file_for(self, name: str) -> Path:
        return self.info(name)["file"]

    def read_raw(self, name: str) -> bytes:
        """
        Read the complete raw byte range of one tensor.
        """

        info = self.info(name)

        start, end = info["offsets"]

        data_start = 8 + info["header_size"]

        with info["file"].open("rb") as f:
            f.seek(data_start + start)

            data = f.read(end - start)

        if len(data) != end - start:
            raise IOError(
                f"Could not read complete tensor: {name}"
            )

        return data

    def read_dim0_slice(
        self,
        name: str,
        index: int,
    ) -> bytes:
        """
        Read one contiguous slice along dimension 0.

        Example:

            shape = [128, 512, 192]

        index=7 reads:

            [7, :, :]
        """

        info = self.info(name)

        shape = info["shape"]

        if len(shape) == 0:
            raise ValueError(
                f"Tensor has no dimensions: {name}"
            )

        if index < 0 or index >= shape[0]:
            raise IndexError(
                f"Index {index} outside dimension 0 "
                f"of size {shape[0]}"
            )

        dtype = info["dtype"]

        if dtype not in self.DTYPE_SIZES:
            raise ValueError(
                f"Unsupported dtype: {dtype}"
            )

        element_size = self.DTYPE_SIZES[dtype]

        elements_per_slice = 1

        for dim in shape[1:]:
            elements_per_slice *= dim

        slice_bytes = elements_per_slice * element_size

        start, _ = info["offsets"]

        data_start = 8 + info["header_size"]

        byte_offset = (
            data_start
            + start
            + index * slice_bytes
        )

        with info["file"].open("rb") as f:
            f.seek(byte_offset)
            data = f.read(slice_bytes)

        if len(data) != slice_bytes:
            raise IOError(
                f"Could not read slice {index} "
                f"from {name}"
            )

        return data

    def find(self, pattern: str) -> List[str]:
        """
        Find tensor names containing a substring.
        """

        pattern = pattern.lower()

        return [
            name
            for name in self._index
            if pattern in name.lower()
        ]

    def tensor_nbytes(self, name: str) -> int:
        """
        Return the raw byte size of a tensor.
        """

        start, end = self.info(name)["offsets"]

        return end - start

    def close(self):
        """
        Nothing needs to be closed because this implementation
        opens files only for individual reads.
        """
        pass

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()
