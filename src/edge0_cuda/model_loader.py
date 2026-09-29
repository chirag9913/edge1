import json
from pathlib import Path

from .mmap_store import SafetensorsMmap


class Edge0ModelLoader:
    """
    Loads Edge0 model configuration and checkpoint metadata.

    This class deliberately does not load the model weights into RAM.
    """

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory does not exist: {self.model_dir}"
            )

        self.config = {}
        self.tensor_store = None

    def load_config(self):
        config_path = self.model_dir / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json not found: {config_path}"
            )

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.config = json.load(f)

        return self.config

    def open_tensors(self):
        self.tensor_store = SafetensorsMmap(
            str(self.model_dir)
        )

        return self.tensor_store

    def load(self):
        self.load_config()
        self.open_tensors()

        return self.inspect()

    def inspect(self):
        if not self.config:
            self.load_config()

        if self.tensor_store is None:
            self.open_tensors()

        print()
        print("=" * 60)
        print("EDGE0 MODEL INSPECTION")
        print("=" * 60)

        print()
        print("Configuration:")

        important_keys = [
            "architectures",
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "num_experts",
            "num_experts_per_tok",
            "n_group",
            "topk_group",
            "routed_scaling_factor",
            "score_function",
            "norm_topk_prob",
            "first_k_dense_replace",
        ]

        for key in important_keys:
            if key in self.config:
                print(
                    f"  {key}: {self.config[key]}"
                )

        print()
        print("Checkpoint:")

        print(
            f"  Files: {len(self.tensor_store.files)}"
        )

        print(
            f"  Tensors: {len(self.tensor_store.keys()):,}"
        )

        print()
        print("MoE tensors:")

        expert_names = self.tensor_store.find(
            ".mlp.experts."
        )

        for name in expert_names[:30]:
            print(f"  {name}")

        if len(expert_names) > 30:
            print(
                f"  ... and "
                f"{len(expert_names) - 30} more"
            )

        return {
            "config": self.config,
            "files": self.tensor_store.files,
            "tensor_count": len(
                self.tensor_store.keys()
            ),
            "expert_tensors": expert_names,
        }

    def close(self):
        if self.tensor_store is not None:
            self.tensor_store.close()
            self.tensor_store = None

    def __enter__(self):
        self.load()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()