import sys
sys.path.insert(0, r".\src")

from edge0_cuda.mmap_store import SafetensorsMmap
from edge0_cuda.layer_factory import layer_factory

MODEL_DIR = r".\models\edge0-8b"

store = SafetensorsMmap(MODEL_DIR)

print("BUILDING LAYERS 1-23")
print("=" * 50)

for layer_idx in range(1, 24):
    print(f"Layer {layer_idx:2d}...", end=" ")

    layer = layer_factory(
        store=store,
        layer_idx=layer_idx,
        cache_size=4,
    )

    kind = "MLA" if (layer_idx + 1) % 4 == 0 else "KimiDelta"
    print(f"OK [{kind}]")

print("=" * 50)
print("ALL MOE LAYERS CONSTRUCTED")

store.close()
