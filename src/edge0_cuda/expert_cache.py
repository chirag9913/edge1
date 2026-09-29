from collections import OrderedDict


class ExpertCache:
    """
    Small LRU cache for dequantized experts.

    Keeps only a fixed number of experts in RAM.
    """

    def __init__(self, max_experts: int = 4):
        if max_experts < 1:
            raise ValueError(
                "max_experts must be >= 1"
            )

        self.max_experts = max_experts
        self._cache = OrderedDict()

        self.hits = 0
        self.misses = 0

    def get(self, expert_id):
        if expert_id not in self._cache:
            self.misses += 1
            return None

        self.hits += 1

        value = self._cache.pop(
            expert_id
        )

        self._cache[expert_id] = value

        return value

    def put(
        self,
        expert_id,
        expert,
    ):
        if expert_id in self._cache:
            self._cache.pop(
                expert_id
            )

        self._cache[expert_id] = expert

        while len(self._cache) > self.max_experts:
            self._cache.popitem(
                last=False
            )

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)

    def keys(self):
        return list(self._cache.keys())

    def stats(self):
        total = self.hits + self.misses

        hit_rate = (
            self.hits / total
            if total
            else 0.0
        )

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cached": len(self._cache),
        }
