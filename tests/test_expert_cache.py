from edge0_cuda.expert_cache import ExpertCache


cache = ExpertCache(
    max_experts=4
)

cache.put(10, "expert10")
cache.put(20, "expert20")
cache.put(30, "expert30")
cache.put(40, "expert40")

print("Initial:", cache.keys())

print("Get 20:", cache.get(20))

cache.put(50, "expert50")

print("After adding 50:", cache.keys())

print("Get 20 again:", cache.get(20))
print("Get 10:", cache.get(10))

print()
print("Stats:")
print(cache.stats())

assert cache.keys() == [
    30,
    40,
    50,
    20,
]

assert cache.get(999) is None

print()
print("EXPERT CACHE OK")
