import sys
sys.path.insert(0, r".\src")

import torch
from tokenizers import Tokenizer

from edge0_cuda.model import Edge0Model
from edge0_cuda.layer_factory import layer_factory


MODEL_DIR = r".\models\edge0-8b"
MAX_NEW_TOKENS = 20


print("Loading tokenizer...")
tokenizer = Tokenizer.from_file(MODEL_DIR + r"\tokenizer.json")

print("Loading Edge0-8B...")

model = Edge0Model(
    model_path=MODEL_DIR,
    layer_factory=layer_factory,
    cache_size=4,
)

print("Model loaded.")

prompt = input("You: ")

encoded = tokenizer.encode(prompt)
token_ids = encoded.ids

print("Assistant: ", end="", flush=True)

for _ in range(MAX_NEW_TOKENS):
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    with torch.no_grad():
        logits = model.logits(input_ids)

    next_token = int(logits[0, -1].argmax().item())

    token_ids.append(next_token)

    text = tokenizer.decode([next_token])
    print(text, end="", flush=True)

    if next_token in tokenizer.get_vocab().values():
        if text.strip() == "":
            continue

print()
