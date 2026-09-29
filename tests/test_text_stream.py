import sys

sys.path.insert(0, r".\src")

import torch
from tokenizers import Tokenizer

from edge0_cuda.model import Edge0Model
from edge0_cuda.layer_factory import layer_factory

MODEL_DIR = r".\models\edge0-8b"

tokenizer = Tokenizer.from_file(
    MODEL_DIR + r"\tokenizer.json"
)

model = Edge0Model(
    MODEL_DIR,
    layer_factory,
    cache_size=4,
)

prompt = "Hello, how are you?"

encoded = tokenizer.encode(prompt)

input_ids = torch.tensor(
    [encoded.ids],
    dtype=torch.long,
)

print("PROMPT:", prompt)
print("MODEL:")

decoder = tokenizer.decode_batch

generated = []

for token_id in model.generate_stream(
    input_ids,
    max_new_tokens=32,
    temperature=0.7,
):
    generated.append(token_id)

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    prompt_text = tokenizer.decode(
        encoded.ids,
        skip_special_tokens=True,
    )

    new_text = text[len(prompt_text):]

    print("\r" + new_text, end="", flush=True)

print()
