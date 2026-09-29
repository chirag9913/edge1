import torch

from .mmap_store import SafetensorsMmap
from .mla import RMSNorm
from .quant import dequantize_affine_int4
from .layer0 import build_layer0
from .layer_factory import layer_factory


class TokenEmbedding:
    def __init__(
        self,
        store,
        name,
        scale_name,
        bias_name,
        vocab_size,
        hidden_size,
    ):
        self.store = store
        self.name = name
        self.scale_name = scale_name
        self.bias_name = bias_name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self._weight = None

        if tuple(store.info(name)["shape"]) != (
            vocab_size,
            hidden_size // 8,
        ):
            raise ValueError(
                "Unexpected embedding weight shape"
            )

    def weight(self):
        if self._weight is None:
            packed = torch.frombuffer(
                bytearray(
                    self.store.read_raw(self.name)
                ),
                dtype=torch.uint32,
            ).clone().reshape(
                self.vocab_size,
                self.hidden_size // 8,
            )

            scales = torch.frombuffer(
                bytearray(
                    self.store.read_raw(self.scale_name)
                ),
                dtype=torch.bfloat16,
            ).clone().reshape(
                self.vocab_size,
                self.hidden_size // 64,
            )

            biases = torch.frombuffer(
                bytearray(
                    self.store.read_raw(self.bias_name)
                ),
                dtype=torch.bfloat16,
            ).clone().reshape(
                self.vocab_size,
                self.hidden_size // 64,
            )

            self._weight = dequantize_affine_int4(
                packed,
                scales,
                biases,
                group_size=64,
            )

        return self._weight

    @torch.no_grad()
    def __call__(self, input_ids):
        return self.weight()[
            input_ids.long()
        ]


class LMHead:
    def __init__(
        self,
        store,
        name,
        scale_name,
        bias_name,
        vocab_size,
        hidden_size,
    ):
        self.store = store
        self.name = name
        self.scale_name = scale_name
        self.bias_name = bias_name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self._weight = None

    def weight(self):
        if self._weight is None:
            packed = torch.frombuffer(
                bytearray(
                    self.store.read_raw(self.name)
                ),
                dtype=torch.uint32,
            ).clone().reshape(
                self.vocab_size,
                self.hidden_size // 8,
            )

            scales = torch.frombuffer(
                bytearray(
                    self.store.read_raw(self.scale_name)
                ),
                dtype=torch.bfloat16,
            ).clone().reshape(
                self.vocab_size,
                self.hidden_size // 64,
            )

            biases = torch.frombuffer(
                bytearray(
                    self.store.read_raw(self.bias_name)
                ),
                dtype=torch.bfloat16,
            ).clone().reshape(
                self.vocab_size,
                self.hidden_size // 64,
            )

            self._weight = dequantize_affine_int4(
                packed,
                scales,
                biases,
                group_size=64,
            )

        return self._weight

    @torch.no_grad()
    def __call__(self, hidden_states):
        shape = hidden_states.shape

        hidden_states = hidden_states.reshape(
            -1,
            self.hidden_size,
        )

        logits = (
            hidden_states.float()
            @ self.weight().float().t()
        )

        return logits.reshape(
            *shape[:-1],
            self.vocab_size,
        )


class Edge0Model:
    VOCAB_SIZE = 157184
    HIDDEN_SIZE = 1536
    NUM_LAYERS = 24

    EMBEDDING_NAME = "model.word_embeddings.weight"
    EMBEDDING_SCALES = "model.word_embeddings.scales"
    EMBEDDING_BIASES = "model.word_embeddings.biases"

    FINAL_NORM_NAME = "model.norm.weight"

    LM_HEAD_NAME = "lm_head.weight"
    LM_HEAD_SCALES = "lm_head.scales"
    LM_HEAD_BIASES = "lm_head.biases"

    def __init__(
        self,
        model_path,
        layer_factory=layer_factory,
        device="cpu",
        cache_size=4,
        cache_embedding=False,
    ):
        self.model_path = model_path
        self.device = device
        self.cache_size = cache_size
        self.layer_factory = layer_factory

        self.store = SafetensorsMmap(
            model_path
        )

        self.embedding = TokenEmbedding(
            self.store,
            self.EMBEDDING_NAME,
            self.EMBEDDING_SCALES,
            self.EMBEDDING_BIASES,
            self.VOCAB_SIZE,
            self.HIDDEN_SIZE,
        )

        self.final_norm = RMSNorm(
            torch.frombuffer(
                bytearray(
                    self.store.read_raw(
                        self.FINAL_NORM_NAME
                    )
                ),
                dtype=torch.bfloat16,
            ).clone(),
            eps=1e-6,
        )

        self.lm_head = LMHead(
            self.store,
            self.LM_HEAD_NAME,
            self.LM_HEAD_SCALES,
            self.LM_HEAD_BIASES,
            self.VOCAB_SIZE,
            self.HIDDEN_SIZE,
        )

        self.layers = {}

    def _build_layer(self, layer_idx):
        if layer_idx in self.layers:
            return self.layers[layer_idx]

        if not 0 <= layer_idx < self.NUM_LAYERS:
            raise ValueError(
                f"Invalid layer index: {layer_idx}"
            )

        if layer_idx == 0:
            layer = build_layer0(
                self.store
            )
        else:
            layer = self.layer_factory(
                store=self.store,
                layer_idx=layer_idx,
                cache_size=self.cache_size,
            )

        self.layers[layer_idx] = layer

        return layer

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        start_layer=0,
        end_layer=None,
        hidden_states=None,
        attention_states=None,
        return_states=False,
    ):
        if end_layer is None:
            end_layer = self.NUM_LAYERS

        if hidden_states is None:
            hidden_states = self.embedding(
                input_ids
            )

        if attention_states is None:
            attention_states = {}

        for layer_idx in range(
            start_layer,
            end_layer,
        ):
            layer = self._build_layer(
                layer_idx
            )

            if layer_idx == 0:
                state = attention_states.get(layer_idx)

                if return_states:
                    hidden_states, next_state = layer(
                        hidden_states,
                        attention_state=state,
                        return_state=True,
                    )
                    attention_states[layer_idx] = next_state
                else:
                    hidden_states = layer(
                        hidden_states,
                        attention_state=state,
                        return_state=False,
                    )

                continue

            state = attention_states.get(
                layer_idx
            )

            if return_states:
                (
                    hidden_states,
                    next_state,
                    _,
                    _,
                    _,
                ) = layer(
                    hidden_states,
                    attention_state=state,
                    return_state=True,
                )

                attention_states[
                    layer_idx
                ] = next_state

            else:
                hidden_states = layer(
                    hidden_states,
                    attention_state=state,
                    return_state=False,
                )

        if return_states:
            return (
                hidden_states,
                attention_states,
            )

        return hidden_states

    @torch.no_grad()
    def logits(self, input_ids):
        hidden_states = self.forward(
            input_ids
        )

        hidden_states = self.final_norm(
            hidden_states
        )

        return self.lm_head(
            hidden_states
        )

    @torch.no_grad()
    def prefill(self, input_ids):
        hidden_states, states = self.forward(
            input_ids,
            return_states=True,
        )

        hidden_states = self.final_norm(
            hidden_states
        )

        logits = self.lm_head(
            hidden_states
        )

        return logits, states

    @torch.no_grad()
    def decode(
        self,
        input_ids,
        attention_states,
    ):
        hidden_states, attention_states = self.forward(
            input_ids,
            attention_states=attention_states,
            return_states=True,
        )

        hidden_states = self.final_norm(
            hidden_states
        )

        logits = self.lm_head(
            hidden_states
        )

        return logits, attention_states



    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=32,
        temperature=0.0,
        eos_token_id=None,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, sequence]")

        logits, attention_states = self.prefill(input_ids)

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(
                    next_logits,
                    dim=-1,
                    keepdim=True,
                )
            else:
                probs = torch.softmax(
                    next_logits / temperature,
                    dim=-1,
                )
                next_token = torch.multinomial(
                    probs,
                    num_samples=1,
                )

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )

            if eos_token_id is not None:
                if torch.all(
                    next_token == eos_token_id
                ):
                    break

            logits, attention_states = self.decode(
                next_token,
                attention_states,
            )

        return generated


    @torch.no_grad()
    def generate_stream(
        self,
        input_ids,
        max_new_tokens=32,
        temperature=0.0,
        eos_token_id=None,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, sequence]")

        logits, attention_states = self.prefill(input_ids)

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(
                    next_logits,
                    dim=-1,
                    keepdim=True,
                )
            else:
                probs = torch.softmax(
                    next_logits / temperature,
                    dim=-1,
                )
                next_token = torch.multinomial(
                    probs,
                    num_samples=1,
                )

            token_id = int(next_token[0, 0].item())

            yield token_id

            if eos_token_id is not None and token_id == eos_token_id:
                break

            logits, attention_states = self.decode(
                next_token,
                attention_states,
            )
