"""A small, readable decoder-only transformer for the CS 158 notebooks.

The implementation intentionally uses only Python and PyTorch.  It includes a
transparent word-and-punctuation tokenizer, a causal transformer, generation,
training helpers, activation caches, and an intervention hook.  The notebooks
build each of these ideas gradually before importing this complete version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import random
import re
from typing import Callable, Iterable

import torch
from torch import nn
from torch.nn import functional as F


TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def seed_everything(seed: int = 158) -> None:
    """Seed Python and PyTorch for reproducible classroom examples."""

    random.seed(seed)
    torch.manual_seed(seed)


def tokenize_words(text: str) -> list[str]:
    """Split text into words and individual punctuation marks."""

    return TOKEN_PATTERN.findall(text)


class WordTokenizer:
    """A deliberately simple word-and-punctuation tokenizer."""

    def __init__(self, vocabulary: list[str]):
        if vocabulary[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError(f"Vocabulary must begin with {SPECIAL_TOKENS}.")
        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError("Vocabulary entries must be unique.")

        self.itos = list(vocabulary)
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        max_vocab_size: int = 4096,
    ) -> "WordTokenizer":
        if max_vocab_size < len(SPECIAL_TOKENS):
            raise ValueError(
                f"max_vocab_size must be at least {len(SPECIAL_TOKENS)}."
            )
        counts: dict[str, int] = {}
        for text in texts:
            for token in tokenize_words(text):
                counts[token] = counts.get(token, 0) + 1

        # Frequency first and alphabetical order as a deterministic tie-breaker.
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        available = max_vocab_size - len(SPECIAL_TOKENS)
        vocabulary = SPECIAL_TOKENS + ordered[:available]
        return cls(vocabulary)

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.stoi.get(token, self.unk_id) for token in tokenize_words(text)]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens = [self.itos[int(index)] for index in ids]
        if skip_special_tokens:
            structural_tokens = {"<pad>", "<bos>", "<eos>"}
            tokens = [token for token in tokens if token not in structural_tokens]

        text = ""
        attach_left = {".", ",", "!", "?", ":", ";", ")", "]", "}"}
        attach_right = {"(", "[", "{"}
        quote_is_open = True
        after_open_quote = False

        for token in tokens:
            if token == '"':
                if quote_is_open:
                    text += (' ' if text and not text.endswith(' ') else '') + token
                    after_open_quote = True
                else:
                    text += token
                    after_open_quote = False
                quote_is_open = not quote_is_open
            elif token in attach_left:
                text += token
            elif after_open_quote or (text and text[-1:] in attach_right):
                text += token
                after_open_quote = False
            else:
                text += (" " if text else "") + token
        return text

    def to_dict(self) -> dict[str, list[str]]:
        return {"vocabulary": self.itos}

    @classmethod
    def from_dict(cls, state: dict[str, list[str]]) -> "WordTokenizer":
        return cls(list(state["vocabulary"]))


def make_demo_stories(number_of_stories: int = 320) -> list[str]:
    """Create a deterministic miniature corpus for fast, offline exercises.

    This corpus is for checking the training and interpretability machinery.  It
    is deliberately structured and is not a replacement for TinyStories.
    """

    names = ["Ava", "Ben", "Cora", "Diego", "Ella", "Finn", "Gia", "Hugo"]
    animals = ["cat", "dog", "fox", "rabbit", "turtle", "bird", "goat", "mouse"]
    colors = ["red", "blue", "green", "gold", "purple", "orange", "silver", "white"]
    objects = ["kite", "ball", "book", "key", "boat", "bell", "hat", "map"]
    places = ["garden", "forest", "river", "school", "farm", "beach", "hill", "park"]
    actions = ["carried", "found", "fixed", "shared", "painted", "washed", "moved", "opened"]
    joys = ["smiled", "laughed", "cheered", "danced"]
    worries = ["felt worried", "felt lonely", "felt tired", "felt afraid"]

    stories: list[str] = []
    for index in range(number_of_stories):
        # The first three indices form a mixed-radix counter, so the first 512
        # stories have distinct (name, animal, color) combinations.  Earlier
        # versions advanced every length-eight list with modular arithmetic and
        # accidentally repeated the entire corpus every 40 stories.
        name_index = index % len(names)
        animal_index = (index // len(names)) % len(animals)
        color_index = (index // (len(names) * len(animals))) % len(colors)

        name = names[name_index]
        friend = names[(name_index + animal_index + 3) % len(names)]
        animal = animals[animal_index]
        color = colors[color_index]
        item = objects[(name_index + 2 * animal_index + color_index) % len(objects)]
        place = places[(3 * name_index + animal_index + color_index) % len(places)]
        action = actions[(name_index + animal_index + 2 * color_index) % len(actions)]

        style = index % 5
        if style == 0:
            story = (
                f"{name} found a {color} {item} near the {place}. "
                f"A little {animal} watched quietly. {name} {action} the {item}. "
                f'"What a good day!" {name} said. {name} {joys[index % len(joys)]}.'
            )
        elif style == 1:
            story = (
                f"{name} wanted to visit the {place}. The path was long. "
                f"A friendly {animal} showed {name} a short path. "
                f"They arrived safely. {name} thanked the {animal}."
            )
        elif style == 2:
            story = (
                f"{name} had a {color} {item}. {friend} did not have one. "
                f"{name} shared the {item} with {friend}. "
                f"They played beside the {place}. Both friends were happy."
            )
        elif style == 3:
            story = (
                f"Rain fell over the {place}. {name} {worries[index % len(worries)]}. "
                f'The {animal} said, "Please come inside." '
                f"{name} followed the {animal}. Soon they were warm and safe."
            )
        else:
            story = (
                f"{name} lost a {color} {item} at the {place}. "
                f"{name} looked under a tree and beside a stone. "
                f"The {animal} found the {item}. {name} was grateful. "
                f'"Thank you!" {name} said.'
            )
        stories.append(story)

    return stories


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, channels = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, sequence_length, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.n_heads, self.head_size).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        mask = self.causal_mask[:, :, :sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)

        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, channels)
        output = self.output_dropout(self.projection(attended))

        if return_weights:
            return output, weights
        return output


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_size = config.mlp_ratio * config.d_model
        self.input = nn.Linear(config.d_model, hidden_size)
        self.output = nn.Linear(hidden_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.input(x))
        output = self.dropout(self.output(hidden))
        if return_hidden:
            return output, hidden
        return output


class TransformerBlock(nn.Module):
    """A pre-LayerNorm decoder block with two residual additions."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_attention = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ln_mlp = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        residual_pre = x
        attention_output, attention_weights = self.attention(
            self.ln_attention(x), return_weights=True
        )
        residual_mid = x + attention_output
        mlp_output, mlp_hidden = self.mlp(self.ln_mlp(residual_mid), return_hidden=True)
        residual_post = residual_mid + mlp_output

        if return_cache:
            cache = {
                "residual_pre": residual_pre,
                "attention_weights": attention_weights,
                "attention_output": attention_output,
                "residual_mid": residual_mid,
                "mlp_hidden": mlp_hidden,
                "mlp_output": mlp_output,
                "residual_post": residual_post,
            }
            return residual_post, cache
        return residual_post


Intervention = Callable[[int, torch.Tensor], torch.Tensor]


class PicoGPT(nn.Module):
    """A decoder-only transformer for next-token prediction."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._initialize_weights)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        indices: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_cache: bool = False,
        intervention: Intervention | None = None,
    ):
        batch_size, sequence_length = indices.shape
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block size "
                f"{self.config.block_size}."
            )

        positions = torch.arange(sequence_length, device=indices.device)
        token_vectors = self.token_embedding(indices)
        position_vectors = self.position_embedding(positions)
        x = token_vectors + position_vectors

        block_caches: list[dict[str, torch.Tensor]] = []
        for layer_index, block in enumerate(self.blocks):
            if return_cache:
                x, block_cache = block(x, return_cache=True)
            else:
                x = block(x)
                block_cache = {}

            if intervention is not None:
                x = intervention(layer_index, x)
                if return_cache:
                    block_cache["residual_post_intervened"] = x

            if return_cache:
                block_caches.append(block_cache)

        normalized = self.final_layer_norm(x)
        logits = self.lm_head(normalized)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        if return_cache:
            cache = {
                "token_embeddings": token_vectors,
                "position_embeddings": position_vectors,
                "embedding_stream": token_vectors + position_vectors,
                "blocks": block_caches,
                "normalized": normalized,
            }
            return logits, loss, cache
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        indices: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
        seed: int = 158,
        intervention: Intervention | None = None,
    ) -> torch.Tensor:
        self.eval()
        if temperature < 0:
            raise ValueError("temperature must be nonnegative.")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be at least 1 when provided.")

        # MPS does not consistently support a device-specific Generator across
        # PyTorch releases.  The global seed is the compatible fallback there.
        if indices.device.type == "mps":
            torch.manual_seed(seed)
            generator = None
        else:
            generator = torch.Generator(device=indices.device)
            generator.manual_seed(seed)

        finished = torch.zeros(indices.shape[0], dtype=torch.bool, device=indices.device)

        for _ in range(max_new_tokens):
            context = indices[:, -self.config.block_size :]
            logits, _ = self(context, intervention=intervention)
            next_logits = logits[:, -1, :]

            if temperature == 0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None:
                    k = min(top_k, next_logits.size(-1))
                    threshold = torch.topk(next_logits, k).values[:, [-1]]
                    next_logits = next_logits.masked_fill(
                        next_logits < threshold, float("-inf")
                    )
                probabilities = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(
                    probabilities, num_samples=1, generator=generator
                )

            if eos_id is not None:
                eos_fill = torch.full_like(next_id, eos_id)
                next_id = torch.where(finished[:, None], eos_fill, next_id)
            indices = torch.cat([indices, next_id], dim=1)
            if eos_id is not None:
                finished |= next_id.squeeze(-1) == eos_id
                if bool(torch.all(finished)):
                    break

        return indices


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def split_stories(
    stories: list[str], validation_fraction: float = 0.1, seed: int = 158
) -> tuple[list[str], list[str]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    shuffled = list(stories)
    random.Random(seed).shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * (1 - validation_fraction)))
    train_stories = shuffled[:split_index]
    validation_stories = shuffled[split_index:]
    if not validation_stories:
        raise ValueError("The split must contain at least one validation story.")
    return train_stories, validation_stories


def build_token_stream(stories: Iterable[str], tokenizer: WordTokenizer) -> torch.Tensor:
    ids: list[int] = []
    for story in stories:
        ids.extend(tokenizer.encode(story, add_special_tokens=True))
    return torch.tensor(ids, dtype=torch.long)


def get_batch(
    stream: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(stream) <= block_size:
        raise ValueError("The token stream must be longer than block_size.")
    starts = torch.randint(
        0, len(stream) - block_size, (batch_size,), generator=generator
    )
    inputs = torch.stack([stream[start : start + block_size] for start in starts])
    targets = torch.stack(
        [stream[start + 1 : start + block_size + 1] for start in starts]
    )
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model: PicoGPT,
    stream: torch.Tensor,
    batches: int,
    batch_size: int,
    seed: int = 159,
) -> float:
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    losses: list[float] = []
    for _ in range(batches):
        inputs, targets = get_batch(
            stream, batch_size, model.config.block_size, generator, device
        )
        _, loss = model(inputs, targets)
        losses.append(float(loss))
    model.train(was_training)
    return sum(losses) / len(losses)


def train_model(
    config: GPTConfig,
    train_stream: torch.Tensor,
    validation_stream: torch.Tensor,
    steps: int = 300,
    batch_size: int = 24,
    learning_rate: float = 3e-3,
    weight_decay: float = 0.01,
    seed: int = 158,
    device: torch.device | str = "cpu",
    report_every: int = 50,
) -> tuple[PicoGPT, list[dict[str, float]]]:
    seed_everything(seed)
    device = torch.device(device)
    model = PicoGPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed + 1)
    history: list[dict[str, float]] = []

    for step in range(steps + 1):
        if step % report_every == 0 or step == steps:
            train_loss = estimate_loss(model, train_stream, 3, batch_size, seed + step)
            validation_loss = estimate_loss(
                model, validation_stream, 3, batch_size, seed + step + 10_000
            )
            history.append(
                {
                    "step": float(step),
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            print(
                f"step {step:4d} | train {train_loss:.3f} | "
                f"validation {validation_loss:.3f}"
            )
        if step == steps:
            break

        inputs, targets = get_batch(
            train_stream, batch_size, config.block_size, generator, device
        )
        _, loss = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return model, history


def save_checkpoint(
    path: str | Path,
    model: PicoGPT,
    tokenizer: WordTokenizer,
    metadata: dict | None = None,
) -> None:
    checkpoint = {
        "config": asdict(model.config),
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "metadata": metadata or {},
    }
    torch.save(checkpoint, Path(path))


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[PicoGPT, WordTokenizer, dict]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    tokenizer = WordTokenizer.from_dict(checkpoint["tokenizer"])
    config = GPTConfig(**checkpoint["config"])
    model = PicoGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, dict(checkpoint.get("metadata", {}))


def train_demo_model(
    steps: int = 300,
    device: torch.device | str = "cpu",
    seed: int = 158,
) -> tuple[PicoGPT, WordTokenizer, list[dict[str, float]], dict[str, torch.Tensor]]:
    stories = make_demo_stories()
    train_stories, validation_stories = split_stories(stories)
    tokenizer = WordTokenizer.from_texts(train_stories, max_vocab_size=512)
    train_stream = build_token_stream(train_stories, tokenizer)
    validation_stream = build_token_stream(validation_stories, tokenizer)
    config = GPTConfig(
        vocab_size=len(tokenizer),
        block_size=48,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )
    model, history = train_model(
        config,
        train_stream,
        validation_stream,
        steps=steps,
        batch_size=24,
        learning_rate=3e-3,
        seed=seed,
        device=device,
        report_every=max(1, steps // 4),
    )
    streams = {"train": train_stream, "validation": validation_stream}
    return model, tokenizer, history, streams


def ensure_demo_checkpoint(
    path: str | Path,
    steps: int = 300,
    device: torch.device | str = "cpu",
) -> tuple[PicoGPT, WordTokenizer, dict]:
    path = Path(path)
    if path.exists():
        return load_checkpoint(path, device=device)

    model, tokenizer, history, streams = train_demo_model(steps=steps, device=device)
    metadata = {
        "purpose": "Fast offline checkpoint for CS 158 notebook exercises",
        "training_steps": steps,
        "history": history,
        "train_tokens": len(streams["train"]),
        "validation_tokens": len(streams["validation"]),
    }
    save_checkpoint(path, model, tokenizer, metadata)
    return model, tokenizer, metadata


def resolve_course_file(filename: str) -> Path:
    """Find a course artifact beside the notebook or in Homework2026/."""

    candidates = [Path(filename), Path("Homework2026") / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def preperiod_mask(indices: torch.Tensor, period_id: int) -> torch.Tensor:
    """Mark positions whose next token is a period."""

    mask = torch.zeros_like(indices, dtype=torch.bool)
    mask[:, :-1] = indices[:, 1:] == period_id
    return mask
