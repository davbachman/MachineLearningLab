import { defineCodeLabQuestion, type CodeLabQuestionSpec, type CodeLabStageSpec } from '../lib/codeLab'

function stage(id: string, title: string, prompt: string, correctOptionId: string, choices: [string, string][], successCopy: string, kind: CodeLabStageSpec['kind'] = 'executionTrace'): CodeLabStageSpec {
  return { id, title, prompt, kind, fields: [{ id: 'answer', label: 'Choose the trace result', correctOptionId, options: choices.map(([id, label]) => ({ id, label })) }], successCopy }
}
function lab(input: Pick<CodeLabQuestionSpec, 'id' | 'title' | 'prompt' | 'code' | 'fixture' | 'invocation' | 'stages'>): CodeLabQuestionSpec {
  return defineCodeLabQuestion({ ...input, kind: 'codeLab', language: 'python', variantId: `${input.id}-v1`, fixtureTitle: 'Fixed inputs and starting state', invocationTitle: 'Exact checks to trace', fixtureHeading: 'Deterministic fixture', invocationLead: 'Use the original implementation with these explicit calls. No checkpoint-specific prediction is required.', hints: ['Read the named tensor axes and follow each operation on the supplied inputs.', 'Each stage is independent; do not use a previous submitted answer as an input.'], hintSchedule: [2, 4], reveal: { explanation: 'The checkpoints trace the supplied notebook implementation on fixed shapes or tiny inputs. Training-dependent decimal values and generated words are deliberately not graded.' } })
}

const pico = lab({
  id: 'pico-gpt-notebook-lab', title: 'Notebook Lab: A Complete Decoder',
  prompt: 'Open Homework2026/21PicoGPT.ipynb, sections S1–S7. The code panel includes its original model definitions and greedy generation. Use the fixed IDs below instead of S5’s random IDs. This is an untrained-model architecture trace.',
  code: `import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
import math

# 21PicoGPT.ipynb, S1–S4: original model definitions.
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 16
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: int = 4
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.number_of_heads = config.n_heads
        self.head_size = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.number_of_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.number_of_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.number_of_heads, self.head_size).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        scores = scores.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)
        return self.projection(attended)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.mlp_ratio * config.d_model
        self.layers = nn.Sequential(
            nn.Linear(config.d_model, hidden_size),
            nn.GELU(), nn.Linear(hidden_size, config.d_model),
        )
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_attention = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.ln_mlp = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)
    def forward(self, x):
        x = x + self.attention(self.ln_attention(x))
        x = x + self.mlp(self.ln_mlp(x))
        return x

class PicoGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        self.lm_head.weight = self.token_embedding.weight
    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    def forward(self, indices, targets=None):
        B, T = indices.shape
        if T > self.config.block_size:
            raise ValueError("The sequence exceeds block_size.")
        positions = torch.arange(T, device=indices.device)
        x = self.token_embedding(indices) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_layer_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            )
        return logits, loss
    @torch.no_grad()
    def generate_greedily(self, indices, number_of_tokens):
        self.eval()
        for _ in range(number_of_tokens):
            context = indices[:, -self.config.block_size:]
            logits, _ = self(context)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            indices = torch.cat([indices, next_id], dim=1)
        return indices`,
  fixture: `# Use GPTConfig and PicoGPT as defined above.
torch.manual_seed(158)
config = GPTConfig(vocab_size=20, block_size=16,
                   d_model=32, n_heads=4, n_layers=2)
model = PicoGPT(config)
input_ids = torch.arange(36).reshape(3, 12) % 20
target_ids = (input_ids + 1) % 20
first = torch.tensor([[1, 2, 3, 4, 5, 6]])
second = torch.tensor([[1, 2, 3, 12, 13, 14]])
prompt = torch.tensor([[0, 4, 5]])`,
  invocation: `logits, loss = model(input_ids, target_ids)
print(tuple(logits.shape))
print(target_ids.numel())
print(model.lm_head.weight is model.token_embedding.weight)
first_logits, _ = model(first)
second_logits, _ = model(second)
print(torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=1e-6))
generated_ids = model.generate_greedily(prompt, number_of_tokens=8)
print(generated_ids.shape[1])`,
  stages: [
    stage('logits', 'Preserve positions, add vocabulary scores', 'What is tuple(logits.shape)?', 'shape', [['shape', '(3, 12, 20)'], ['channels', '(3, 12, 32)'], ['last', '(3, 20)']], 'The final head replaces the channel axis with the vocabulary axis while retaining every batch and sequence position.'),
    stage('loss', 'Count the supervised positions', 'How many next-token targets enter the single averaged cross-entropy loss?', '36', [['3', '3'], ['36', '36'], ['720', '720']], 'The reshape combines 3 × 12 positions into 36 classification examples; 20 is the number of classes, not the number of targets.'),
    stage('tying', 'Check object identity', 'What does the weight-identity check print?', 'true', [['equalonly', 'False'], ['true', 'True'], ['error', 'It raises an error because the shapes differ.']], 'Assignment of the Parameter itself creates weight tying. It is not merely a one-time copy of values.'),
    stage('causality', 'Change a suffix', 'What does the allclose check on the first three positions print?', 'true', [['false', 'False'], ['true', 'True'], ['random', 'It depends on which random token has the highest logit.']], 'Masked attention, per-token normalization, and per-token MLPs preserve causal-prefix invariance through the full model.'),
    stage('generation', 'Count appended tokens', 'What value does generated_ids.shape[1] print?', '11', [['8', '8'], ['16', '16'], ['11', '11']], 'The loop appends eight tokens to the three-token prompt. This notebook method has no early EOS stopping condition.'),
  ],
})

const training = lab({
  id: 'training-pico-gpt-notebook-lab', title: 'Notebook Lab: Windows, Updates, and Sampling',
  prompt: 'Open Homework2026/22TrainingPicoGPT.ipynb, S3–S7, and Homework2026/pico_gpt.py. The notebook imports the quoted helper functions. Trace the one-possible-start window and fixed score vector below; do not train a model or rely on the reference checkpoint’s sampled text.',
  code: `import torch
from torch.nn import functional as F
from dataclasses import asdict

# pico_gpt.py: get_batch (type annotations omitted)
def get_batch(stream, batch_size, block_size, generator, device="cpu"):
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

# Lab wrapper around one actual train_model iteration; not invoked here.
def one_training_iteration(model, optimizer, inputs, targets):
    _, loss = model(inputs, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

# Lab wrapper around generate's positive-temperature branch.
def distribution(next_logits, temperature, top_k):
    next_logits = next_logits / temperature
    if top_k is not None:
        k = min(top_k, next_logits.size(-1))
        threshold = torch.topk(next_logits, k).values[:, [-1]]
        next_logits = next_logits.masked_fill(
            next_logits < threshold, float("-inf")
        )
    probabilities = F.softmax(next_logits, dim=-1)
    return probabilities

# Lab wrapper around the dictionary assembled by save_checkpoint.
def checkpoint_dictionary(model, tokenizer, metadata):
    checkpoint = {
        "config": asdict(model.config),
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "metadata": metadata or {},
    }
    return checkpoint`,
  fixture: `import torch
from torch.nn import functional as F
stream = torch.tensor([10, 20, 30, 40])
generator = torch.Generator().manual_seed(159)
# len(stream) - block_size = 1, so start is always 0.

# A separate toy next-token score fixture, not checkpoint logits:
next_logits = torch.tensor([[2.0, 1.0, 1.0, 0.0]])
temperature = 1.0
top_k = 2

# For the update-order question only, assume scalar gradient 3.0
# exists after loss.backward(), and there are no other gradients.
# For checkpoint contents, model/tokenizer denote the S4/S2 objects;
# no file write is needed to inspect the quoted dictionary.`,
  invocation: `inputs, targets = get_batch(stream, batch_size=1, block_size=3,
                            generator=generator, device="cpu")
print(targets.tolist())
print(tuple(inputs.shape))

# Run the wrapped positive-temperature branch with the score fixture.
probabilities = distribution(next_logits, temperature, top_k)
print((probabilities > 0).sum().item())

# Conceptually inspect this S7 call's dictionary before it is saved:
# save_checkpoint(COURSE_DIRECTORY / "pico_gpt_student.pt",
#                 model, tokenizer, metadata={"training_steps": 120})`,
  stages: [
    stage('target', 'Shift by one', 'What does targets.tolist() return?', 'shift', [['same', '[[10, 20, 30]]'], ['shift', '[[20, 30, 40]]'], ['scalar', '[40]']], 'The only valid start is 0. The target slice starts at 1 and includes the last stream token.'),
    stage('shape', 'Keep batch and context axes', 'What is tuple(inputs.shape)?', '1x3', [['3', '(3,)'], ['1x4', '(1, 4)'], ['1x3', '(1, 3)']], 'Stacking one length-3 slice retains a batch axis of size 1.'),
    stage('clip', 'Locate clipping in the update', 'Which gradient is presented to optimizer.step() in the supplied one-gradient fixture?', 'clipped', [['raw', 'The original gradient 3.0, because clipping happens after the step.'], ['clipped', 'A gradient with magnitude approximately 1.0, clipped after backpropagation.'], ['zero', 'A zero gradient, because zero_grad is the final operation.']], 'The loop zeros old gradients, computes new ones, clips their global norm, then takes the optimizer step. The numerical value is approximate because the clipping routine uses a small epsilon.'),
    stage('ties', 'Audit the top-k boundary', 'How many tokens have positive sampling probability for logits (2, 1, 1, 0) and top_k = 2?', '3', [['2', '2'], ['3', '3'], ['4', '4']], 'The threshold is 1. The code removes scores strictly below it, so both tied scores of 1 remain. This implementation can keep more than k tokens when boundary scores tie.'),
    stage('checkpoint', 'Keep token meanings with weights', 'Which saved field allows token IDs to be mapped back to the original words and punctuation?', 'tokenizer', [['metadata', 'metadata["training_steps"]'], ['state', 'model_state alone'], ['tokenizer', 'tokenizer, containing its vocabulary']], 'Weights and architecture are insufficient if the token-ID vocabulary mapping is lost. The tokenizer state preserves that mapping.'),
  ],
})

const activations = lab({
  id: 'transformer-activations-notebook-lab', title: 'Notebook Lab: Cache Axes and Context Indices',
  prompt: 'Open Homework2026/23TransformerActivations.ipynb, S3–S7, with Homework2026/pico_gpt.py. Use the explicitly specified untrained tiny model for shape/lens checks, and the separate supplied activation fixture for the direction check. Neither fixture is a reference-checkpoint measurement.',
  code: `import torch
from pico_gpt import GPTConfig, PicoGPT

# Read-only excerpts of the existing helper's shape operations:
shape_excerpt = """
# pico_gpt.py: attention reshapes
q = q.view(batch_size, sequence_length, self.n_heads, self.head_size).transpose(1, 2)
k = k.view(batch_size, sequence_length, self.n_heads, self.head_size).transpose(1, 2)
scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
# FeedForward.forward: input expands d_model to mlp_ratio * d_model.
hidden = F.gelu(self.input(x))
output = self.dropout(self.output(hidden))
"""

# Lab wrappers around unchanged 23TransformerActivations.ipynb statements.
def decode_stage(model, residual):  # S5
    stage_logits = model.lm_head(model.final_layer_norm(residual))
    return stage_logits

def contrast_direction(hidden, before_period):  # S6
    period_mean = hidden[before_period].mean(dim=0)
    other_mean = hidden[~before_period].mean(dim=0)
    period_direction = period_mean - other_mean
    period_direction = period_direction / period_direction.norm()
    return period_direction

def ranked_context(inputs, flat_index, sequence_length):  # S7
    row = int(flat_index) // sequence_length
    position = int(flat_index) % sequence_length
    start = max(0, position - 5)
    context_ids = inputs[row, start : position + 1]
    return context_ids`,
  fixture: `import torch
from pico_gpt import GPTConfig, PicoGPT
torch.manual_seed(158)
config = GPTConfig(vocab_size=11, block_size=4, d_model=6,
                   n_heads=2, n_layers=2, mlp_ratio=3, dropout=0.0)
model = PicoGPT(config).eval()
inputs = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

# Separate supplied activations; overwrite hidden only for S6.
hidden = torch.tensor([[1., 1.], [3., 1.], [1., 3.], [3., 3.]])
before_period = torch.tensor([True, True, False, False])
flat_index = 6
sequence_length = 4`,
  invocation: `with torch.no_grad():
    logits, loss, cache = model(inputs, targets, return_cache=True)
print(tuple(cache["blocks"][0]["attention_weights"].shape))
print(tuple(cache["blocks"][0]["mlp_hidden"].shape))

# Run the wrapped S7 statements with flat_index = 6.
context_ids = ranked_context(inputs, flat_index, sequence_length)
print(context_ids.tolist())
# Run the wrapped S6 statements with the supplied hidden/mask.
period_direction = contrast_direction(hidden, before_period)
print(period_direction.tolist())

residual = cache["blocks"][-1]["residual_post"]
stage_logits = model.lm_head(model.final_layer_norm(residual))
print(torch.allclose(stage_logits, logits, atol=1e-6))`,
  stages: [
    stage('attention', 'Name all four attention axes', 'What is the shape of block 0’s attention_weights?', 'shape', [['shape', '(2, 2, 4, 4)'], ['channels', '(2, 4, 6)'], ['transpose', '(2, 4, 2, 4)']], 'The cache uses batch, heads, query positions, key positions. Head size is 3, but it is contracted in the query–key product.'),
    stage('mlp', 'Find the expanded hidden width', 'What is the shape of block 0’s mlp_hidden?', 'expanded', [['preserve', '(2, 4, 6)'], ['expanded', '(2, 4, 18)'], ['vocab', '(2, 4, 11)']], 'The MLP first expands the six channels by mlp_ratio = 3. Its later output projection returns to six channels.'),
    stage('index', 'Recover a context from a flat rank', 'For flat_index = 6, what does context_ids.tolist() return?', 'prefix', [['prefix', '[4, 5, 6]'], ['wrongrow', '[0, 1, 2]'], ['future', '[4, 5, 6, 7]']], 'Integer division gives row 1 and remainder gives position 2. The slice ends at position + 1, so it includes the selected token but excludes its future.'),
    stage('direction', 'Normalize the contrast', 'What is period_direction for the supplied two-channel activation fixture?', 'down', [['up', '[0.0, 1.0]'], ['raw', '[0.0, -2.0]'], ['down', '[0.0, -1.0]']], 'The labeled means are (2, 1) and (2, 3). Their difference is (0, −2), whose unit vector is (0, −1).'),
    stage('lens', 'Check the final-block lens', 'What does the final allclose check print?', 'true', [['false', 'False'], ['true', 'True'], ['loss', 'It compares a scalar loss with a logit tensor and raises an error.']], 'The final residual is passed through exactly the same final normalization and head as in the actual forward pass.'),
  ],
})

const sae = lab({
  id: 'sparse-autoencoders-notebook-lab', title: 'Notebook Lab: A Tiny SAE and Exact Thresholds',
  prompt: 'Open Homework2026/24SparseAutoencoders.ipynb, S3–S5. Instantiate its SparseAutoencoder class using the explicitly overwritten weights below. These are toy values, not learned checkpoint weights. The separate feature tables make each metric check independent of earlier answers.',
  code: `import torch
from torch import nn

# 24SparseAutoencoders.ipynb, S3: original class.
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dimension, number_of_features):
        super().__init__()
        self.input_dimension = input_dimension
        self.number_of_features = number_of_features
        self.encoder = nn.Linear(input_dimension, number_of_features)
        self.decoder_directions = nn.Parameter(
            torch.empty(number_of_features, input_dimension)
        )
        self.decoder_bias = nn.Parameter(torch.zeros(input_dimension))
        nn.init.kaiming_uniform_(self.decoder_directions)
        self.normalize_decoder()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder_directions)
            self.encoder.bias.zero_()

    def encode(self, activations):
        centered = activations - self.decoder_bias
        return torch.relu(self.encoder(centered))

    def forward(self, activations):
        features = self.encode(activations)
        reconstruction = features @ self.decoder_directions + self.decoder_bias
        return reconstruction, features

    @torch.no_grad()
    def normalize_decoder(self):
        norms = self.decoder_directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.decoder_directions.div_(norms)

# Lab wrappers around the notebook's unchanged S4/S5 calculations.
def sae_loss(reconstruction, batch, features, LAMBDA_L1):
    reconstruction_loss = ((reconstruction - batch) ** 2).mean()
    sparsity_loss = features.abs().sum(dim=-1).mean()
    total_loss = reconstruction_loss + LAMBDA_L1 * sparsity_loss
    return total_loss

def threshold_metrics(validation_features):
    mean_l0 = (validation_features > 1e-4).float().sum(dim=1).mean()
    dead_fraction = (validation_features.max(dim=0).values <= 1e-4).float().mean()
    return mean_l0, dead_fraction`,
  fixture: `# Use the original class definition above.
sae = SparseAutoencoder(input_dimension=2, number_of_features=3)
directions = torch.tensor([[1., 0.], [0., 1.], [0.6, 0.8]])
with torch.no_grad():
    sae.decoder_directions.copy_(directions)
    sae.encoder.weight.copy_(directions)
    sae.encoder.bias.zero_()
    sae.decoder_bias.copy_(torch.tensor([1., 1.]))
batch = torch.tensor([[2., 1.], [1., 3.]])

# Independent supplied feature table for the L1 check:
l1_features = torch.tensor([[1., 0., 0.6], [0., 2., 1.6]])
# Independent table for both threshold-based metrics:
validation_features = torch.tensor([
    [0., 1e-4, 2.], [0., 0., 0.], [0., 2e-4, 1.]
])`,
  invocation: `reconstruction, features = sae(batch)
print(features[0].tolist())
print(reconstruction[1].tolist())
print(l1_features.abs().sum(dim=-1).mean().item())
# Run the wrapped S5 metric statements with validation_features above.
mean_l0, dead_fraction = threshold_metrics(validation_features)
print(mean_l0.item())
print(dead_fraction.item())
# Interpret displayed decimals with ordinary float32 rounding.`,
  stages: [
    stage('encode', 'Center before encoding', 'What is features[0], rounded to two decimal places?', 'centered', [['raw', '[2.00, 1.00, 2.00]'], ['centered', '[1.00, 0.00, 0.60]'], ['bias', '[1.00, 0.00, 1.00]']], 'Subtracting decoder bias yields (1, 0); the three encoder row dot products are 1, 0, and 0.6, all preserved by ReLU.'),
    stage('decode', 'Add the decoder bias back', 'What is reconstruction[1], rounded to two decimal places?', 'decoded', [['nobias', '[0.96, 3.28]'], ['original', '[1.00, 3.00]'], ['decoded', '[1.96, 4.28]']], 'The centered second row is (0, 2), encoded as (0, 2, 1.6). Its reconstruction is 2(0,1) + 1.6(0.6,0.8) + (1,1).'),
    stage('l1', 'Sum features, then average examples', 'What is the mean per-example L1 of l1_features?', '2.6', [['averageall', 'Approximately 0.8667'], ['2.6', '2.6'], ['sumall', '5.2']], 'The row sums are 1.6 and 3.6, averaging to 2.6. Averaging all six entries would divide the penalty by the feature width.'),
    stage('l0', 'Use the strict active threshold', 'What is mean_l0 for validation_features?', '1', [['4thirds', '4/3'], ['1', '1'], ['2', '2']], 'The active rule is strictly greater than 1e-4. Row counts are 1, 0, and 2, averaging to 1; equality is not active.'),
    stage('dead', 'Distinguish per-example sparsity from dead features', 'What is dead_fraction for validation_features, approximately?', 'third', [['twothirds', '0.6666667'], ['none', '0'], ['third', '0.3333333']], 'Only feature column 0 never exceeds the threshold. Column 1 is active on the third example even though it is zero or at threshold elsewhere.'),
  ],
})

const steering = lab({
  id: 'feature-steering-notebook-lab', title: 'Notebook Lab: Controlled Residual Edits',
  prompt: 'Open Homework2026/25FeatureSteering.ipynb, S5, S9, and S10. Copy the quoted functions, then supply the fixed two-channel fixture below instead of loading an SAE. The constant feature-value helper is a deliberate test double, not a claim about a learned encoder. All answers are independent of generated prose and reference checkpoints.',
  code: `# 25FeatureSteering.ipynb, S5
def ablate_feature(layer_index, residual):
    if layer_index != target_layer:
        return residual
    feature_values = selected_feature_values(residual)
    return residual - feature_values.unsqueeze(-1) * feature_direction

def amplify_feature(layer_index, residual, factor=2.0):
    if layer_index != target_layer:
        return residual
    feature_values = selected_feature_values(residual)
    extra = (factor - 1.0) * feature_values.unsqueeze(-1)
    return residual + extra * feature_direction

def make_random_control(direction):
    def random_control(layer_index, residual):
        if layer_index != target_layer:
            return residual
        feature_values = selected_feature_values(residual)
        return residual + feature_values.unsqueeze(-1) * direction
    return random_control

# Lab wrapper around the unchanged S9 bias/scaling statements.
def bias_and_scale(next_logits, period_id, period_logit_bias, temperature):
    next_logits[:, period_id] += period_logit_bias
    next_logits = next_logits / temperature
    return next_logits

# Lab wrapper around the unchanged S10 suffix/count statements.
def period_statistic(ids, prompt_length, tokenizer):
    generated_ids = ids[0, prompt_length:].tolist()
    tokens = [tokenizer.itos[index] for index in generated_ids]
    ordinary_tokens = [
        token for token in tokens
        if token not in {"<pad>", "<unk>", "<bos>", "<eos>"}
    ]
    number_of_periods = ordinary_tokens.count(".")
    return round(100 * number_of_periods / max(1, len(ordinary_tokens)), 2)`,
  fixture: `import torch
from types import SimpleNamespace
target_layer = 1
feature_direction = torch.tensor([0.6, 0.8])
residual = torch.tensor([[[3., 2.]]])
def selected_feature_values(residual):
    return torch.full(residual.shape[:-1], 2.0)
random_direction = torch.tensor([-0.8, 0.6])  # unit norm

# Independent raw-logit fixture: token 0 is period.
next_logits = torch.tensor([[2., 1.]])
period_id = 0
period_logit_bias = -4.0
temperature = 0.5

# Independent suffix-statistics fixture:
tokenizer = SimpleNamespace(itos=["<bos>", "Ava", ".", "<eos>"])
ids = torch.tensor([[0, 1, 1, 2, 3]])
prompt_length = 2`,
  invocation: `print(ablate_feature(1, residual).tolist())
print(amplify_feature(1, residual, factor=3.0).tolist())
print(ablate_feature(0, residual).tolist())
print(make_random_control(random_direction)(1, residual).tolist())
# Execute the wrapped S9 bias/scaling lines exactly once.
next_logits = bias_and_scale(next_logits, period_id, period_logit_bias, temperature)
print(next_logits.tolist())
# Execute the wrapped S10 suffix/count lines with the supplied ids.
print(period_statistic(ids, prompt_length, tokenizer))`,
  stages: [
    stage('ablate', 'Subtract only the selected contribution', 'What is ablate_feature(1, residual), rounded to two decimals?', 'ablate', [['zero', '[[[0.00, 0.00]]]'], ['ablate', '[[[1.80, 0.40]]]'], ['direction', '[[[2.40, 1.20]]]']], 'The contribution is coefficient 2 times direction (0.6,0.8). Subtracting it leaves the rest of the stream intact.'),
    stage('amplify', 'Account for the contribution already present', 'What is amplify_feature(1, residual, factor=3.0), rounded to two decimals?', 'factor', [['overadd', '[[[6.60, 6.80]]]'], ['factor', '[[[5.40, 5.20]]]'], ['double', '[[[4.20, 3.60]]]']], 'Tripling the represented contribution adds two extra copies, not three: (factor − 1) × coefficient = 4.'),
    stage('layer', 'Restrict the intervention site', 'What is ablate_feature(0, residual)?', 'unchanged', [['ablate', '[[[1.8, 0.4]]]'], ['zero', '[[[0.0, 0.0]]]'], ['unchanged', '[[[3.0, 2.0]]]']], 'The layer guard returns the original residual when the hook is called at a non-target layer.'),
    stage('random', 'Match edit magnitude, change orientation', 'What is make_random_control(random_direction)(1, residual), rounded to two decimals?', 'random', [['unit', '[[[2.20, 2.60]]]'], ['random', '[[[1.40, 3.20]]]'], ['feature', '[[[4.20, 3.60]]]']], 'The control adds 2 times a unit random direction, preserving the norm of a factor-2 feature-amplification edit.'),
    stage('bias', 'Trace operation order', 'After the raw-logit bias and temperature lines run once, what is next_logits?', 'before', [['after', '[[0.0, 2.0]]'], ['before', '[[-4.0, 2.0]]'], ['unscaled', '[[-2.0, 1.0]]']], 'First 2 + (−4) = −2, then division by 0.5 gives −4. Applying the bias after temperature would be a different experiment.'),
    stage('statistics', 'Use the ordinary-token denominator', 'What periods_per_100_tokens value is printed for the supplied suffix?', '50', [['33', '33.33'], ['20', '20.0'], ['50', '50.0']], 'After the two-token prompt, the suffix is Ava, period, EOS. EOS counts as a generated token elsewhere, but is removed from this statistic’s denominator, leaving one period in two ordinary tokens.'),
  ],
})

// Field labels name the exact expressions students trace; this also lets the
// repository's Python verification script check numeric keys independently.
const expressions: Record<string, Record<string, string>> = {
  [pico.id]: { logits: 'tuple(logits.shape)', loss: 'target_ids.numel()', tying: 'model.lm_head.weight is model.token_embedding.weight', causality: 'torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=1e-6)', generation: 'generated_ids.shape[1]' },
  [training.id]: { target: 'targets.tolist()', shape: 'tuple(inputs.shape)', ties: '(probabilities > 0).sum().item()' },
  [activations.id]: { attention: 'tuple(cache["blocks"][0]["attention_weights"].shape)', mlp: 'tuple(cache["blocks"][0]["mlp_hidden"].shape)', index: 'context_ids.tolist()', direction: 'period_direction.tolist()', lens: 'torch.allclose(stage_logits, logits, atol=1e-6)' },
  [sae.id]: { encode: 'features[0].tolist()', decode: 'reconstruction[1].tolist()', l1: 'l1_features.abs().sum(dim=-1).mean().item()', l0: 'mean_l0.item()', dead: 'dead_fraction.item()' },
  [steering.id]: { ablate: 'ablate_feature(1, residual).tolist()', amplify: 'amplify_feature(1, residual, factor=3.0).tolist()', layer: 'ablate_feature(0, residual).tolist()', random: 'make_random_control(random_direction)(1, residual).tolist()', bias: 'next_logits.tolist()', statistics: 'period_statistic(ids, prompt_length, tokenizer)' },
}
for (const question of [pico, training, activations, sae, steering]) {
  for (const checkpoint of question.stages) {
    checkpoint.fields[0].label = expressions[question.id]?.[checkpoint.id] ?? checkpoint.fields[0].label
  }
}

export const transformerInterpretabilityLabs = { pico, training, activations, sae, steering }
