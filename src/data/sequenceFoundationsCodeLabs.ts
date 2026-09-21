import { defineCodeLabQuestion, type CodeLabQuestionSpec, type CodeLabStageSpec } from '../lib/codeLab'

function stage(id: string, title: string, prompt: string, correctOptionId: string, options: [string, string][], successCopy: string, kind: CodeLabStageSpec['kind'] = 'executionTrace'): CodeLabStageSpec {
  return { id, kind, title, prompt, fields: [{ id: 'answer', label: prompt, correctOptionId, options: options.map(([key, label]) => ({ id: key, label })) }], successCopy }
}
function lab(id: string, filename: string, title: string, instructions: string, code: string, fixture: string, invocation: string, stages: [CodeLabStageSpec, ...CodeLabStageSpec[]], explanation: string): CodeLabQuestionSpec {
  return defineCodeLabQuestion({ id: `${id}-notebook-lab`, kind: 'codeLab', title: `Notebook Lab: ${title}`, prompt: `Trace the actual implementation in Homework2026/${filename}. The quoted implementation and the exact test invocation below contain the evidence needed for each checkpoint.`, instructions, language: 'python', variantId: `${id}-notebook-v1`, datasetId: `${id}-notebook-v1`, code, fixtureTitle: 'Explicit deterministic trace fixture', fixture, invocationTitle: 'Exact statements to evaluate', invocation, stages, hints: ['Track each tensor axis separately. Shapes do not depend on learned numeric weights.', 'Use the supplied invocation, not a training run or a different default configuration.'], hintSchedule: [2, 4], reveal: { explanation } })
}

const cnnCode = `import torch
import torch.nn as nn

model=nn.Sequential(
    nn.Conv2d(1,32,3,padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2,2),
    nn.Conv2d(32,64,3,padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2,2),
    nn.Conv2d(64,128,3,padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2,2),
    nn.Flatten(),
    nn.Linear(3072,512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512,128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128,32),
    nn.ReLU(),
    nn.Linear(32,7)
)`

const autoencoderCode = `import torch
import torch.nn as nn

encoder=nn.Sequential(
    nn.Dropout(0.1),
    nn.Conv2d(1,32,3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32,64,3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2)
)

decoder=nn.Sequential(
    nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(32,1,3,stride=2,padding=1,output_padding=1),
    nn.Sigmoid()
)

autoencoder=nn.Sequential(encoder,decoder)`

const tokenCode = `import re
from collections import Counter
import torch
from torch import nn

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\\d+|[^\\w\\s]")

def tokenize(text):
    return TOKEN_PATTERN.findall(text)

stories = [
    "Ava found a blue kite. The kite flew high!",
    "Ben found a red ball. The ball rolled away.",
    'Cora said, "Please bring the blue ball."',
    "A dog found the ball. Ava thanked the dog.",
]

special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
counts = Counter(token for story in stories for token in tokenize(story))

# Frequency first; alphabetical order breaks ties reproducibly.
ordinary_tokens = sorted(counts, key=lambda token: (-counts[token], token))
vocabulary = special_tokens + ordinary_tokens
stoi = {token: index for index, token in enumerate(vocabulary)}
itos = {index: token for token, index in stoi.items()}

def encode(text, add_special_tokens=True):
    ids = [stoi.get(token, stoi["<unk>"]) for token in tokenize(text)]
    if add_special_tokens:
        ids = [stoi["<bos>"]] + ids + [stoi["<eos>"]]
    return ids

def decode_to_tokens(ids):
    return [itos[int(index)] for index in ids]

context_length = 6

def make_window(stream, start, length=context_length):
    inputs = stream[start : start + length]
    targets = stream[start + 1 : start + length + 1]
    return inputs, targets`

const attentionCode = `import math
import torch
from torch import nn

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 6)
        self.activation = nn.ReLU()
        self.second = nn.Linear(6, 4)

    def forward(self, x):
        hidden = self.activation(self.first(x))
        return self.second(hidden)

class SelfAttentionHead(nn.Module):
    def __init__(self, input_dimension, head_dimension):
        super().__init__()
        self.head_dimension = head_dimension
        self.query = nn.Linear(input_dimension, head_dimension, bias=False)
        self.key = nn.Linear(input_dimension, head_dimension, bias=False)
        self.value = nn.Linear(input_dimension, head_dimension, bias=False)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dimension)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ v
        return output, weights`

const causalCode = `import math
import torch
from torch import nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, number_of_heads, maximum_context):
        super().__init__()
        assert d_model % number_of_heads == 0
        self.number_of_heads = number_of_heads
        self.head_size = d_model // number_of_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.projection = nn.Linear(d_model, d_model)

        mask = torch.tril(torch.ones(maximum_context, maximum_context, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, maximum_context, maximum_context))

    def forward(self, x, return_weights=False):
        batch_size, sequence_length, d_model = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, sequence_length, self.number_of_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.number_of_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.number_of_heads, self.head_size).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        mask = self.causal_mask[:, :, :sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, sequence_length, d_model)
        output = self.projection(attended)

        if return_weights:
            return output, weights
        return output`

export const sequenceNotebookLabs: Record<string, CodeLabQuestionSpec> = {
  cnns: lab('cnns', '16CNNs.ipynb', 'Follow the CNN Image Shapes',
    'The model below is quoted from the notebook. The explicit two-image fixture replaces the downloaded face data; it has the same per-image shape (1, 50, 37). No training or accuracy measurement is required. All shape tuples are (batch, channels, height, width) until Flatten; ordinary pooling rounds spatial sizes down. Python slicing excludes the end index.',
    cnnCode,
    `X_trace = torch.zeros(2, 1, 50, 37)
y_trace = torch.tensor([0, 6])`,
    `model.eval()
with torch.no_grad():
    first_pool = model[:4](X_trace)
    second_pool = model[:8](X_trace)
    flat = model[:13](X_trace)
    logits = model(X_trace)
    predicted_ids = logits.argmax(dim=1)
    loss = nn.CrossEntropyLoss()(logits, y_trace)

dropout_active = model[15].training`,
    [
      stage('first-pool', 'First spatial reduction', 'What is tuple(first_pool.shape)?', 'shape', [['shape', '(2, 32, 25, 18)'], ['same', '(2, 32, 50, 37)'], ['round-up', '(2, 32, 25, 19)']], 'Padding 1 preserves the 50 × 37 convolution size. Stride-2 pooling makes it 25 × 18; the odd width rounds down.'),
      stage('second-pool', 'Track channels separately', 'What is tuple(second_pool.shape)?', 'shape', [['first-channels', '(2, 32, 12, 9)'], ['shape', '(2, 64, 12, 9)'], ['double', '(2, 64, 25, 18)']], 'The second convolution creates 64 channels without changing spatial size; the next pool makes 25 × 18 into 12 × 9.'),
      stage('flatten', 'Connect images to the dense layer', 'What is tuple(flat.shape)?', 'shape', [['batch', '(6144,)'], ['spatial', '(2, 24)'], ['shape', '(2, 3072)']], 'After the third pool there are 128 × 6 × 4 = 3072 entries per image. Flatten preserves the batch dimension.'),
      stage('logits', 'One class-score row per image', 'What is tuple(logits.shape)?', 'shape', [['shape', '(2, 7)'], ['labels', '(2,)'], ['hidden', '(2, 32)']], 'The final Linear layer maps 32 hidden features to seven class scores, retaining two observations.'),
      stage('prediction', 'Reduce the class axis', 'What is tuple(predicted_ids.shape)?', 'shape', [['scores', '(2, 7)'], ['shape', '(2,)'], ['column', '(2, 1)']], 'argmax(dim=1) returns one integer class ID per row. The exact IDs depend on weights, but this shape does not.'),
      stage('mode', 'Read the dropout mode', 'What is dropout_active after this invocation?', 'false', [['false', 'False'], ['true', 'True'], ['depends', 'It depends on the randomly initialized weights.']], 'model.eval() recursively puts child modules into evaluation mode, so the dropout training flag is False. no_grad separately disables derivative recording.'),
    ], 'The pipeline changes spatial extent and channel count independently. The trace uses fixed input shapes and evaluation mode, without grading any random class choice or trained accuracy.'),

  autoencoders: lab('autoencoders', '17AutoEncoders.ipynb', 'Trace Corruption and Reconstruction',
    'The encoder and decoder are the notebook definitions. The two-image fixture replaces the download and uses its cropped shape (1, 48, 36). The encoded tensor has 64 × 12 × 9 = 6912 scalars per image—more than the input’s 1728—so this architecture is not an undercomplete scalar bottleneck. For ConvTranspose2d, output size is (input − 1) × stride − 2 × padding + kernel_size + output_padding. No learned image quality is graded.',
    autoencoderCode,
    `batch_X = torch.zeros(2, 1, 48, 36)
probe = torch.tensor([0.9])`,
    `autoencoder.eval()
with torch.no_grad():
    encoded = encoder(batch_X)
    first_expansion = decoder[0](encoded)
    reconstructed = autoencoder(batch_X)
    MSE = nn.MSELoss()(autoencoder(batch_X),batch_X)
    eval_pixel = encoder[0](probe)

# Separate mutation test: same hyperparameters except output_padding=0.
alternate_first = nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=0)
alternate_shape = tuple(alternate_first(encoded).shape)`,
    [
      stage('encoded', 'Encoder dimensions', 'What is tuple(encoded.shape)?', 'shape', [['shape', '(2, 64, 12, 9)'], ['half', '(2, 64, 24, 18)'], ['flat', '(2, 6912)']], 'Both padded convolutions preserve spatial extent. The two pools halve 48 × 36 to 24 × 18 and then 12 × 9.'),
      stage('expansion', 'First decoder expansion', 'What is tuple(first_expansion.shape)?', 'shape', [['same', '(2, 32, 12, 9)'], ['shape', '(2, 32, 24, 18)'], ['off-by-one', '(2, 32, 23, 17)']], 'The spatial rule is (n − 1) × 2 − 2 + 3 + 1 = 2n. The first transpose convolution gives 32 channels and 24 × 18 spatial dimensions.'),
      stage('reconstruction', 'Recover the input layout', 'What is tuple(reconstructed.shape)?', 'shape', [['rgb', '(2, 3, 48, 36)'], ['flat', '(2, 1728)'], ['shape', '(2, 1, 48, 36)']], 'The second transpose convolution doubles spatial extent again and outputs one channel; Sigmoid preserves the shape.'),
      stage('target', 'Identify the supervised target', 'In the displayed MSE expression, what tensor is the target (second argument)?', 'clean', [['clean', 'batch_X, the unchanged clean batch.'], ['encoded', 'encoded, the encoder output.'], ['prediction', 'The model reconstruction itself.']], 'The notebook keeps batch_X as the target. During training, dropout corrupts data inside the model rather than changing that target tensor.'),
      stage('dropout', 'Evaluation disables new corruption', 'What is eval_pixel.tolist(), to one decimal place?', '09', [['zero', '[0.0]'], ['09', '[0.9]'], ['one', '[1.0]']], 'Evaluation makes the Dropout module the identity. In training, retained values are instead divided by 0.9; the notebook’s manually created binary-noise image does not include that rescaling.'),
      stage('output-padding', 'Diagnose a one-pixel change', 'What is alternate_shape?', 'smaller', [['same', '(2, 32, 24, 18)'], ['smaller', '(2, 32, 23, 17)'], ['larger', '(2, 32, 25, 19)']], 'Changing only output_padding from 1 to 0 removes one output position along each spatial axis of this layer.', 'diagnoseMutation'),
    ], 'The actual decoder expands 12 × 9 to 24 × 18 and then 48 × 36. The clean batch is the MSE target. Evaluation turns off internal dropout but cannot itself repair corruption already in an input image.'),

  tokenization: lab('tokenization', '18Tokenization.ipynb', 'Trace Tokens, IDs, and Windows',
    'The tokenizer, four stories, vocabulary construction, encoding functions, and window function are quoted from S2–S7. Run the quoted definitions before the fixture and invocation. Case is preserved, the ordinary vocabulary uses decreasing frequency then alphabetical order, and special tokens occupy IDs 0–3. Embedding values are not needed: only their shapes are graded.',
    tokenCode,
    `embedding_dimension = 8
maximum_context = 12
token_embedding = nn.Embedding(len(vocabulary), embedding_dimension)
position_embedding = nn.Embedding(maximum_context, embedding_dimension)
encoded_story = encode(stories[0])
example_ids = torch.tensor([encoded_story[:maximum_context]])

token_stream = []
for story in stories:
    token_stream.extend(encode(story))
token_stream = torch.tensor(token_stream)`,
    `punctuation_probe = tokenize("kite! kite.")
period_id = stoi["."]
unknown_tokens = decode_to_tokens(encode("Ava found a spaceship."))
positions = torch.arange(example_ids.shape[1])
input_vectors = token_embedding(example_ids) + position_embedding(positions)
inputs, targets = make_window(token_stream, start=0)
target_tokens = decode_to_tokens(targets)
overlap_matches = torch.equal(inputs[1:], targets[:-1])`,
    [
      stage('punctuation', 'Apply the tokenizer', 'What is punctuation_probe?', 'separate', [['attached', '["kite!", "kite."]'], ['separate', '["kite", "!", "kite", "."]'], ['removed', '["kite", "kite"]']], 'The word alternative matches kite, and the punctuation alternative matches each following mark separately.'),
      stage('period', 'Trace a vocabulary ID', 'What is period_id?', '4', [['0', '0'], ['4', '4'], ['6', '6']], 'The four special IDs come first. Period is the most frequent ordinary token, so it receives ID 4.'),
      stage('unknown', 'Preserve sequence boundaries', 'What is unknown_tokens?', 'bounded', [['bounded', '["<bos>", "Ava", "found", "a", "<unk>", ".", "<eos>"]'], ['missing', '["<bos>", "Ava", "found", "a", ".", "<eos>"]'], ['literal', '["<bos>", "Ava", "found", "a", "spaceship", ".", "<eos>"]']], 'spaceship is not in the fixed vocabulary, so it maps to <unk>; punctuation and boundary tokens remain present.'),
      stage('embedding-shape', 'Broadcast the position vectors', 'What is tuple(input_vectors.shape)?', 'shape', [['shape', '(1, 12, 8)'], ['no-batch', '(12, 8)'], ['concatenate', '(1, 12, 16)']], 'The twelve 8-coordinate position vectors are added to, not concatenated with, the token vectors; broadcasting preserves the single batch axis.'),
      stage('targets', 'Read the shifted slice', 'What is target_tokens for start=0?', 'next', [['same', '["<bos>", "Ava", "found", "a", "blue", "kite"]'], ['next', '["Ava", "found", "a", "blue", "kite", "."]'], ['later', '["found", "a", "blue", "kite", ".", "The"]']], 'The first input is <bos>; its target is Ava. The six targets extend one stream position past the six inputs.'),
      stage('overlap', 'Check the alignment invariant', 'What is overlap_matches?', 'true', [['false', 'False'], ['true', 'True'], ['error', 'It raises a shape mismatch error.']], 'Both slices contain the same five IDs: Ava, found, a, blue, kite. Their equality is the intended next-token alignment.'),
    ], 'The exact tokenization convention determines the vocabulary and every shifted label. Learned embedding values do not alter token IDs, slice alignment, or tensor dimensions.'),

  attention: lab('attention', '19Attention.ipynb', 'Trace Custom Modules and One Attention Head',
    'The displayed TinyMLP and SelfAttentionHead are the complete class definitions from S7 and S9. The fixture quotes the S6–S7 bridge setup and S4 token vectors. Compare functions after copying parameters; do not guess randomly initialized numerical outputs. Linear(a,b) with bias has a×b weights plus b biases; ReLU has no trainable tensors.',
    attentionCode,
    `sequential_mlp = nn.Sequential(
    nn.Linear(4, 6),
    nn.ReLU(),
    nn.Linear(6, 4),
)
bridge_input = torch.tensor([[1.0, -1.0, 0.5, 2.0]])
custom_mlp = TinyMLP()
custom_mlp.first.load_state_dict(sequential_mlp[0].state_dict())
custom_mlp.second.load_state_dict(sequential_mlp[2].state_dict())

token_vectors = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
])
head = SelfAttentionHead(input_dimension=4, head_dimension=2)
batched_tokens = token_vectors.unsqueeze(0)`,
    `sequential_output = sequential_mlp(bridge_input)
custom_output = custom_mlp(bridge_input)
bridge_outputs_match = torch.allclose(sequential_output, custom_output)
custom_parameter_names = [name for name, _ in custom_mlp.named_parameters()]
custom_parameter_count = sum(parameter.numel() for parameter in custom_mlp.parameters())
learned_output, learned_weights = head(batched_tokens)
row_sums = learned_weights.sum(dim=-1)`,
    [
      stage('bridge', 'Compare two registered models', 'What is bridge_outputs_match?', 'true', [['true', 'True'], ['false', 'False: Sequential cannot be equivalent to a custom module.'], ['random', 'It depends on initialization even after both linear layers are copied.']], 'The copied linear parameters and identical intervening ReLU make both forward computations the same function.'),
      stage('registration', 'Identify trainable tensors', 'What is custom_parameter_names?', 'four', [['all', '["first.weight", "first.bias", "activation.weight", "second.weight", "second.bias"]'], ['four', '["first.weight", "first.bias", "second.weight", "second.bias"]'], ['none', '[]']], 'Layers assigned to self are registered, but ReLU has no parameter tensors. The two Linear layers each register a weight and bias.'),
      stage('count', 'Count scalar parameters', 'What is custom_parameter_count?', '58', [['48', '48'], ['58', '58'], ['10', '10']], 'The first layer has 4×6 + 6 = 30 parameters and the second has 6×4 + 4 = 28, totaling 58.'),
      stage('weight-shape', 'Trace all query–key pairs', 'What is tuple(learned_weights.shape)?', '144', [['142', '(1, 4, 2)'], ['144', '(1, 4, 4)'], ['122', '(1, 2, 2)']], 'The input batch has four token positions. q and k each have shape (1,4,2), and q @ k.transpose(-2,-1) creates (1,4,4).'),
      stage('output-shape', 'Mix value vectors', 'What is tuple(learned_output.shape)?', '142', [['142', '(1, 4, 2)'], ['144', '(1, 4, 4)'], ['12', '(1, 2)']], 'The (1,4,4) weights mix (1,4,2) value vectors, giving one two-channel output per query.'),
      stage('normalize', 'Identify the normalized axis', 'What is row_sums, up to floating-point roundoff?', 'ones', [['column', 'A shape-(1,4) tensor with unconstrained entries.'], ['ones', 'A shape-(1,4) tensor whose entries are all 1.'], ['zeros', 'A shape-(1,4) tensor whose entries are all 0.']], 'softmax(dim=-1) normalizes over the four key positions independently for each query row.'),
    ], 'Custom modules participate in the same parameter-registration system as Sequential. Attention uses that machinery for three parallel projections and query–key/value matrix products, with a separate normalized retrieval for each query.'),

  'causal-attention': lab('causal-attention', '20CausalAttention.ipynb', 'Trace Masks, Heads, and Prefix Invariance',
    'The class is quoted from S3. The raw-score fixture and masking expressions are from S2; the model configuration and sequence shapes match S4–S5. Fixed seed 158 makes the fixture repeatable, but no randomly initialized value is an answer. register_buffer stores a tensor with the module without making it a trainable parameter.',
    causalCode,
    `_ = torch.manual_seed(158)
raw_scores = torch.tensor([
    [2.0, 1.0, 0.0, -1.0],
    [0.0, 2.0, 1.0, 0.0],
    [1.0, 0.0, 2.0, 1.0],
    [0.0, 1.0, 0.0, 2.0],
])
attention = CausalSelfAttention(
    d_model=8,
    number_of_heads=2,
    maximum_context=10,
)
example = torch.randn(3, 6, 8)
first_sequence = torch.randn(1, 6, 8)
second_sequence = first_sequence.clone()
second_sequence[:, 3:] = torch.randn(1, 3, 8)`,
    `causal_mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
masked_scores = raw_scores.masked_fill(~causal_mask, float("-inf"))
causal_weights = torch.softmax(masked_scores, dim=-1)
attention_output, attention_weights = attention(example, return_weights=True)
mask_is_parameter = "causal_mask" in dict(attention.named_parameters())
first_output = attention(first_sequence)
second_output = attention(second_sequence)
prefix_matches = torch.allclose(first_output[:, :3], second_output[:, :3], atol=1e-6)`,
    [
      stage('mask', 'Read a lower-triangular row', 'What is causal_mask[2].int().tolist()?', 'prefix', [['strict', '[1, 1, 0, 0]'], ['prefix', '[1, 1, 1, 0]'], ['suffix', '[0, 0, 1, 1]']], 'A lower triangle including its diagonal allows row 2 to access keys 0, 1, and 2.'),
      stage('first-row', 'Normalize one allowed key', 'What is causal_weights[0].tolist()?', 'one', [['one', '[1.0, 0.0, 0.0, 0.0]'], ['uniform', '[0.25, 0.25, 0.25, 0.25]'], ['zero', '[0.0, 0.0, 0.0, 0.0]']], 'Every future key receives a −infinity score before softmax, leaving one positive exponential to normalize to 1.'),
      stage('head-size', 'Split the channel dimension', 'What is attention.head_size?', '4', [['2', '2'], ['4', '4'], ['8', '8']], 'The constructor divides d_model 8 by two heads, giving four coordinates per head.'),
      stage('weights', 'Keep batch, head, query, and key axes', 'What is tuple(attention_weights.shape)?', 'shape', [['max-context', '(3, 2, 10, 10)'], ['shape', '(3, 2, 6, 6)'], ['channels', '(3, 2, 6, 4)']], 'The stored 10×10 mask is sliced to the actual six-position sequence. Each of the two heads in each of three batches has a 6×6 weight matrix.'),
      stage('buffer', 'Distinguish a buffer from a parameter', 'What is mask_is_parameter?', 'false', [['true', 'True'], ['false', 'False'], ['depends', 'It changes from True to False after evaluation.']], 'The registered causal mask is in named_buffers and state_dict, not named_parameters; the optimizer does not learn its entries.'),
      stage('prefix', 'Test the absence of future leakage', 'What is prefix_matches?', 'true', [['false', 'False, because the changed suffix shares the same projection weights.'], ['true', 'True, within the specified numerical tolerance.'], ['random', 'It is equally likely to be True or False because the weights are random.']], 'The two sequences have identical positions 0–2. None of those queries may access positions 3–5, so the changed suffix cannot alter the prefix outputs.', 'distinguishingTest'),
    ], 'Causality is implemented before softmax, head splitting preserves all sequence positions, and the fixed mask is a buffer. Shared-prefix invariance tests the resulting information restriction directly.'),
}
