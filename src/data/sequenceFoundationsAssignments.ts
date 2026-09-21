import type { AssignmentSpec, MultipleChoicePart, MultipleChoiceQuestionSpec } from '../types'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import { sequenceNotebookLabs } from './sequenceFoundationsCodeLabs'

export interface SequenceFoundationsVisualDataset {
  id: string
  kind: 'sequenceFoundationsVisual'
  mode: 'filter' | 'pool' | 'denoise' | 'bottleneck' | 'tokenize' | 'window' | 'embedding' | 'retrieve' | 'matrix' | 'mask' | 'heads' | 'notes'
  title: string
  caption: string
  notes?: string[]
}
function visual(id: string, mode: SequenceFoundationsVisualDataset['mode'], title: string, caption: string, notes?: string[]): SequenceFoundationsVisualDataset {
  return { id, kind: 'sequenceFoundationsVisual', mode, title, caption, notes }
}
export const sequenceFoundationsVisualDatasets: Record<string, SequenceFoundationsVisualDataset> = {
  cnnFilter: visual('cnnFilter', 'filter', 'A local filter travels across an image', 'Move the highlighted 2 × 2 patch. The response uses the same four filter weights at every location.'),
  cnnPool: visual('cnnPool', 'pool', 'Summarize spatial neighborhoods', 'Compare maximum and mean pooling on identical input data. Values printed in each cell make the comparison independent of color.'),
  cnnEvaluation: visual('cnnEvaluation', 'notes', 'Three separate choices during prediction', 'A classifier has dropout, normalization with running statistics, and learned weights.', ['Training mode: dropout randomly removes and rescales activations; normalization uses batch statistics.', 'Evaluation mode: dropout is inactive; normalization uses stored running statistics.', 'Recording derivatives and updating weights are separate operations. A forward prediction alone does not update the learned parameters.']),
  aeDenoise: visual('aeDenoise', 'denoise', 'Which image is the target?', 'A four-pixel image loses its third pixel. Compare copying the damaged input with a fixed attempted reconstruction.'),
  aeBottleneck: visual('aeBottleneck', 'bottleneck', 'What gets lost in a representation?', 'A two-number code represents a four-pixel image. Select two inputs that differ in within-pair order.'),
  aeNoise: visual('aeNoise', 'notes', 'Random corruption is not an output target', 'During denoising training, corrupt the input while preserving the original clean target.', ['Example clean pixel: 0.9. With dropout probability 0.1, a training pass outputs 0 with probability 0.1, or 0.9 / 0.9 = 1 with probability 0.9.', 'At evaluation, dropout passes the input through unchanged. It does not invent a missing value or remove existing corruption.', 'A bounded output layer keeps reconstruction values in the valid intensity range [0, 1]; it cannot guarantee that they match the original image.']),
  tokensUnits: visual('tokensUnits', 'tokenize', 'Choose the units in the sequence', 'Compare characters with case-sensitive words and separate punctuation.'),
  tokensWindows: visual('tokensWindows', 'window', 'Align inputs with the next token', 'Each row predicts the token immediately after its input token, not the same token.'),
  tokensEmbedding: visual('tokensEmbedding', 'embedding', 'IDs select vectors; positions modify them', 'An integer ID is a lookup label. The coordinates of its learned vector carry the representation.'),
  attentionRetrieve: visual('attentionRetrieve', 'retrieve', 'Where to look versus what to retrieve', 'Change the query or one stored value and inspect the normalized attention weights and weighted output.'),
  attentionMatrix: visual('attentionMatrix', 'matrix', 'One retrieval distribution per query', 'Each query gets its own row of weights. Select a row to inspect the scalar it retrieves.'),
  attentionFlow: visual('attentionFlow', 'notes', 'Three parallel projections', 'An input has shape (batch 2, tokens 5, channels 4). Three separate learned linear transformations each map 4 channels to 3.', ['Queries Q, keys K, and values V each have shape (2, 5, 3).', 'For each batch, Q multiplies the transpose of K to compare every query with every key. Softmax normalizes each query’s scores across keys.', 'The normalized weights multiply V, so each output is a weighted mixture of value vectors. Tokens are mixed here, not during the independent per-token projections.']),
  causalMask: visual('causalMask', 'mask', 'Block future keys before normalizing', 'During next-token prediction, input position i can retrieve from positions 0 through i, including itself.'),
  causalHeads: visual('causalHeads', 'heads', 'Distribute channels among heads', 'Inspect how several retrieval patterns fit inside a fixed channel budget.'),
  causalPosition: visual('causalPosition', 'notes', 'Two controlled sequence comparisons', 'Assume a deterministic attention layer with no dropout and the same fixed weights for every comparison.', ['Prefix test: input sequences A and B have identical vectors at positions 0, 1, 2 but different vectors at positions 3, 4, 5.', 'Position test: one repeated token has the same token vector at all positions. Add distinct position vectors before attention.', 'Unmasked attention without positions simply reorders outputs if its inputs are reordered. A causal mask already introduces prefix asymmetry; explicit position vectors add direct position information.']),
}
function part(id: string, prompt: string, correctOptionId: string, options: [string, string][]): MultipleChoicePart {
  return { id, prompt, correctOptionId, options: options.map(([key, title]) => ({ id: key, title, description: '' })) }
}
function question(id: string, title: string, datasetId: string, prompt: string, instructions: string, parts: MultipleChoicePart[], explanation: string, hints: string[]): MultipleChoiceQuestionSpec {
  return { id, kind: 'multipleChoice', title, datasetId, prompt, instructions, parts, hints, hintSchedule: [2, 4], reveal: { explanation }, validator: validateMultipleChoiceSelections(Object.fromEntries(parts.map(p => [p.id, p.correctOptionId])), 'Use the stated reference conditions and compare the mechanism, not just the final dimensions.') }
}

export const sequenceFoundationsAssignments: AssignmentSpec[] = [
  {
    id: 'cnns', version: 1, displayNumber: 16, published: true, title: 'Convolutional Neural Networks', topic: 'Local Image Features',
    description: 'Explore shared filters, pooling, and prediction modes before tracing the image classifier’s tensor shapes.',
    questions: [
      question('cnns-shared-filter', 'Follow a Shared Local Filter', 'cnnFilter', 'A filter detects patterns using local pixel differences. Its weights are reused at every valid image location.', 'Scored reference: use top row 0 and compare patch left columns 0, 1, and 2. Multiply corresponding entries without flipping the displayed filter.', [
        part('responses', 'What are the responses at left columns 0, 1, 2, respectively?', 'edge', [['constant', '[2, 2, 2]'], ['edge', '[0, 2, 0]'], ['reverse', '[0, −2, 0]']]),
        part('sharing', 'Move the same local pattern one column to the right, away from boundaries. What happens?', 'moves', [['new', 'A newly learned set of weights is needed at the new location.'], ['moves', 'The same response moves one column right because the weights are shared.'], ['fixed', 'The response remains fixed at its original image location.']]),
      ], 'Each filter row subtracts the left pixel from the right pixel. The crossing patch contributes 1 + 1 = 2; constant patches contribute 0. Shared weights let a feature detector respond to the same pattern at different locations.', ['Compute one pair of products per filter row.', 'Distinguish moving the pattern from changing the filter weights.']),
      question('cnns-pooling', 'Pool Without Preserving Every Pixel', 'cnnPool', 'A 2 × 2 maximum-pooling operation keeps one number from each nonoverlapping window.', 'Scored reference: Maximum rule, window size 2 × 2, stride 2, no padding. Changing the control to Mean is unscored exploration.', [
        part('values', 'What is the top output row under maximum pooling?', '45', [['32', '[3, 2]'], ['45', '[4, 5]'], ['25', '[2.5, 2]']]),
        part('loss', 'Can the pooled output uniquely determine all 16 original pixels?', 'no', [['yes', 'Yes, because every input pixel was included in a window.'], ['no', 'No; changing a nonmaximum value without changing its window’s maximum leaves the output unchanged.'], ['channels', 'Yes, if the input has exactly one channel.']]),
      ], 'The first two windows have maxima 4 and 5. Four summaries replace 16 pixels, and the locations and values of nonmaximal pixels are lost. Pooling summarizes local evidence rather than being an invertible image compression.', ['Inspect each 2 × 2 block separately.', 'Try changing an entry smaller than its block maximum.']),
      question('cnns-evaluation', 'Separate Evaluation from Weight Updates', 'cnnEvaluation', 'Predicting on held-out images should not inject training-time dropout noise or update normalization statistics.', 'Use the three distinct mechanisms described in the panel: layer mode, derivative recording, and optimizer updates.', [
        part('mode', 'Which configuration is appropriate for ordinary held-out evaluation?', 'evaluation', [['training', 'Keep training mode so the held-out batch updates the normalization statistics.'], ['evaluation', 'Use evaluation mode and avoid recording derivatives when they are unnecessary.'], ['zero', 'Set all learned weights to zero to disable training.']]),
        part('weights', 'Does selecting evaluation mode erase learned weights or itself make an optimizer step?', 'neither', [['erase', 'It erases the weights.'], ['step', 'It makes one final optimizer step.'], ['neither', 'Neither; it changes mode-dependent layer behavior.']]),
      ], 'Evaluation mode disables dropout and selects stored normalization statistics. Derivative recording is separate, and learned parameters change only when an update operation changes them. This makes train/evaluation comparisons meaningful.', ['Which behavior should depend on a held-out batch?', 'Changing how a layer operates is not the same as changing its learned parameters.']),
      sequenceNotebookLabs.cnns,
    ],
  },
  {
    id: 'autoencoders', version: 1, displayNumber: 17, published: true, title: 'Denoising Autoencoders', topic: 'Reconstruction and Representations',
    description: 'Compare reconstruction targets and information loss, then trace the encoder, decoder, and corruption mechanism.',
    questions: [
      question('autoencoders-target', 'Reward Reconstructing the Clean Image', 'aeDenoise', 'A denoising model receives corrupted pixels but should reconstruct the original clean image.', 'Scored reference: Clean original target; A and B are the fixed candidates shown. The loss is mean squared error over four pixels.', [
        part('winner', 'Which candidate has lower clean-target reconstruction loss?', 'B', [['A', 'A, copying the corrupted input.'], ['B', 'B, reconstructing the missing detail.'], ['tie', 'They tie because both preserve the first pixel.']]),
        part('target', 'If the corrupted input were also used as the target, which behavior would be rewarded?', 'copy', [['restore', 'Restoring the hidden original pixel exactly.'], ['copy', 'Copying the corruption, because that gives zero target error.'], ['none', 'No behavior can obtain zero error.']]),
      ], 'Against the clean image, A has error 0.8²/4 = 0.160; B has (0.1² + 0.2² + 0.1²)/4 = 0.015. If the damaged input becomes the target, copying it earns zero loss. The target defines what the model is encouraged to preserve.', ['Square each pixel difference before averaging.', 'Ask what image would match every target pixel.']),
      question('autoencoders-bottleneck', 'Identify Information Lost in a Code', 'aeBottleneck', 'An encoder maps an image to a representation; a decoder maps that representation back to pixels.', 'Compare both specified input images under the displayed pair-mean encoder and repeat-mean decoder. These fixed operations are a toy example, not a claim about every autoencoder.', [
        part('code', 'How do the representations of images A and B compare?', 'same', [['different', 'Different: both pixel pairs were reversed.'], ['same', 'Identical: both representations are [0.5, 0.5].'], ['four', 'They each retain all four original pixel values.']]),
        part('recover', 'Could any deterministic decoder of only this two-number code perfectly reconstruct both A and B?', 'impossible', [['yes', 'Yes, if the decoder is made deeper.'], ['impossible', 'No: the same code would have to produce two different outputs.'], ['average', 'Yes: the pair averages identify the original order.']]),
      ], 'Both inputs collapse to [0.5, 0.5]. A decoder cannot distinguish them from that code alone. A useful learned representation must preserve information needed for the intended reconstruction; increasing decoder capacity cannot recover distinctions already discarded.', ['Compute each adjacent pair average for both images.', 'A deterministic function has one output for a given input.']),
      question('autoencoders-noise', 'Understand Corruption and Inference', 'aeNoise', 'Training-time corruption can teach a model to use surrounding structure instead of copying every input pixel.', 'Use the exact dropout convention stated in the panel: drop probability 0.1 and divide retained values by 0.9. At evaluation, the dropout transformation is the identity.', [
        part('expectation', 'What is the average dropout output for the clean pixel 0.9 over many training passes?', '09', [['081', '0.81'], ['09', '0.9'], ['1', '1.0']]),
        part('inference', 'At evaluation, a previously corrupted input pixel has value 0. What does disabling dropout do to that input pixel by itself?', 'zero', [['recover', 'Recovers its original clean value.'], ['zero', 'Leaves it at 0; the learned reconstruction layers must infer missing content.'], ['random', 'Replaces it with fresh random noise.']]),
      ], 'The expectation is 0.1 × 0 + 0.9 × 1 = 0.9. Inverted dropout preserves the expected activation, not each individual value. Turning it off prevents new corruption; it does not undo corruption already present in an image.', ['Weight both possible training outputs by their probabilities.', 'Separate the corruption operation from the learned reconstruction function.']),
      sequenceNotebookLabs.autoencoders,
    ],
  },
  {
    id: 'tokenization', version: 1, displayNumber: 18, published: true, title: 'Tokenization and Next-Token Prediction', topic: 'Text as Sequences',
    description: 'Choose token units, separate IDs from embeddings, and align next-token targets before tracing the text pipeline.',
    questions: [
      question('tokenization-units', 'Choose What One Position Means', 'tokensUnits', 'The same text can produce different sequence lengths depending on the units used as tokens.', 'Scored reference: “blue kite!”; characters include the space, whereas word-and-punctuation tokenization omits spaces and separates the exclamation point.', [
        part('length', 'What are the lengths (characters, words-and-punctuation)?', '10-3', [['9-2', '(9, 2)'], ['10-3', '(10, 3)'], ['10-2', '(10, 2)']]),
        part('punctuation', 'With words and punctuation separate, compare “blue kite!” and “blue kite.”. Which token changes?', 'punct', [['kite', 'The word kite changes to a new word token.'], ['punct', 'Only the final punctuation token changes.'], ['all', 'All three token identities change.']]),
      ], 'There are four letters, a space, four letters, and punctuation: 10 characters. Word-and-punctuation tokens are [blue, kite, !]. Separating punctuation reuses the same word token across these two sentence endings.', ['Count the space in character mode.', 'Compare the displayed token boundaries before comparing token identities.']),
      question('tokenization-embeddings', 'Distinguish Labels from Coordinates', 'tokensEmbedding', 'A token ID selects a learned vector; adding a position vector makes the representation position-sensitive.', 'Scored reference: token “kite” at positions 0 and 1 with the displayed fixed vectors. Do not multiply the vector by its token ID.', [
        part('combined', 'What are the combined vectors at positions 0 and 1, respectively?', 'sum', [['same', '[2, −1] and [2, −1]'], ['sum', '[2, 0] and [3, −1]'], ['id', '[14, −7] and [14, −7]']]),
        part('ids', 'If vocabulary IDs are renumbered and the lookup rows are rearranged consistently, what happens to token vectors?', 'unchanged', [['larger', 'Larger IDs necessarily get larger vector magnitudes.'], ['unchanged', 'The vectors can remain unchanged; the IDs are labels.'], ['order', 'Semantic similarity must follow the new numeric ID distances.']]),
      ], 'Componentwise addition gives [2, −1] + [0, 1] = [2, 0] and [2, −1] + [1, 0] = [3, −1]. Renumbering labels does not change a representation when the lookup is updated consistently.', ['Add each position vector coordinate to the matching token coordinate.', 'A label identifies an entry; it does not impose a numeric geometry on that entry.']),
      question('tokenization-shift', 'Line Up the Targets', 'tokensWindows', 'A next-token model predicts one token to the right at every input position.', 'Scored reference: start index 0 and four input tokens. Explore other starts without changing the scored reference.', [
        part('last', 'What target belongs to the final input token “a” in the reference window?', 'blue', [['a', 'a'], ['blue', 'blue'], ['kite', 'kite']]),
        part('overlap', 'How do the last three inputs compare with the first three targets?', 'equal', [['different', 'They must all be different to avoid repeated labels.'], ['equal', 'They are the same ordered sequence: Ava, found, a.'], ['reverse', 'They contain the same tokens in reverse order.']]),
      ], 'The inputs are [<bos>, Ava, found, a] and targets [Ava, found, a, blue]. The shifted overlap is intentional: each position predicts its successor, and the last prediction needs one extra token beyond the input slice.', ['Use stream positions, not word meanings, to select the next token.', 'Slide the entire target row one place along the stream.']),
      sequenceNotebookLabs.tokenization,
    ],
  },
  {
    id: 'attention', version: 1, displayNumber: 19, published: true, title: 'Attention', topic: 'Content-Based Retrieval',
    description: 'Explore queries, keys, and values; inspect row-wise retrieval; and trace learned attention and registered modules.',
    questions: [
      question('attention-roles', 'Separate Matching from Retrieved Content', 'attentionRetrieve', 'Attention turns query–key similarities into weights and uses those weights to combine values.', 'Scored reference: query [1, 0] and values [2, 8, 5]. Then change only the last value from 5 to 9, leaving the query and all keys fixed.', [
        part('tie', 'Which keys tie for the largest weight in the reference?', '02', [['01', 'Keys 0 and 1'], ['02', 'Keys 0 and 2'], ['12', 'Keys 1 and 2']]),
        part('value', 'What changes when only the last value becomes 9?', 'output', [['both', 'The last key’s weight and the output both increase.'], ['output', 'The weights stay fixed; the output increases.'], ['none', 'Neither the weights nor the output changes.']]),
      ], 'The query gives dot products [1, 0, 1], so keys 0 and 2 tie. Values are not used to compute these similarities. Increasing value 2 by 4 increases the output by four times its positive attention weight, without changing any weight.', ['Compute similarities before looking at values.', 'Which quantities enter the score computation, and which enter the weighted average?']),
      question('attention-rows', 'Read the Attention Matrix', 'attentionMatrix', 'A weight in row i, column j describes how much query i retrieves from key j.', 'Scored reference: inspect query row 1. Values are scalar [2, 8, 5]; all three key positions are allowed.', [
        part('row', 'Which statement correctly describes query row 1?', '12', [['0', 'It gives the greatest weight to key 0.'], ['12', 'It gives equal greatest weight to keys 1 and 2.'], ['column', 'Its entries are normalized down their columns, not across the row.']]),
        part('range', 'What can be guaranteed about its weighted scalar output?', 'range', [['range', 'It lies between the smallest value 2 and the largest value 8.'], ['max', 'It equals the largest value 8 because that key is among the leaders.'], ['outside', 'It exceeds 8 because three values are added together.']]),
      ], 'Query [0, 1] matches keys 1 and 2 equally. Softmax weights are nonnegative and sum to one within the query row, so their value mixture stays in [2, 8]. Attention is generally a weighted retrieval, not a hard choice of one memory.', ['The selected query’s second coordinate distinguishes the keys.', 'A normalized nonnegative weighted average cannot exceed every value it averages.']),
      question('attention-dataflow', 'Trace Parallel Projections and Token Mixing', 'attentionFlow', 'Queries, keys, and values have distinct learned roles even when they have the same shape.', 'Use batch size 2, sequence length 5, and projected channel count 3 from the panel. Each batch is processed separately.', [
        part('scores', 'What is the shape of the score tensor before normalization?', '255', [['253', '(2, 5, 3)'], ['255', '(2, 5, 5)'], ['233', '(2, 3, 3)']]),
        part('output', 'What is the shape after the normalized weights multiply the values?', '253', [['255', '(2, 5, 5)'], ['253', '(2, 5, 3)'], ['23', '(2, 3)']]),
      ], 'Within each batch, (5 × 3) times (3 × 5) gives a (5 × 5) score matrix. Multiplying its weights by (5 × 3) values produces one three-channel output per query, so the output is (2, 5, 3).', ['The score matrix has one axis for each query and another for each key.', 'The weighted mixture keeps the query count and the value channel count.']),
      sequenceNotebookLabs.attention,
    ],
  },
  {
    id: 'causal-attention', version: 1, displayNumber: 20, published: true, title: 'Causal and Multi-Head Attention', topic: 'Autoregressive Information Flow',
    description: 'Explore causal masks, fixed-width multi-head splitting, prefix invariance, and explicit positions before tracing the implementation.',
    questions: [
      question('causal-attention-mask', 'Choose the Allowed Context', 'causalMask', 'When position i predicts token i + 1, the target must not leak into its own input computation.', 'Scored reference: enable the causal mask; indices start at 0. Block future scores before normalizing across keys.', [
        part('allowed', 'At query position 2, which key positions are allowed?', '012', [['01', '0 and 1 only'], ['012', '0, 1, and 2'], ['123', '1, 2, and 3']]),
        part('first', 'What is the attention distribution at query position 0?', 'one', [['uniform', '[0.25, 0.25, 0.25, 0.25]'], ['one', '[1, 0, 0, 0]'], ['zero', '[0, 0, 0, 0]']]),
      ], 'Causality allows the current input token, but excludes all later input tokens. At the first query, only its own key is available, so normalization assigns it probability one. The next-token target sits at the next position and remains hidden.', ['The allowed condition is key position ≤ query position.', 'Normalize over the available keys, not all four original scores.']),
      question('causal-attention-heads', 'Split Channels, Keep the Sequence', 'causalHeads', 'Multiple heads compute several attention patterns at each position while sharing a fixed total channel budget.', 'Scored reference: batch size 3, sequence length 6, total channels 8, and 2 heads. Shapes are ordered exactly as labeled in the panel.', [
        part('size', 'How many channels are assigned to each head?', '4', [['8', '8'], ['4', '4'], ['2', '2']]),
        part('weights', 'What is the shape of the attention-weight tensor?', '3266', [['3264', '(3, 2, 6, 4)'], ['3266', '(3, 2, 6, 6)'], ['3622', '(3, 6, 2, 2)']]),
      ], 'Each head receives 8/2 = 4 channels, but still has six queries and six keys. Therefore the weights are (batch 3, heads 2, queries 6, keys 6). Head channels affect similarity calculation, not the number of sequence positions.', ['Separate the channel axis from the sequence axis.', 'Each head compares every allowed query–key pair.']),
      question('causal-attention-invariance', 'Test What Information Can Reach the Prefix', 'causalPosition', 'A causal attention layer must be insensitive to changes that occur strictly after a query position.', 'Use the deterministic shared-prefix comparison and the separate position-vector comparison described in the panel. Earlier inputs and all layer parameters stay fixed.', [
        part('prefix', 'After changing positions 3–5, what must happen to outputs at positions 0–2 in a correct causal layer?', 'same', [['change', 'They must change because all positions share the learned weights.'], ['same', 'They remain the same because no allowed retrieval reaches the changed suffix.'], ['zero', 'They become zero because the future was masked.']]),
        part('positions', 'Why add distinct position vectors to repeated identical tokens?', 'explicit', [['explicit', 'To supply direct position information even though the causal mask already distinguishes prefixes.'], ['no-order', 'Because a causal mask provides absolutely no order-related information.'], ['ids', 'To force the repeated word to have different vocabulary IDs at every position.']]),
      ], 'Shared parameters do not imply information flow from future inputs. Causal access restrictions preserve all outputs for a shared prefix. Position vectors change the input representation without changing token identity; they add explicit position information beyond the asymmetry of the causal mask.', ['Trace which input positions can influence each prefix output.', 'Distinguish explicit position coordinates from the allowed-history pattern.']),
      sequenceNotebookLabs['causal-attention'],
    ],
  },
]
