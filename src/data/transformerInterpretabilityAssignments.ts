import type { AssignmentSpec, MultipleChoicePart, MultipleChoiceQuestionSpec } from '../types'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import { transformerInterpretabilityLabs } from './transformerInterpretabilityLabs'

export interface TransformerInterpretabilityVisualDataset {
  id: string
  kind: 'transformerInterpretabilityVisual'
  mode: 'residual' | 'normalization' | 'tying' | 'windows' | 'sampling' | 'split' | 'attention' | 'direction' | 'lens' | 'dictionary' | 'sparsity' | 'scale' | 'steering' | 'quality' | 'evaluation'
  title: string
  caption: string
}

const visual = (mode: TransformerInterpretabilityVisualDataset['mode'], title: string, caption: string): TransformerInterpretabilityVisualDataset => ({ id: `ti-${mode}`, kind: 'transformerInterpretabilityVisual', mode, title, caption })
export const transformerInterpretabilityVisualDatasets: Record<string, TransformerInterpretabilityVisualDataset> = Object.fromEntries([
  visual('residual', 'Follow the residual additions', 'A residual stream retains its input and adds each sublayer’s update. Arrows show actual vector sums in a two-channel toy example.'),
  visual('normalization', 'Normalize one token at a time', 'Change a later token and inspect both normalized vectors. Layer normalization uses channels within one token, not statistics across sequence positions.'),
  visual('tying', 'One matrix, two uses', 'A five-token language model has three channels. The output projection reuses the token-embedding matrix; no output bias is present. Count only this shared matrix.'),
  visual('windows', 'Align inputs with next-token targets', 'The letters denote token identities, not numerical feature values. A context window contributes one prediction per input position.'),
  visual('sampling', 'Separate support from concentration', 'Top-k restricts eligible tokens. Temperature changes their relative probabilities. These fixed logits have no ties.'),
  visual('split', 'What exactly is held out?', 'A, B, and C are distinct complete stories. Vocabulary construction uses only the training portion under each policy.'),
  visual('attention', 'Read a query-by-key pattern', 'A causal attention head updates each query using scalar values from permitted keys. Read one row, not one column, to follow an update.'),
  visual('direction', 'From labeled clouds to a direction', 'Compare two context labels in activation space, then inspect projections onto a unit direction. All four observations remain visible.'),
  visual('lens', 'Intermediate decoding is a diagnostic', 'These supplied toy logits use the same final normalization and vocabulary head at three residual-stream stages. There is no intervention after the final block.'),
  visual('dictionary', 'Several directions, few active features', 'An overcomplete dictionary represents a two-channel activation with three candidate feature directions. Coefficients are nonnegative.'),
  visual('sparsity', 'Compare reconstruction with sparsity', 'Penalize the sum of absolute feature coefficients per example, alongside mean squared reconstruction error. Both candidate codes use the same fixed dictionary.'),
  visual('scale', 'A loophole in an unconstrained decoder', 'Multiplying a decoder direction and inversely scaling its coefficient can leave reconstruction unchanged. Inspect the two equivalent contributions.'),
  visual('steering', 'Remove or amplify a feature contribution', 'Intervene on an activation vector, not directly on a vocabulary probability. The toy residual may contain a remainder that the selected feature does not explain.'),
  visual('quality', 'A target effect has a cost', 'Compare supplied held-out measurements across interventions. Higher target probability is desirable here, but predictive quality must also satisfy a fixed budget.'),
  visual('evaluation', 'Selection and evaluation have different jobs', 'A feature’s high average activation before periods motivates a hypothesis. Controlled edits on independent data test that hypothesis.'),
].map(d => [d.id, d]))

const part = (id: string, prompt: string, correctOptionId: string, options: [string, string][]): MultipleChoicePart => ({ id, prompt, correctOptionId, options: options.map(([id, title]) => ({ id, title, description: '' })) })
function warmup(id: string, title: string, prompt: string, mode: TransformerInterpretabilityVisualDataset['mode'], parts: MultipleChoicePart[], explanation: string, hints: string[]): MultipleChoiceQuestionSpec {
  return { id, title, prompt, kind: 'multipleChoice', datasetId: `ti-${mode}`, parts, instructions: 'Exploration controls are not graded. Answer using the fixed reference conditions stated in the panel and each question.', hints, hintSchedule: [2, 4], reveal: { explanation }, validator: validateMultipleChoiceSelections(Object.fromEntries(parts.map(p => [p.id, p.correctOptionId]))) }
}

export const transformerInterpretabilityAssignments: AssignmentSpec[] = [
  {
    id: 'pico-gpt', displayNumber: 21, version: 1, published: true, title: 'Building PicoGPT', topic: 'Decoder-Only Transformers',
    description: 'Trace residual additions, token-local normalization, and shared vocabulary weights before inspecting the complete next-token model.',
    questions: [
      warmup('pico-residual', 'What Does a Residual Connection Preserve?', 'A pre-normalized transformer block adds attention’s update to the input, then adds an MLP update to the resulting stream. The second sublayer reads the updated stream.', 'residual', [
        part('sum', 'At both update scales 1, what vector leaves the block?', 'sum', [['replace', '(0.5, −0.5)'], ['sum', '(1.5, 2.5)'], ['input', '(2, 1)']]),
        part('zero', 'If both sublayers produce zero updates, what leaves the block?', 'identity', [['zero', 'The zero vector.'], ['normalize', 'Only the normalized input.'], ['identity', 'The original input vector.']]),
      ], 'Residual additions accumulate updates: (2, 1) + (−1, 2) + (0.5, −0.5) = (1.5, 2.5). Zero sublayer outputs leave the direct input path intact; normalization happens inside the branches, not in place of the residual.', ['Follow the attention addition before the MLP addition.', 'Adding a zero update differs from replacing the stream with a zero output.']),
      warmup('pico-normalization', 'Can a Later Token Change an Earlier Normalization?', 'Causal attention blocks future information. Other operations must also avoid mixing future tokens into an earlier position.', 'normalization', [
        part('local', 'When token 1’s first channel changes from 10 to 16, what happens to normalized token 0?', 'same', [['changes', 'It changes because the sequence mean changes.'], ['same', 'It stays unchanged.'], ['zero', 'It becomes the zero vector.']]),
        part('leak', 'Which replacement could break causality even if attention remains masked?', 'sequence', [['channels', 'Normalizing channels separately at each token.'], ['sequence', 'Normalizing each channel using the entire sequence’s mean.'], ['mlp', 'Applying the same MLP separately to every token.']]),
      ], 'Token-local normalization never consults another position. Normalizing across the entire sequence would let a future token alter an earlier result, bypassing the attention mask. Sharing MLP weights across positions does not itself mix their inputs.', ['Identify which values contribute to token 0’s mean.', 'A causal guarantee must hold through every operation, not only attention.']),
      warmup('pico-weight-tying', 'Shared Geometry for Reading and Predicting', 'Weight tying makes token lookup and vocabulary projection use the same learned matrix. It does not make output logits identical to input token IDs.', 'tying', [
        part('parameters', 'How many distinct scalar weights belong to the shared matrix?', '15', [['30', '30'], ['15', '15'], ['8', '8']]),
        part('logits', 'For two sequences of length 4, how many next-token logits are produced?', '40', [['8', '8: one scalar per input token.'], ['24', '24: one scalar per channel.'], ['40', '40: five vocabulary scores at each of eight positions.']]),
      ], 'The shared 5 × 3 matrix has 15 parameters, counted once despite its two uses. Every one of the 2 × 4 positions still receives 5 logits: sharing parameters does not remove the vocabulary dimension.', ['Count entries, not module references.', 'The output needs a score for every possible next token at every position.']),
      transformerInterpretabilityLabs.pico,
    ],
  },
  {
    id: 'training-pico-gpt', displayNumber: 22, version: 1, published: true, title: 'Training PicoGPT', topic: 'Language-Model Training and Sampling',
    description: 'Align next-token targets, vary sampling distributions, and reason about held-out stories before tracing the training pipeline.',
    questions: [
      warmup('training-windows', 'One Window, Several Predictions', 'A language model learns to predict the next token at every context position, not only after the last token in a window.', 'windows', [
        part('target', 'At start position 2, which target sequence aligns with inputs C, D, E?', 'shift', [['same', 'C, D, E'], ['shift', 'D, E, F'], ['last', 'F, F, F']]),
        part('positions', 'How many supervised next-token predictions does a batch of four length-3 windows contribute?', '12', [['4', '4'], ['3', '3'], ['12', '12']]),
      ], 'Targets are shifted by one position, so each input token has its own next-token label. Four windows with three positions each supply twelve loss terms before the batch mean is taken.', ['Read the arrow under each highlighted input token.', 'Count batch positions, not just windows.']),
      warmup('training-sampling', 'Temperature Is Not Training', 'Compare sampling choices while holding the model’s logits fixed. A sampling distribution describes possible draws; it does not promise one particular generated token.', 'sampling', [
        part('temperature', 'With k = 2, lowering temperature from 1 to 0.5 has which effect?', 'sharp', [['switch', 'Changes the highest-probability token to B.'], ['sharp', 'Keeps only A and B eligible, while increasing A’s probability.'], ['new', 'Makes C and D eligible again.']]),
        part('deterministic', 'With these distinct logits and k = 1, what does positive-temperature sampling do?', 'A', [['A', 'Always selects A.'], ['uniform', 'Selects all four tokens equally often.'], ['learn', 'Updates model weights toward A.']]),
      ], 'Positive temperature preserves ranking; lower temperature sharpens the retained distribution. Top-k controls support separately. With k = 1 and no tie only A remains, so sampling is deterministic without any weight update.', ['Compare which bars are exactly zero with how the nonzero bars change.', 'Only eligible tokens can be sampled.']),
      warmup('training-split', 'What Would a Validation Result Establish?', 'A training pipeline must decide what it holds out before tokenization and window sampling. A tiny structured corpus is useful for checking machinery, but it is a limited test of language ability.', 'split', [
        part('policy', 'Which policy avoids putting parts of story B on both sides of the split?', 'Y', [['X', 'Policy X.'], ['Y', 'Policy Y.'], ['both', 'Both policies.']]),
        part('claim', 'A model’s loss falls on held-out stories built from the same few templates. What is supported?', 'limited', [['english', 'The model understands unrestricted English.'], ['none', 'No result is meaningful unless generated prose is perfect.'], ['limited', 'The pipeline learns patterns that transfer within this structured corpus.']]),
      ], 'Splitting complete stories avoids adjacent fragments crossing the split. Even disjoint stories may share strong templates, so held-out success supports a narrower claim than broad language understanding.', ['Track story identity rather than token count.', 'Ask how different the held-out distribution is from training.']),
      transformerInterpretabilityLabs.training,
    ],
  },
  {
    id: 'transformer-activations', displayNumber: 23, version: 1, published: true, title: 'Looking Inside PicoGPT', topic: 'Transformer Activation Analysis',
    description: 'Read attention patterns, construct label-associated directions, and distinguish intermediate diagnostics from causal claims.',
    questions: [
      warmup('activations-attention', 'Which Direction Does Attention Read?', 'An attention row describes where one query reads information. Its entries weight value vectors, not the token IDs themselves.', 'attention', [
        part('weighted', 'For query 2, what is the scalar weighted value?', '2.1', [['1.5', '1.50'], ['2.1', '2.10'], ['4', '4.00']]),
        part('future', 'If only the value at key 3 changes, what happens to the query-2 weighted value with this fixed pattern?', 'same', [['change', 'It must change because key 3 is in the sequence.'], ['same', 'It stays unchanged because its weight is zero.'], ['renorm', 'All query-2 weights become uniform.']]),
      ], 'Query 2 uses row 2: 0.2 × 2 + 0.3 × (−1) + 0.5 × 4 = 2.1. Its masked future key has zero weight. A large attention weight is a routing quantity, not by itself proof of a semantic explanation.', ['Use the entire selected row and the signed values.', 'A zero coefficient removes that key’s direct contribution.']),
      warmup('activations-direction', 'A Direction Is Not a Single Neuron', 'A contrastive direction is the positive-label mean minus the other-label mean, normalized to length 1. A context’s score is its dot product with that full direction.', 'direction', [
        part('direction', 'For these four points, which normalized contrastive direction should be used?', 'x', [['y', '(0, 1)'], ['x', '(1, 0)'], ['diag', '(0.707, 0.707)']]),
        part('causal', 'At angle 0°, P2 scores highest. What does this establish?', 'association', [['causal', 'Adding this direction will always cause the positive label.'], ['single', 'Channel 1 has exactly one semantic meaning.'], ['association', 'This observed activation is strongly aligned with a label-associated direction.']]),
      ], 'The means are (3, 2) and (1, 2), giving normalized difference (1, 0). Projection is an association in observed activations. It does not prove a causal effect of changing that vector or a unique interpretation of one coordinate.', ['Average each coordinate within each label group.', 'Inspection observes a model; an intervention changes it.']),
      warmup('activations-lens', 'Read an Intermediate Representation Cautiously', 'A logit lens applies the final normalization and output head to an earlier residual stream. Earlier representations were not necessarily trained to be decoded by this shortcut.', 'lens', [
        part('early', 'The block-0 lens favors A, but the final lens favors C. What follows?', 'refine', [['bug', 'The transformer must be implemented incorrectly.'], ['refine', 'Later computation changes which token the final head favors.'], ['A', 'The model’s actual final prediction is A.']]),
        part('final', 'With no post-block edit, how does the final-block lens compare with the actual final logits?', 'same', [['same', 'It gives the same logits because it applies the same remaining operations.'], ['earlier', 'It must reproduce the embedding-stage logits.'], ['different', 'It must differ because all logit-lens outputs are approximate.']]),
      ], 'Intermediate lenses are diagnostics rather than substitute model outputs. After the final block, however, applying the actual final normalization and head is precisely the remaining forward pass, so the logits match.', ['Compare the remaining operations after each stage.', 'The final stage is a special case of the diagnostic.']),
      transformerInterpretabilityLabs.activations,
    ],
  },
  {
    id: 'sparse-autoencoders', displayNumber: 24, version: 1, published: true, title: 'Sparse Autoencoders', topic: 'Sparse Activation Dictionaries',
    description: 'Reconstruct activations from sparse feature directions and examine reconstruction–sparsity tradeoffs and decoder scaling.',
    questions: [
      warmup('sae-dictionary', 'Overcomplete Does Not Mean All Features Are Active', 'An activation autoencoder can offer more candidate features than input coordinates. Sparsity asks it to use few features on each example.', 'dictionary', [
        part('code', 'At coefficients (0, 0, 2), how many features are active and is reconstruction exact?', 'one', [['three', 'Three active features; exact reconstruction.'], ['one', 'One active feature; exact reconstruction.'], ['none', 'No active features; zero reconstruction.']]),
        part('compare', 'Compared with code (1.2, 1.6, 0), the reference code (0, 0, 2) has what properties?', 'sparse', [['worse', 'Greater MSE but equal L1.'], ['same', 'Equal MSE, equal L1, and equal active-feature count.'], ['sparse', 'Equal MSE, smaller L1, and fewer active features.']]),
      ], 'Two copies of direction (0.6, 0.8) reconstruct (1.2, 1.6). Both codes reconstruct exactly, but their L1 costs are 2 versus 2.8 and their active counts are 1 versus 2. Overcompleteness offers candidate directions without requiring dense use.', ['A zero coefficient contributes no direction.', 'Compare sums of coefficients separately from reconstruction error.']),
      warmup('sae-tradeoff', 'A Sparse Code Can Accept Some Error', 'The objective is MSE + λ × L1, where L1 sums absolute feature activations for one example. Smaller total objective is better under this specified rule.', 'sparsity', [
        part('choice', 'At λ = 0.2, which fixed candidate has the smaller total objective?', 'shrunk', [['exact', 'Exact.'], ['shrunk', 'Sparse/shrunk.'], ['tie', 'They tie.']]),
        part('zero', 'At λ = 0, which criterion determines the preferred candidate?', 'mse', [['count', 'Only the active-feature count.'], ['mse', 'Only reconstruction MSE.'], ['width', 'Only the number of available decoder directions.']]),
      ], 'The exact code costs 0 + 0.2 × 2.8 = 0.56. The sparse/shrunk code costs 0.125 + 0.2 × 1.5 = 0.425. At zero penalty strength only MSE matters. Lower feature activity is not sufficient by itself to establish a useful SAE.', ['Multiply the displayed L1 by the fixed penalty strength.', 'Keep reconstruction error and sparsity cost in separate columns before adding.']),
      warmup('sae-decoder-norm', 'Why Constrain Decoder Directions?', 'A sparsity penalty is vulnerable to a change of scale if the decoder can grow without limit. The table isolates one feature contribution.', 'scale', [
        part('loophole', 'Moving from the original to the rescaled row does what?', 'cheat', [['double', 'Doubles reconstruction and doubles L1.'], ['cheat', 'Preserves reconstruction while halving this coefficient’s L1 cost.'], ['same', 'Preserves both reconstruction and L1 cost.']]),
        part('constraint', 'Which condition blocks this particular rescaling loophole?', 'unit', [['wide', 'Making the feature dictionary wider.'], ['nonnegative', 'Requiring nonnegative coefficients only.'], ['unit', 'Keeping every decoder direction at unit norm.']]),
      ], 'The product of coefficient and direction is unchanged, but L1 depends on the coefficient alone. The rescaled direction has norm 2, violating a unit-norm constraint. Nonnegativity and wider dictionaries do not prevent this scaling trick.', ['Compare the products and coefficient magnitudes separately.', 'Compute the length of each decoder direction.']),
      transformerInterpretabilityLabs.sae,
    ],
  },
  {
    id: 'feature-steering', displayNumber: 25, version: 1, published: true, title: 'Feature Steering', topic: 'Causal Interventions and Controls',
    description: 'Ablate and amplify feature contributions, compare target effects with quality costs, and separate feature selection from causal evaluation.',
    questions: [
      warmup('steering-contribution', 'Ablation Removes a Contribution, Not the Whole Stream', 'Suppose h = zd + r, where z is one selected feature coefficient, d its unit decoder direction, and r the remainder. Change zd while preserving r.', 'steering', [
        part('ablation', 'At factor 0, what is the edited residual?', 'remainder', [['zero', '(0, 0)'], ['remainder', '(1.8, 0.4)'], ['negative', '(−1.2, −1.6)']]),
        part('control', 'To match the size of the factor-2 edit with a random-direction control, what should be added?', 'matched', [['unit', 'One unit random direction, regardless of z.'], ['matched', 'z times a unit random direction.'], ['replace', 'A completely random replacement residual.']]),
      ], 'Ablation subtracts 2(0.6, 0.8) from (3, 2), leaving (1.8, 0.4). Amplification by factor 2 adds a vector of norm z = 2, so a magnitude-matched random control should also add z times a unit direction.', ['Separate the selected contribution from the residual remainder.', 'A unit direction sets orientation; its coefficient sets edit magnitude.']),
      warmup('steering-quality', 'Which Intervention Meets the Quality Budget?', 'An intervention that pushes the target statistic most strongly may also damage general prediction. Compare both effects under one explicit decision rule.', 'quality', [
        part('budget', 'Maximize mean P(period) subject to a cross-entropy increase no greater than 0.10. Which row wins?', 'moderate', [['strong', 'Strong.'], ['random', 'Matched random.'], ['moderate', 'Moderate.'], ['baseline', 'Baseline.']]),
        part('comparison', 'Why evaluate the conditions on the same held-out examples?', 'paired', [['identical', 'It forces their predictions to be identical.'], ['paired', 'It prevents a change in evaluation examples from explaining the measured difference.'], ['train', 'It lets the model train directly on the final-test examples.']]),
      ], 'Moderate steering increases P(period) to 0.45 at a cost of 0.03; strong steering violates the 0.10 budget with a cost of 0.90. Paired held-out examples isolate the comparison from changes in evaluation data, without guaranteeing identical outputs or proving universal semantics.', ['Subtract baseline cross-entropy before testing the constraint.', 'Apply the quality constraint before comparing the target statistic.']),
      warmup('steering-heldout', 'A Feature Hypothesis Needs an Independent Test', 'Use calibration data to choose a promising feature. Then evaluate interventions, matched controls, and quality costs on an untouched final-test split.', 'evaluation', [
        part('selection', 'Which procedure preserves the intended independent final test?', 'calibration', [['testpick', 'Try every feature on final-test data and report only the best one.'], ['calibration', 'Select on calibration data, then evaluate the chosen feature once on final-test data.'], ['sample', 'Pick the prettiest generated sample and skip aggregate measurements.']]),
        part('logit', 'Directly lowering the period output logit lowers its probability. What does that result establish about an internal SAE feature?', 'none', [['meaning', 'The internal feature has a unique period meaning.'], ['none', 'Nothing by itself; it is an output-level control, not evidence about the internal feature.'], ['equivalent', 'The output edit is identical to internal feature ablation in every context.']]),
      ], 'Choosing the winner on final-test data turns that set into another selection set. Direct logit suppression demonstrates an output-level mechanism but cannot establish what an internal feature represents or whether its edit has broader structured effects.', ['Separate choosing a hypothesis from testing it.', 'Compare where in the model each intervention is applied.']),
      transformerInterpretabilityLabs.steering,
    ],
  },
]
