import type { AssignmentSpec, MultipleChoiceQuestionSpec } from '../types'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import { neuralFoundationsNotebookLabs } from './neuralFoundationsCodeLabs'

export interface NeuralFoundationsVisualDataset {
  id: string
  kind: 'neuralFoundationsVisual'
  mode: 'hinge' | 'branches' | 'shapes' | 'graph' | 'chain' | 'pairedLoss' | 'logits' | 'moments' | 'batches' | 'dropout' | 'correlation' | 'pooling'
  caption: string
}
const dataset = (id: string, mode: NeuralFoundationsVisualDataset['mode'], caption: string): NeuralFoundationsVisualDataset => ({ id, kind: 'neuralFoundationsVisual', mode, caption })
export const neuralFoundationsVisualDatasets: Record<string, NeuralFoundationsVisualDataset> = {
  neuralHinge: dataset('neuralHinge', 'hinge', 'A rectified linear unit (ReLU) outputs max(0, z). Here its input is z = x − threshold. Both axes measure scalar values, not probabilities.'),
  neuralBranches: dataset('neuralBranches', 'branches', 'Two hidden units receive x and −x. Their outputs are added with weights 1 and 1. Toggle whether each hidden unit applies ReLU.'),
  neuralShapes: dataset('neuralShapes', 'shapes', 'Each row is one observation. A dense network applies the same weights and biases to every row; hidden and output widths describe features, not observations.'),
  neuralGraph: dataset('neuralGraph', 'graph', 'The scalar function f = ab + a uses a along two paths. Arrows show the computation; derivative labels describe how a small change propagates backward.'),
  neuralChain: dataset('neuralChain', 'chain', 'For u > 0, h = ln(√u). The graph shows the function and a short tangent segment at the movable probe. Horizontal axis: u. Vertical axis: h.'),
  neuralPairedLoss: dataset('neuralPairedLoss', 'pairedLoss', 'Two observations have predictions (1, 3) and targets (2, 0). Each matrix entry is (row prediction − column target)². Diagonal entries compare matching observations.'),
  neuralLogits: dataset('neuralLogits', 'logits', 'Three raw class scores are (2, −1, 0), in class order A, B, C. Scores can be negative and need not sum to 1. Probability bars apply softmax across the three classes.'),
  neuralMoments: dataset('neuralMoments', 'moments', 'A simplified first-moment accumulator starts at m₀ = 0 and follows mₜ = βmₜ₋₁ + (1 − β)gₜ. Gradients are fixed at (2, −2, 2, −2). This is an ingredient of Adam, not its complete or bias-corrected update.'),
  neuralBatches: dataset('neuralBatches', 'batches', 'Twelve labeled observations are partitioned into batches after choosing an order. Every row shows one batch. Changing order does not duplicate or remove observations.'),
  neuralDropout: dataset('neuralDropout', 'dropout', 'Activations are (1, 2, 3, 4). For the displayed training example, dropout probability is 0.25 and the second activation is dropped. Retained values are divided by 0.75. This is one fixed mask, not an average over masks.'),
  neuralCorrelation: dataset('neuralCorrelation', 'correlation', 'Slide one 2 × 2 contrast filter over a 4 × 4 image, one cell at a time, without padding. At each location multiply corresponding entries and sum. The filter stays in the shown orientation.'),
  neuralPooling: dataset('neuralPooling', 'pooling', 'Replace each complete, non-overlapping 2 × 2 block by its maximum. Partial blocks at the right and bottom are discarded. Labels and borders mark the retained blocks.'),
}

const option = (id: string, title: string) => ({ id, title, description: '' })
function question(input: Omit<MultipleChoiceQuestionSpec, 'kind' | 'validator' | 'hintSchedule'>): MultipleChoiceQuestionSpec {
  return { ...input, kind: 'multipleChoice', hintSchedule: [2, 4], validator: validateMultipleChoiceSelections(Object.fromEntries(input.parts.map(p => [p.id, p.correctOptionId])), 'Use the stated reference conditions and compare the displayed quantities.') }
}

const mlps: AssignmentSpec = {
  id: 'mlps', version: 1, displayNumber: 11, published: true, title: 'Multilayer Perceptrons', topic: 'Neural Networks',
  description: 'Explore rectified features, nonlinear combinations, and shared parameters, then trace a small network forward.',
  questions: [
    question({ id: 'mlps-rectified-feature', title: 'Move a Feature’s Activation Threshold', datasetId: 'neuralHinge',
      prompt: 'A hidden unit can ignore one region of the input while responding linearly in another.',
      instructions: 'Explore the threshold slider. Answer at the fixed reference threshold 1, regardless of the current slider setting.',
      parts: [{ id: 'outputs', prompt: 'At threshold 1, what are the outputs for x = −1 and x = 2, in that order?', correctOptionId: 'zero-one', options: [option('negative-one', '(−2, 1)'), option('zero-one', '(0, 1)'), option('one-two', '(1, 2)')] },
        { id: 'inactive', prompt: 'At threshold 1, moving x from −2 to −1 has what effect on this unit’s output?', correctOptionId: 'same', options: [option('increase', 'It increases by 1.'), option('decrease', 'It decreases by 1.'), option('same', 'It stays at 0.')] }],
      hints: ['Compute x − threshold before applying ReLU.', 'ReLU keeps positive values but replaces negative values with zero.'],
      reveal: { explanation: 'At threshold 1 the inputs −1 and 2 produce preactivations −2 and 1, then outputs 0 and 1. Both −2 and −1 lie in the flat, inactive region.' },
    }),
    question({ id: 'mlps-nonlinear-composition', title: 'Why Put Activations Between Layers?', datasetId: 'neuralBranches',
      prompt: 'Adding hidden layers does not by itself guarantee a nonlinear input–output relationship.',
      instructions: 'Compare the same two branches with ReLU on and off. The scored comparison uses x = −2 and x = 2 and fixed branch weights 1 and 1.',
      parts: [{ id: 'active', prompt: 'With ReLU enabled, what outputs does the network give at x = −2 and x = 2?', correctOptionId: 'two-two', options: [option('negative-positive', '(−2, 2)'), option('zero-zero', '(0, 0)'), option('two-two', '(2, 2)')] },
        { id: 'depth', prompt: 'For any number of dense affine layers with no nonlinear activations, what is the overall transformation?', correctOptionId: 'affine', options: [option('affine', 'Still an affine transformation of the input.'), option('curved', 'Always curved once there are two layers.'), option('relu', 'Automatically a rectified transformation.')] }],
      hints: ['Only one of x and −x is positive when x is nonzero.', 'Substitute one affine formula into another and collect coefficients and constants.'],
      reveal: { explanation: 'The activated network computes max(0, x) + max(0, −x) = |x|. Without activations these particular branches sum to zero. More generally, composing affine layers remains affine, so a nonlinear activation is what allows a kink.' },
    }),
    question({ id: 'mlps-shared-parameters', title: 'More Observations, Same Network', datasetId: 'neuralShapes',
      prompt: 'A dense layer connects every input feature to every output feature and gives each output feature one bias.',
      instructions: 'The reference network has 2 input features, 2 hidden units, and 1 output, with biases in both dense layers. Explore batch size; parameter values are shared across all observations.',
      parts: [{ id: 'parameters', prompt: 'How many trainable scalar parameters does this 2 → 2 → 1 network have?', correctOptionId: 'nine', options: [option('six', '6'), option('nine', '9'), option('seven', '7')] },
        { id: 'batch', prompt: 'If the batch grows from 3 observations to 6, what happens to the number of model parameters?', correctOptionId: 'fixed', options: [option('double', 'It doubles.'), option('six-more', 'It increases by 6.'), option('fixed', 'It stays the same.')] }],
      hints: ['Count input-to-hidden weights and hidden biases, then hidden-to-output weights and the output bias.', 'A batch reuses one model; it does not create a separate model for every row.'],
      reveal: { explanation: 'There are 2×2 + 2 = 6 parameters in the first layer and 2×1 + 1 = 3 in the second: 9 total. More rows increase the amount of computation, not the number of learned parameters.' },
    }), neuralFoundationsNotebookLabs.mlps,
  ],
}

const autograd: AssignmentSpec = {
  id: 'autograd', version: 1, displayNumber: 12, published: true, title: 'Automatic Differentiation', topic: 'Computation Graphs',
  description: 'Follow gradients through branches and compositions, distinguish fresh derivatives from accumulated state, and trace scalar backpropagation.',
  questions: [
    question({ id: 'autograd-two-paths', title: 'Add Every Path’s Contribution', datasetId: 'neuralGraph',
      prompt: 'When a variable influences a result through more than one path, its derivative includes every path’s contribution.',
      instructions: 'Explore a and b. Answer for the fixed reference a = 2, b = 3 and f = ab + a.',
      parts: [{ id: 'da', prompt: 'What is ∂f/∂a at the reference point?', correctOptionId: 'four', options: [option('three', '3'), option('four', '4'), option('six', '6')] },
        { id: 'db', prompt: 'If only b increases by a small amount δ, what is the change in f?', correctOptionId: 'twodelta', options: [option('fourdelta', '4δ'), option('delta', 'δ'), option('twodelta', '2δ')] }],
      hints: ['The multiplication path contributes b to ∂f/∂a; the direct addition path contributes 1.', 'The variable b appears only in the product ab.'],
      reveal: { explanation: 'The derivative with respect to a is b + 1 = 4 because both paths contribute. With a fixed at 2, f is linear in b with slope 2.' },
    }),
    question({ id: 'autograd-chain-rule', title: 'Multiply Along a Chain', datasetId: 'neuralChain',
      prompt: 'For h = ln(√u), the square root changes u into an intermediate value before the logarithm acts. Use d√u/du = 1/(2√u) and d ln(v)/dv = 1/v.',
      instructions: 'Move the probe to compare slopes. Answer at the fixed reference u = 4; all logarithms are natural.',
      parts: [{ id: 'slope', prompt: 'What is dh/du at u = 4?', correctOptionId: 'eighth', options: [option('half', '1/2'), option('quarter', '1/4'), option('eighth', '1/8')] },
        { id: 'trend', prompt: 'As u increases over positive values, what happens to dh/du?', correctOptionId: 'smaller', options: [option('smaller', 'It remains positive and becomes smaller.'), option('larger', 'It remains positive and becomes larger.'), option('negative', 'It becomes negative as soon as u exceeds 1.')] }],
      hints: ['At u = 4 the intermediate value √u is 2.', 'Multiply local derivatives: [1/√u] × [1/(2√u)].'],
      reveal: { explanation: 'The chain rule gives dh/du = 1/(2u), so the slope is 1/8 at u = 4. A positive but decreasing slope means the function keeps increasing, more slowly.' },
    }),
    question({ id: 'autograd-reset-state', title: 'A Derivative and Its Storage Are Different', datasetId: 'neuralGraph',
      prompt: 'Suppose a gradient accumulator adds the result of every backward pass to its stored total. A fresh graph is built for each pass and the parameter values do not change.',
      instructions: 'Use f = ab + a at fixed a = 2, b = 3. The stored gradient with respect to a begins at zero. This question uses two fresh graphs, not a reused graph with uncleared intermediate gradients.',
      parts: [{ id: 'stored', prompt: 'After two passes without clearing the accumulator, what is the stored gradient with respect to a?', correctOptionId: 'eight', options: [option('four', '4'), option('eight', '8'), option('sixteen', '16')] },
        { id: 'meaning', prompt: 'What does that accumulated value tell us about the mathematical derivative at the unchanged point?', correctOptionId: 'still-four', options: [option('steeper', 'The function has become twice as steep.'), option('minimum', 'The parameter has reached a minimum.'), option('still-four', 'The derivative is still 4; its contribution was stored twice.')] }],
      hints: ['Each fresh pass computes the same derivative at the same point.', 'Accumulation changes stored state, not the function or the parameter values.'],
      reveal: { explanation: 'Each fresh graph contributes 4, so the stored total is 8 while the derivative remains 4. Clear accumulators when the intended update should use only the current pass.' },
    }), neuralFoundationsNotebookLabs.autograd,
  ],
}

const pytorch: AssignmentSpec = {
  id: 'pytorch', version: 1, displayNumber: 13, published: true, title: 'PyTorch Networks', topic: 'Tensor Shapes and Losses',
  description: 'Reason about batch and feature dimensions, matched targets, and class scores before inspecting deterministic PyTorch expressions.',
  questions: [
    question({ id: 'pytorch-observations-features', title: 'Keep Observations Separate from Features', datasetId: 'neuralShapes',
      prompt: 'A batch is a matrix: rows are observations, columns are features. Each dense layer transforms features independently for every row.',
      instructions: 'The reference network in this panel is 2 → 2 → 1. Answer for a batch of 5 observations; exploratory batch-size changes do not change the scored reference.',
      parts: [{ id: 'shape', prompt: 'What is the output matrix shape for the reference batch?', correctOptionId: 'five-one', options: [option('one-five', '(1, 5)'), option('five-one', '(5, 1)'), option('five-two', '(5, 2)')] },
        { id: 'mixing', prompt: 'What does a dense layer mix when computing one output row?', correctOptionId: 'features', options: [option('rows', 'Different observations in the batch.'), option('both', 'Every observation and every feature in the batch.'), option('features', 'Features of that same observation.')] }],
      hints: ['The network produces one prediction per observation.', 'Shared parameters do not mean that input observations are added together.'],
      reveal: { explanation: 'Five observations produce five rows, each with one output. A dense layer mixes feature columns within each row; the same transformation is applied to all five rows.' },
    }),
    question({ id: 'pytorch-paired-targets', title: 'Compare Each Prediction with Its Own Target', datasetId: 'neuralPairedLoss',
      prompt: 'A regression loss should pair each observation’s prediction with that observation’s target. Comparing all possible pairs is a different objective.',
      instructions: 'Use the fixed predictions (1, 3) and targets (2, 0). Toggle the highlighted comparisons. The scored answers always refer to these fixed values.',
      parts: [{ id: 'paired', prompt: 'What is the mean squared error using only the two matching pairs?', correctOptionId: 'five', options: [option('three', '3'), option('ten', '10'), option('five', '5')] },
        { id: 'all', prompt: 'What is the mean squared error if all four prediction–target pairs are averaged instead?', correctOptionId: 'three', options: [option('three', '3'), option('five', '5'), option('six', '6')] }],
      hints: ['The matching errors are 1 − 2 and 3 − 0.', 'For all pairs, include every cell in the matrix, not just its diagonal.'],
      reveal: { explanation: 'Matching observations gives (1 + 9)/2 = 5. The all-pairs calculation averages 1, 1, 1, and 9, giving 3. A valid-looking scalar loss can still measure the wrong comparisons.' },
    }),
    question({ id: 'pytorch-class-scores', title: 'Class Scores Are Not Probabilities', datasetId: 'neuralLogits',
      prompt: 'A classifier emits one raw score per class. The greatest score chooses the predicted class; softmax converts scores into a normalized probability distribution.',
      instructions: 'Explore a positive score multiplier. The fixed reference scores are (2, −1, 0) for classes A, B, C, at multiplier 1.',
      parts: [{ id: 'prediction', prompt: 'Which class has the greatest probability after softmax at the reference scores?', correctOptionId: 'a', options: [option('b', 'B'), option('a', 'A'), option('c', 'C')] },
        { id: 'loss', prompt: 'With scores (2, −1, 0), what shows that the raw scores cannot themselves be class probabilities?', correctOptionId: 'negative', options: [option('three', 'There are three values.'), option('winner', 'One score is larger than the others.'), option('negative', 'One score is negative and another is greater than 1.')] }],
      hints: ['Softmax preserves the ranking of scores.', 'Every probability must lie between 0 and 1; a mutually exclusive class distribution also sums to 1.'],
      reveal: { explanation: 'A has the greatest raw score and therefore the greatest softmax probability. Raw scores are unconstrained; negative scores are allowed and do not mean negative probabilities.' },
    }), neuralFoundationsNotebookLabs.pytorch,
  ],
}

const optimizers: AssignmentSpec = {
  id: 'optimizers', version: 1, displayNumber: 14, published: true, title: 'Optimizers and Training Modes', topic: 'Training Neural Networks',
  description: 'Explore a running gradient average, mini-batch coverage, and dropout modes, then trace the digits training pipeline.',
  questions: [
    question({ id: 'optimizers-memory', title: 'Give a Gradient Estimate Some Memory', datasetId: 'neuralMoments',
      prompt: 'Adam maintains running summaries of gradients. This experiment isolates its first-moment idea: blend the previous summary with the current gradient.',
      instructions: 'Explore β. For the scored reference use β = 0.5, m₀ = 0, and first two gradients 2 then −2. Do not apply bias correction or treat m as Adam’s full parameter step.',
      parts: [{ id: 'second', prompt: 'What is the running summary m₂ after those first two gradients?', correctOptionId: 'minus-half', options: [option('zero', '0'), option('minus-half', '−0.5'), option('minus-two', '−2')] },
        { id: 'memory', prompt: 'In this recurrence, setting β = 0 has what effect?', correctOptionId: 'current', options: [option('freeze', 'The summary stays at zero forever.'), option('all', 'All earlier gradients receive equal weight.'), option('current', 'The summary equals only the current gradient.')] }],
      hints: ['Compute m₁ before using it in the formula for m₂.', 'At β = 0, the coefficient multiplying the previous summary is zero.'],
      reveal: { explanation: 'The reference gives m₁ = 1 and m₂ = 0.5×1 + 0.5×(−2) = −0.5. At β = 0 the formula reduces to mₜ = gₜ. Actual Adam also tracks squared gradients and applies bias correction.' },
    }),
    question({ id: 'optimizers-batch-coverage', title: 'Finish the Last Batch', datasetId: 'neuralBatches',
      prompt: 'An epoch partitions one ordering of the training observations into consecutive mini-batches. A final short batch is still used.',
      instructions: 'Explore batch size and ordering. Answer for 12 observations and fixed batch size 5; ordering changes only which observations appear together.',
      parts: [{ id: 'sizes', prompt: 'What are the batch sizes in one reference epoch?', correctOptionId: 'five-five-two', options: [option('five-five', '(5, 5), with two observations unused'), option('five-five-two', '(5, 5, 2)'), option('four-four-four', '(4, 4, 4)')] },
        { id: 'coverage', prompt: 'After reshuffling without replacement, how many times does each observation appear in one complete epoch?', correctOptionId: 'once', options: [option('variable', 'Possibly zero or several times.'), option('batches', 'Once in every batch.'), option('once', 'Exactly once.')] }],
      hints: ['Take five observations, then another five, then use what remains.', 'A permutation changes order but not the membership or length of the dataset.'],
      reveal: { explanation: 'The batch sizes are 5, 5, and 2. Every observation is used exactly once in an epoch because the shuffled order is a permutation, not a sample with replacement.' },
    }),
    question({ id: 'optimizers-evaluation-mode', title: 'Training Behavior Is Not Evaluation Behavior', datasetId: 'neuralDropout',
      prompt: 'During training, dropout randomly removes activations and rescales the retained ones. During evaluation, it passes all activations through unchanged.',
      instructions: 'Switch modes to compare the same input. For the reference training mask, only the second activation is dropped; the dropout probability is 0.25.',
      parts: [{ id: 'evaluation', prompt: 'What are the output activations in evaluation mode?', correctOptionId: 'unchanged', options: [option('masked', '(4/3, 0, 4, 16/3)'), option('scaled', '(4/3, 8/3, 4, 16/3)'), option('unchanged', '(1, 2, 3, 4)')] },
        { id: 'recording', prompt: 'Turning off gradient recording alone guarantees which of the following?', correctOptionId: 'record-only', options: [option('record-only', 'New operations are not recorded for differentiation; the training/evaluation mode is a separate setting.'), option('dropout-off', 'Dropout is automatically disabled.'), option('params-zero', 'All learned parameters are reset to zero.')] }],
      hints: ['Evaluation uses the unmodified activations, not a new random training mask.', 'Recording derivatives and selecting layer behavior solve two different problems.'],
      reveal: { explanation: 'Evaluation returns (1, 2, 3, 4). Turning off derivative recording saves gradient-tracking work, but evaluation behavior must also be selected so dropout stops masking and batch normalization uses its stored evaluation statistics.' },
    }), neuralFoundationsNotebookLabs.optimizers,
  ],
}

const convolutions: AssignmentSpec = {
  id: 'convolutions', version: 1, displayNumber: 15, published: true, title: 'Convolutions and Pooling', topic: 'Local Image Features',
  description: 'Inspect a sliding contrast detector, shared filter weights, and pooling information loss, then trace image-array operations.',
  questions: [
    question({ id: 'convolutions-local-contrast', title: 'Read a Local Contrast Response', datasetId: 'neuralCorrelation',
      prompt: 'The filter subtracts the two left-column intensities from the two right-column intensities in each 2 × 2 patch.',
      instructions: 'Move the highlighted patch. For the scored reference use the top-left patch (row 0, column 0), which contains [[1, 2], [3, 4]]. The filter is [[−1, 1], [−1, 1]] without flipping.',
      parts: [{ id: 'response', prompt: 'What is the reference response?', correctOptionId: 'two', options: [option('ten', '10'), option('two', '2'), option('minus-two', '−2')] },
        { id: 'brightness', prompt: 'If 5 is added to all four intensities in that patch, what happens to this filter’s response?', correctOptionId: 'same', options: [option('plus-five', 'It increases by 5.'), option('plus-twenty', 'It increases by 20.'), option('same', 'It stays the same.')] }],
      hints: ['Compute (2 + 4) − (1 + 3).', 'The filter weights sum to zero, so a uniform brightness shift contributes zero.'],
      reveal: { explanation: 'The response is 6 − 4 = 2. Adding equal brightness to every cell cancels between the positive and negative filter weights, so the detector responds to contrast rather than a uniform offset.' },
    }),
    question({ id: 'convolutions-shared-filter', title: 'One Detector, Many Locations', datasetId: 'neuralCorrelation',
      prompt: 'A filter reuses the same small set of weights at every valid spatial position. No separate weights are learned for each location.',
      instructions: 'Use the fixed 4 × 4 image and 2 × 2 filter, stride 1, no padding. Explore the response map without changing these reference dimensions.',
      parts: [{ id: 'map', prompt: 'What is the spatial shape of the response map?', correctOptionId: 'three-three', options: [option('two-two', '(2, 2)'), option('four-four', '(4, 4)'), option('three-three', '(3, 3)')] },
        { id: 'weights', prompt: 'If this filter’s entries were learned, how many filter weights would be shared across the whole response map, excluding any bias?', correctOptionId: 'four', options: [option('four', '4'), option('nine', '9'), option('thirty-six', '36')] }],
      hints: ['A two-cell window can begin in three positions along an axis of length four.', 'Count the entries of one filter, not the total number of multiply operations.'],
      reveal: { explanation: 'There are three valid positions along each axis, giving a 3 × 3 response map. Its nine outputs reuse the same four filter weights; parameter sharing is distinct from the amount of computation.' },
    }),
    question({ id: 'convolutions-pooling-information', title: 'Keep a Peak, Lose Its Exact Position', datasetId: 'neuralPooling',
      prompt: 'Max pooling summarizes each retained block by one maximum. It keeps the strongest response but loses some location information.',
      instructions: 'The reference uses the displayed 3 × 5 grid and complete 2 × 2 blocks. The buttons explore rearranging values within the first block; neither its maximum nor the discarded border changes.',
      parts: [{ id: 'output', prompt: 'What is the pooled output for the reference image?', correctOptionId: 'five-six', options: [option('ninety-nine', '[[99, 99]]'), option('five-six', '[[5, 6]]'), option('five-six-border', '[[5, 6, 99], [99, 99, 99]]')] },
        { id: 'location', prompt: 'Moving the value 5 to another cell inside the first block, while permuting its other values, does what to that block’s pooled output?', correctOptionId: 'same', options: [option('position', 'The output records the new position of 5.'), option('smaller', 'It necessarily becomes smaller.'), option('same', 'It stays 5; the exact location is not retained.')] }],
      hints: ['The incomplete bottom row and rightmost column are outside the complete blocks.', 'The maximum depends on which values a block contains, not their order within it.'],
      reveal: { explanation: 'The complete blocks have maxima 5 and 6. All values 99 are on discarded edges. Permuting the values inside a retained block leaves its maximum unchanged, which shows the local position information pooling loses.' },
    }), neuralFoundationsNotebookLabs.convolutions,
  ],
}

export const neuralFoundationsAssignments: AssignmentSpec[] = [mlps, autograd, pytorch, optimizers, convolutions]
