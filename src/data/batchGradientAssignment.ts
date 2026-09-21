import type { AssignmentSpec, MultipleChoiceQuestionSpec } from '../types'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import { batchGradientDescentNotebookLab } from './classificationCodeLabQuestions'

export interface BatchGradientVisualDataset {
  id: string
  kind: 'batchGradientVisual'
  mode: 'directions' | 'explorer' | 'budget' | 'shuffle' | 'nearMinimum'
  caption: string
}

export const batchGradientVisualDatasets: Record<string, BatchGradientVisualDataset> = {
  batchDirections: { id: 'batchDirections', kind: 'batchGradientVisual', mode: 'directions', caption: 'One adjustable constant w predicts the same value for every row. The horizontal axis identifies observations; it is not an input feature.' },
  batchExplorer: { id: 'batchExplorer', kind: 'batchGradientVisual', mode: 'explorer', caption: 'A real optimization trace for the same eight targets. Every batch-size change restarts at w = 0 and uses the same four fixed shuffled row orders, with learning rate 0.4.' },
  batchBudget: { id: 'batchBudget', kind: 'batchGradientVisual', mode: 'budget', caption: 'One epoch visits each of 120 training observations once. Each block below causes one parameter update. Every row represents the same total number of observations.' },
  batchShuffle: { id: 'batchShuffle', kind: 'batchGradientVisual', mode: 'shuffle', caption: 'Each numbered card is a two-row batch; each line identifies one observation and its target. Both schedules visit exactly the same eight observations once.' },
  batchNearMinimum: { id: 'batchNearMinimum', kind: 'batchGradientVisual', mode: 'nearMinimum', caption: 'The solid curve averages all eight observations. The dashed curve uses only row R7, whose target is 7. Both losses are half mean squared error, on identical axes.' },
}

const option = (id: string, title: string) => ({ id, title, description: '' })
function question(input: Omit<MultipleChoiceQuestionSpec, 'kind' | 'validator' | 'hintSchedule'>): MultipleChoiceQuestionSpec {
  return { ...input, kind: 'multipleChoice', hintSchedule: [2, 4],
    validator: validateMultipleChoiceSelections(Object.fromEntries(input.parts.map(p => [p.id, p.correctOptionId])), 'Use the displayed observations, loss curves, or update counts to reconsider your selections.') }
}

export const batchGradientDescentAssignment: AssignmentSpec = {
  id: 'batch-gradient-descent', version: 2, displayNumber: 7, published: true,
  title: 'Batch Gradient Descent', topic: 'Optimization with Subsets of Data',
  description: 'Compare batch and full-data gradients, explore noisy updates, distinguish epochs from updates, and reason about shuffling before tracing the notebook implementation.',
  questions: [
    question({
      id: 'batch-directions-concept', title: 'Can a Batch Point the Other Way?',
      prompt: 'The current prediction is w = 3. Compare a step using the first four rows with a step using all eight rows.',
      instructions: 'Explore the prediction line, then return to w = 3 for both questions. Here the loss is half the mean squared error. For this constant-prediction model, a group’s gradient is w minus that group’s mean target. A small positive learning rate moves w opposite the gradient. Each update uses the mean gradient of its chosen rows.',
      datasetId: 'batchDirections',
      parts: [{ id: 'directions', prompt: 'Which way does the first update move w?', correctOptionId: 'batch-left-full-right', options: [
        option('both-right', 'Right for both the first-four-row batch and the full dataset.'),
        option('batch-left-full-right', 'Left for the first-four-row batch; right for the full dataset.'),
        option('both-left', 'Left for both the first-four-row batch and the full dataset.'),
        option('batch-right-full-left', 'Right for the first-four-row batch; left for the full dataset.'),
      ] }, { id: 'average', prompt: 'At the SAME current w, how does the full-data gradient relate to the mean gradients of the first four and last four rows?', correctOptionId: 'mean', options: [
        option('sum', 'It is their sum.'), option('choose', 'It equals whichever batch gradient has larger magnitude.'),
        option('mean', 'It is their average, because these two batches have equal size.'),
      ] }],
      hints: ['Compare 3 with the mean of targets 0, 1, 2, 3, then with the mean of all eight targets.', 'For equal-sized groups evaluated at the same w, averaging the two group means gives the overall mean.'],
      reveal: { explanation: 'The first four targets average 1.5, so their gradient is 3 − 1.5 = 1.5 and w decreases. All eight average 4, so the full gradient is −1 and w increases. The other four have gradient −3.5; averaging 1.5 and −3.5 gives −1. A batch need not agree with the full-data direction. This averaging identity requires evaluating both gradients at the same parameter value.' },
    }),
    question({
      id: 'batch-noisy-updates', title: 'Explore What a Noisy Update Does',
      prompt: 'Use the batch-size selector and update viewer to explore four epochs. Then inspect batch size 1, update 5.',
      instructions: 'The curve always measures loss over ALL eight training rows, even when an update uses a smaller batch. The table compares the same chosen batch before and after one update. Loss is half MSE; gradients are batch means. Batch size 1 means stochastic gradient descent; batch size 8 uses the full dataset.',
      datasetId: 'batchExplorer',
      parts: [{ id: 'fifth-update', prompt: 'For batch size 1, what happens at update 5?', correctOptionId: 'batch-down-full-up', options: [
        option('both-down', 'Both the chosen batch’s loss and the full-data loss decrease.'),
        option('both-up', 'Both the chosen batch’s loss and the full-data loss increase.'),
        option('batch-down-full-up', 'The chosen batch’s loss decreases, but the full-data loss increases.'),
        option('batch-up-full-down', 'The chosen batch’s loss increases, but the full-data loss decreases.'),
      ] }, { id: 'interpret', prompt: 'What explains this behavior?', correctOptionId: 'different-objective', options: [
        option('different-objective', 'The selected row favors a different prediction from the average of all rows; decreasing its loss need not decrease the full-data loss.'),
        option('corrupt', 'Any increase in full-data loss proves the observations were mismatched with their targets.'),
        option('validation', 'The curve increased because it switched from training data to validation data at that step.'),
      ] }],
      hints: ['Use “Show batch size 1, update 5” to return to the requested comparison. Read both rows of the before/after table.', 'At this step the selected target is 1, while the best constant prediction over all eight rows is 4.'],
      reveal: { explanation: 'Update 5 moves w from 4.1280 to 2.8768 toward the selected target 1. That row’s half-squared error falls from 4.8922 to 1.7612, but full-data loss rises from 3.7582 to 4.3808. Mini-batch progress need not be monotonic on the full training objective. This isolated increase does not by itself prove divergence, overfitting, or a data error.' },
    }),
    question({
      id: 'batch-epochs-budget', title: 'Compare Equal Amounts of Data',
      prompt: 'Two training plans both make eight complete passes through the same 120 observations.',
      instructions: 'Plan A uses batches of 20. Plan B uses all 120 observations in each update. Count one per-observation gradient contribution each time a row participates in an update. This is a simplified work count, not a measured wall-clock runtime.',
      datasetId: 'batchBudget',
      parts: [{ id: 'updates', prompt: 'After eight epochs, how many updates has each plan made?', correctOptionId: '48-8', options: [
        option('8-8', 'A: 8; B: 8.'), option('48-960', 'A: 48; B: 960.'), option('48-8', 'A: 48; B: 8.'), option('960-960', 'A: 960; B: 960.'),
      ] }, { id: 'work', prompt: 'Which comparison is justified?', correctOptionId: 'same-visits', options: [
        option('six-times', 'A must take six times longer because it makes six times as many updates.'),
        option('same-visits', 'Both use 960 per-observation gradient contributions, but A updates parameters more often; runtime and final accuracy still need to be measured.'),
        option('guarantee', 'A must have the lower final loss because its batch size is smaller.'),
      ] }],
      hints: ['A has 120/20 updates per epoch; B has one. Multiply by eight.', 'An epoch counts passes over observations, not parameter updates. Larger batches can also use hardware more efficiently.'],
      reveal: { explanation: 'Plan A makes 6 × 8 = 48 updates; Plan B makes 1 × 8 = 8. Both process 120 × 8 = 960 observations. Equal update counts would not be an equal-work comparison. Smaller batches update sooner and use fewer rows per update, but overhead, parallelism, learning rate, and noise affect time to reach a desired loss.' },
    }),
    question({
      id: 'batch-shuffle-concept', title: 'Reason About the Order of Observations',
      prompt: 'These two schedules form different two-row batches from the same training set.',
      instructions: 'Schedule A is sorted by target; schedule B is one possible shuffle. Inspect the actual targets within each batch. When shuffling a supervised-learning dataset, an observation consists of its input features together with its target.',
      datasetId: 'batchShuffle',
      parts: [{ id: 'composition', prompt: 'Which description matches these schedules?', correctOptionId: 'mix', options: [
        option('same', 'The batches must have identical gradients because the epoch contains the same observations.'),
        option('mix', 'A puts only low targets in early batches and only high targets in later ones; this particular shuffle mixes low and high targets within batches.'),
        option('guaranteed', 'Every random shuffle guarantees every batch has exactly the full-data mean target.'),
      ] }, { id: 'pairing', prompt: 'How should features and targets be reordered?', correctOptionId: 'pairs', options: [
        option('independent', 'Shuffle feature rows and target values independently.'), option('targets', 'Shuffle only targets; leave feature rows fixed.'),
        option('pairs', 'Move each feature row together with its original target using one shared ordering.'),
      ] }],
      hints: ['For A, compare the first batch’s targets with the last batch’s targets. Then inspect B.', 'Shuffling changes presentation order, not which target belongs to each observation.'],
      reveal: { explanation: 'Sorted targets can give a sequence of systematically different batch directions. This shuffle mixes targets more evenly, although not every shuffle or batch is perfectly representative. Because parameters change between batches, ordering can change the trajectory even when each epoch uses the same observations. Inputs and their targets must remain paired.' },
    }),
    question({
      id: 'batch-at-optimum', title: 'Why Can SGD Keep Moving Near a Minimum?',
      prompt: 'The full-data loss is minimized at w = 4. Now select only row R7, whose target is 7.',
      instructions: 'Use the solid and dashed loss curves to compare their slopes at w = 4. Explore how changing the learning rate affects one R7-only update from that same starting point. Each update subtracts learning rate × the selected batch’s mean gradient; all allowed rates are positive.',
      datasetId: 'batchNearMinimum',
      parts: [{ id: 'next-step', prompt: 'What happens from w = 4?', correctOptionId: 'full-stays-single-right', options: [
        option('both-stay', 'Both a full-data update and the R7-only update leave w unchanged.'),
        option('full-stays-single-right', 'The full-data update leaves w unchanged; the R7-only update moves w to the right.'),
        option('full-right-single-stays', 'The full-data update moves right; the R7-only update leaves w unchanged.'),
      ] }, { id: 'smaller-rate', prompt: 'For the SAME selected row and starting w = 4, what does halving the learning rate do?', correctOptionId: 'half-step', options: [
        option('zero-gradient', 'It makes the selected row’s gradient exactly zero.'),
        option('half-step', 'It halves the size of that update; smaller rates can reduce fluctuations near the optimum.'),
        option('change-optimum', 'It changes the full-data loss minimizer from 4 to 2.'),
      ] }],
      hints: ['At w = 4, the solid curve is flat but the dashed curve is not.', 'The gradient is set by the current parameter and chosen data. The learning rate multiplies it.'],
      reveal: { explanation: 'The full gradient is 4 − 4 = 0, but the R7 gradient is 4 − 7 = −3, so its update moves right. Individual gradients can be nonzero even when their average is zero. With heterogeneous observations, a fixed positive learning rate can keep stochastic updates fluctuating near the best fit. A smaller rate reduces these steps but may also slow earlier progress.' },
    }),
    batchGradientDescentNotebookLab,
  ],
}
