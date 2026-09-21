import type { AssignmentSpec, BaseQuestionSpec, MultipleChoiceQuestionSpec } from '../types'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import { isRidgeStrengthIndex, minimumRidgeValidationMSE, ridgeFits } from '../lib/regularizationConcepts'
import { regularizationNotebookLab } from './classificationCodeLabQuestions'

export interface RegularizationVisualDataset {
  kind: 'regularizationVisual'
  mode: 'fit' | 'paths' | 'scaling' | 'evaluation'
  caption: string
}
export const regularizationVisualDatasets: Record<string, RegularizationVisualDataset> = {
  regularizationFit: { kind: 'regularizationVisual', mode: 'fit', caption: 'The same four training observations and three candidate lines, all with intercept 1. For this comparison λ = 1; only the slope w is penalized.' },
  regularizationPaths: { kind: 'regularizationVisual', mode: 'paths', caption: 'Exact optimal coefficients as penalty strength changes in a small independent-coordinate example. Both plots start from (2, −1, 0.4). Axes and feature labels match across panels.' },
  regularizationScaling: { kind: 'regularizationVisual', mode: 'scaling', caption: 'Two coordinate systems for exactly the same physical measurements and predictions. Only the distance unit changes.' },
  regularizationEvaluation: { kind: 'regularizationVisual', mode: 'evaluation', caption: 'Three disjoint sets, each with a different job. Fitting a parameter and choosing a hyperparameter are both ways information can affect the final model.' },
}

function question(input: Omit<MultipleChoiceQuestionSpec, 'kind' | 'validator' | 'hintSchedule'>): MultipleChoiceQuestionSpec {
  return { ...input, kind: 'multipleChoice', hintSchedule: [2,4], validator: validateMultipleChoiceSelections(
    Object.fromEntries(input.parts.map(part => [part.id, part.correctOptionId])),
    'Compare your selections with the displayed models, metrics, and stated objective.',
  ) }
}
const option = (id: string, title: string) => ({ id, title, description: '' })

export interface RegularizationTuningQuestionSpec extends BaseQuestionSpec<'regularizationTuning'> {
  initialStrengthIndex: number
}
export const regularizationTuningQuestion: RegularizationTuningQuestionSpec = {
  id: 'regularization-tuning', kind: 'regularizationTuning', initialStrengthIndex: 0,
  title: 'Tune the Penalty, Keep the Degree Fixed',
  prompt: 'Choose the L2 strength with the smallest validation MSE among the six slider settings.',
  instructions: 'Every setting fits the same degree-8 polynomial using features (x/2), (x/2)², …, (x/2)⁸ by minimizing training MSE + λ × the sum of squared coefficients. The intercept is not penalized. Changing λ does not change the degree or use validation observations to fit the coefficients. The slider visits 0, 0.001, 0.01, 0.1, 1, and 10; compare the four-decimal readouts.',
  hintSchedule: [2,4], hints: [
    'The tightest fit to training observations need not have the smallest error on the validation crosses.',
    'Compare a weak penalty with both no penalty and a strong penalty. A very strong penalty makes this model nearly constant.',
  ],
  validator: submission => {
    const index = submission && typeof submission === 'object' && 'strengthIndex' in submission ? submission.strengthIndex : null
    if (!isRidgeStrengthIndex(index)) return { correct: false, message: 'Choose one of the six penalty settings on the slider.' }
    const correct = ridgeFits[index].validationMSE <= minimumRidgeValidationMSE + 1e-10
    return { correct, message: correct ? 'Correct! This setting has the lowest validation MSE among the six candidates.' : 'Another setting has lower validation MSE. Keep the degree fixed and compare the validation readout across the slider.' }
  },
  reveal: { explanation: 'Here λ = 0.001 has the smallest validation MSE (about 0.0846). It gives up some training fit but improves validation predictions relative to λ = 0. Increasing the penalty much further underfits. Regularization strength is a hyperparameter: its useful scale depends on the features, data, and objective convention, and stronger is not always better.' },
}

export const regularizationAssignment: AssignmentSpec = {
  id: 'regularization', version: 2, displayNumber: 8, published: true,
  title: 'Regularization', topic: 'Model Complexity and Generalization',
  description: 'Balance fit and coefficient size, tune L2 regularization visually, compare L1 sparsity with L2 shrinkage, and reason about feature scaling before the notebook lab.',
  questions: [
    question({
      id: 'regularization-fit-and-penalty', title: 'Balance Fit Against Coefficient Size',
      prompt: 'A regularized model is judged by more than its training error. Compare the three candidate lines below.',
      instructions: 'Use objective = training MSE + λw² with λ = 1. The intercept is fixed at 1 and is not penalized. Choose only among A, B, and C; you are not being asked to find the best of all possible lines.',
      datasetId: 'regularizationFit',
      parts: [{ id: 'objective', prompt: 'Which candidate has the smallest regularized objective?', correctOptionId: 'B', options: ['A','B','C'].map(id => option(id, `Line ${id}`)) },
      { id: 'tradeoff', prompt: 'Why can the regularized choice have a larger training MSE than the best-fitting candidate?', correctOptionId: 'combined', options: [
        option('bad-fit', 'Regularization deliberately maximizes training MSE.'),
        option('combined', 'It accepts some extra training error in exchange for a smaller coefficient penalty.'),
        option('zero', 'Regularization requires every coefficient to equal zero.'),
      ] }],
      hints: ['For each candidate, add its training MSE to the square of its slope.', 'The objective contains two terms; improving one can worsen the other.'],
      reveal: { explanation: 'A has objective 0.45 + 2.4² = 6.21; B has 2.25 + 1.2² = 3.69; C has 7.65 + 0² = 7.65. B is best among these three. This does not prove B will predict unseen data best: no validation data have been shown for this example.' },
    }),
    regularizationTuningQuestion,
    question({
      id: 'regularization-coefficient-paths', title: 'Shrink a Feature or Remove It?',
      prompt: 'Two penalty methods change the three coefficients differently as regularization grows.',
      instructions: 'These are coefficient paths, not prediction curves or optimization steps. Each point represents a fully minimized objective at that penalty strength. L1 adds a sum of absolute coefficient values; L2 adds a sum of squared coefficient values.',
      datasetId: 'regularizationPaths',
      parts: [{ id: 'method', prompt: 'Which plot illustrates L1, with coefficients reaching zero and remaining zero as the penalty grows?', correctOptionId: 'A', options: [option('B','Plot B'),option('A','Plot A'),option('neither','Neither plot')] },
      { id: 'zero', prompt: 'At λ = 0.6 in plot A, what does w₃ = 0 tell you?', correctOptionId: 'removed', options: [
        option('irrelevant', 'Feature 3 is unrelated to the target in every possible dataset.'),
        option('none', 'Feature 3 still changes this linear model’s prediction through w₃x₃.'),
        option('removed', 'Feature 3 contributes nothing to this fitted linear prediction, though that is not proof of universal irrelevance.'),
      ] }],
      hints: ['Look for a coefficient path that reaches the horizontal zero line and then stays there.', 'In a linear prediction, a feature enters as its coefficient multiplied by its value.'],
      reveal: { explanation: 'Plot A uses L1 and yields (1.4, −0.4, 0) at λ = 0.6. Plot B uses L2 and yields (1.25, −0.625, 0.25): all three are smaller in magnitude but nonzero. L1 can produce sparse exact optima; L2 generally shrinks rather than selecting exact zeros. These plots show exact solutions, not a claim that every finite-step optimization algorithm reaches exact zeros.' },
    }),
    question({
      id: 'regularization-units', title: 'Should Changing Units Change the Penalty?',
      prompt: 'The two lines make identical predictions for the same physical distances, but their numerical slopes differ.',
      instructions: 'Use L2 penalty λw² with the same positive λ in both coordinate systems. Only the slope is penalized.',
      datasetId: 'regularizationScaling',
      parts: [{ id: 'penalty', prompt: 'Which representation receives the smaller slope penalty?', correctOptionId: 'centimeters', options: [
        option('meters', 'Distance in meters, with slope 2.'),option('same', 'They receive equal penalties because predictions match.'),option('centimeters', 'Distance in centimeters, with slope 0.02.'),
      ] },{ id: 'scaling', prompt: 'What is the sensible preprocessing policy before comparing coefficient penalties across differently scaled features?', correctOptionId: 'training', options: [
        option('each', 'Scale training and validation sets independently using each set’s own statistics.'),
        option('training', 'Learn feature scales from training data and reuse those same scales for validation and test data.'),
        option('none', 'Never scale: L1 and L2 penalties are automatically unaffected by measurement units.'),
      ] }],
      hints: ['Compare 2² with 0.02². Identical predictions do not imply identical coefficient magnitudes.', 'The model needs one consistent coordinate system learned without held-out data.'],
      reveal: { explanation: 'The centimeter coefficient has 1/10,000 of the squared penalty of the meter coefficient, even though predictions match. Regularization depends on feature units. Training-fitted standardization can make coefficient penalties more comparable across features; reuse the fitted transformation on held-out data.' },
    }),
    question({
      id: 'regularization-honest-evaluation', title: 'Choose a Penalty Without Using the Test Set',
      prompt: 'You have fitted the six candidates and selected a penalty strength. Decide what evidence is needed next.',
      instructions: 'The sets below are disjoint. Model coefficients and preprocessing were fitted using training data only.',
      datasetId: 'regularizationEvaluation',
      parts: [{ id: 'selection', prompt: 'Which measurement should choose the penalty strength for predictive performance?', correctOptionId: 'validation', options: [
        option('penalty', 'The smallest coefficient penalty, regardless of prediction errors.'),option('training', 'The smallest training MSE.'),option('validation', 'The smallest validation MSE among the candidate settings.'),
      ] },{ id: 'final', prompt: 'After selection, how should you obtain a final independent performance estimate?', correctOptionId: 'test', options: [
        option('test', 'Evaluate the selected model on the untouched test set, without using that result to retune this model.'),
        option('retune', 'Try more strengths until the test MSE is smallest, then report that same test result as independent.'),
        option('guarantee', 'Skip evaluation: a positive penalty guarantees better predictions on new data.'),
      ] }],
      hints: ['A smaller coefficient norm is part of the training objective, not itself a measure of prediction quality.', 'A set used repeatedly to choose the model is no longer independent of that choice.'],
      reveal: { explanation: 'Training data fit coefficients; validation errors guide penalty selection; an untouched test set estimates final performance. Repeated test-driven tuning turns the test set into another selection set. Regularization can help generalization, but improvement is not guaranteed on every dataset.' },
    }),
    regularizationNotebookLab,
  ],
}
