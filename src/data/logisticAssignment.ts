import type { AssignmentSpec, BaseQuestionSpec, MultipleChoiceQuestionSpec } from '../types'
import { logisticBestFeasiblePrecision, logisticMinimumRecall, logisticThresholdMetrics, isLogisticThreshold } from '../lib/logisticConcepts'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import { logisticRegressionNotebookLab } from './classificationCodeLabQuestions'

export interface LogisticVisualDataset {
  id: string
  kind: 'logisticVisual'
  mode: 'sigmoid' | 'boundary' | 'loss'
  caption: string
}

export const logisticVisualDatasets: Record<string, LogisticVisualDataset> = {
  logisticScores: {
    id: 'logisticScores', kind: 'logisticVisual', mode: 'sigmoid',
    caption: 'The horizontal axis is the linear score z, not a probability. The vertical axis is the predicted probability of class 1: p = 1 / (1 + e^(−z)).',
  },
  logisticBoundary: {
    id: 'logisticBoundary', kind: 'logisticVisual', mode: 'boundary',
    caption: 'Both axes are input features. The model has score z = x₁ + 2x₂ − 4 and probability p = 1 / (1 + e^(−z)). The line is where p = 0.5. A–D are observations to classify, not their true class labels.',
  },
  logisticLoss: {
    id: 'logisticLoss', kind: 'logisticVisual', mode: 'loss',
    caption: 'Two already-fitted models predict the same four held-out observations. Both use p > 0.5 to predict class 1. The bars show each observation’s contribution to cross-entropy, on a shared vertical scale.',
  },
}

export interface LogisticThresholdQuestionSpec extends BaseQuestionSpec<'logisticThreshold'> {
  initialThreshold: number
}

export const logisticThresholdQuestion: LogisticThresholdQuestionSpec = {
  id: 'logistic-threshold-tradeoff', kind: 'logisticThreshold', initialThreshold: 0.5,
  title: 'Choose a Threshold for the Task',
  prompt: 'Maximize precision while keeping recall at 80% or higher on these validation observations.',
  instructions: 'Move the threshold from 0.05 to 0.95 in steps of 0.05. Predict class 1 only when p > threshold; equality predicts class 0. Probabilities and true labels stay fixed: nothing is retrained. Any allowed threshold tied for the highest precision among those meeting the recall requirement is accepted. Precision is undefined if no observations are predicted positive.',
  hintSchedule: [2, 4],
  hints: [
    'Recall counts how many actual class-1 observations you catch. First find thresholds that catch at least four of the five.',
    'Among those feasible thresholds, compare precision: what fraction of the predicted-positive observations are actually class 1? A high precision is not enough if recall is too low.',
  ],
  validator: submission => {
    const threshold = submission && typeof submission === 'object' && 'threshold' in submission ? submission.threshold : null
    if (!isLogisticThreshold(threshold)) return { correct: false, message: 'Choose a threshold from 0.05 to 0.95 in steps of 0.05.' }
    const metrics = logisticThresholdMetrics(threshold)
    if (metrics.recall === null || metrics.recall < logisticMinimumRecall) return { correct: false, message: 'Recall is below 80%. Adjust the threshold to catch at least four actual class-1 observations.' }
    const correct = metrics.precision !== null && metrics.precision >= logisticBestFeasiblePrecision - 1e-12
    return { correct, message: correct ? 'Correct: this threshold achieves the highest precision available while meeting the recall requirement.' : 'This threshold meets the recall requirement, but another allowed threshold has higher precision. Compare the false positives.' }
  },
  reveal: { explanation: 'Thresholds 0.40 and 0.45 both catch 4 of 5 positives (80% recall), with 2 false positives (4/6 ≈ 66.7% precision). Higher thresholds miss too many positives; lower feasible thresholds have lower precision. The task determines the tradeoff. Changing the threshold changes decisions, not probabilities or cross-entropy. Use a separate untouched test set for a final evaluation after choosing the threshold.' },
}

const option = (id: string, title: string) => ({ id, title, description: '' })
function question(input: Omit<MultipleChoiceQuestionSpec, 'kind' | 'validator' | 'hintSchedule'>): MultipleChoiceQuestionSpec {
  return { ...input, kind: 'multipleChoice', hintSchedule: [2, 4], validator: validateMultipleChoiceSelections(
    Object.fromEntries(input.parts.map(part => [part.id, part.correctOptionId])),
    'Use the displayed probabilities, geometry, and decision rule to reconsider your selections.',
  ) }
}

export const logisticRegressionAssignment: AssignmentSpec = {
  id: 'logistic-regression', version: 2, displayNumber: 9, published: true,
  title: 'Logistic Regression', topic: 'Binary Classification',
  description: 'Turn linear scores into probabilities, interpret decision boundaries, choose a threshold for a task, and compare probability-based losses before tracing the implementation.',
  questions: [
    question({
      id: 'logistic-scores-to-probabilities', title: 'From Score to Probability',
      prompt: 'A logistic model first computes a linear score, then applies the sigmoid curve shown below.',
      instructions: 'Use the fixed reference scores A = −2, B = 0, and C = 2; the exploratory slider does not move them. In this assignment the prediction is class 1 only if p > 0.5; a tie at 0.5 is class 0. No exponential calculation is needed.',
      datasetId: 'logisticScores',
      parts: [
        { id: 'classes', prompt: 'Which class predictions correspond to A, B, C, in that order?', correctOptionId: 'zero-zero-one', options: [
          option('zero-one-one', '0, 1, 1'), option('zero-zero-one', '0, 0, 1'), option('one-zero-zero', '1, 0, 0'),
        ] },
        { id: 'change', prompt: 'As the score increases from 2 to 4, what happens to the predicted probability of class 1?', correctOptionId: 'saturate', options: [
          option('double', 'It doubles because the score doubles.'), option('unchanged', 'It stays fixed because both scores predict class 1.'),
          option('saturate', 'It increases toward 1, with a smaller change than when the score rose from 0 to 2.'),
        ] },
      ],
      hints: ['Positive scores lie above probability 0.5; a score of zero lies exactly at 0.5.', 'The sigmoid flattens toward its upper end. A class label discards information about probability.'],
      reveal: { explanation: 'The three probabilities are approximately 0.119, 0.500, and 0.881, so the strict decision rule gives 0, 0, 1. At score 4 the probability is about 0.982. Scores and probabilities are not interchangeable, and a probability can change even when the predicted class does not.' },
    }),
    question({
      id: 'logistic-boundary-geometry', title: 'Read the Decision Boundary',
      prompt: 'Use the model’s linear boundary to classify the observations and reason about a stricter threshold.',
      instructions: 'Explore the threshold slider, then answer at the settings specified in each part. At threshold 0.5, class 1 requires z > 0; a point on the boundary predicts class 0 under the strict rule. For the second part, keep the model’s weights and intercept fixed.',
      datasetId: 'logisticBoundary',
      parts: [
        { id: 'positive-region', prompt: 'Which of the four observations are predicted as class 1 at threshold 0.5?', correctOptionId: 'B-D', options: [
          option('B-C-D', 'B, C, and D'), option('A-C', 'A and C'), option('B-D', 'B and D'), option('D', 'D only'),
        ] },
        { id: 'stricter', prompt: 'If the probability threshold rises from 0.5 to 0.75, how does the class-1 region change?', correctOptionId: 'parallel-smaller', options: [
          option('parallel-smaller', 'The boundary shifts parallel toward the upper-right side, shrinking the class-1 region.'),
          option('rotate', 'The boundary rotates, because the relative weights of the two features change.'),
          option('larger', 'The boundary shifts toward the lower-left side, expanding the class-1 region.'),
          option('same', 'The region is unchanged unless the model is retrained.'),
        ] },
      ],
      hints: ['Larger x₁ or larger x₂ increases the score. Which side of the line contains B and D?', 'A higher probability threshold requires a higher linear score. It changes the cutoff, not the relative feature weights.'],
      reveal: { explanation: 'A, B, C, D have scores −1, 0.5, 0, 1.5. Only B and D have positive scores. At threshold 0.75, the boundary is z = log(3) ≈ 1.10, so it is parallel to the old boundary and farther into the positive-score side. The probability surface itself has not changed.' },
    }),
    logisticThresholdQuestion,
    question({
      id: 'logistic-confident-mistakes', title: 'Accuracy Misses Confident Mistakes',
      prompt: 'Models A and B both classify three of four observations correctly, but they make different probability predictions.',
      instructions: 'For one observation, cross-entropy is −ln(p) if its true class is 1, or −ln(1 − p) if its true class is 0. Smaller is better. Mean cross-entropy averages the four bar heights. Compare the bars; you do not need to calculate logarithms.',
      datasetId: 'logisticLoss',
      parts: [
        { id: 'compare', prompt: 'Which model has lower mean cross-entropy, and why?', correctOptionId: 'A-confident-error', options: [
          option('B-confident', 'B, because its three correct predictions are more confident.'),
          option('same', 'They tie, because their predicted classes and accuracies are the same.'),
          option('A-confident-error', 'A: B’s very confident mistake on observation 4 outweighs its improvements on the other three.'),
        ] },
        { id: 'threshold-loss', prompt: 'Change only a model’s classification threshold, keeping its predicted probabilities fixed. What can change?', correctOptionId: 'decisions-only', options: [
          option('loss-only', 'Cross-entropy can change, but predicted classes cannot.'),
          option('decisions-only', 'Predicted classes and accuracy can change; cross-entropy cannot.'),
          option('all', 'Predicted probabilities, classes, accuracy, and cross-entropy must all change.'),
        ] },
      ],
      hints: ['Compare the total height of all four bars for each model, not just the three correct predictions.', 'Cross-entropy uses the probabilities and true labels. The classification threshold does not appear in its formula.'],
      reveal: { explanation: 'Mean cross-entropy is about 0.535 for A and 1.190 for B, despite both having 75% accuracy. Assigning probability 0.99 to class 1 when the truth is class 0 incurs loss −ln(0.01) ≈ 4.605. Training adjusts weights to improve probability-based loss; choosing a threshold afterward sets the decision tradeoff without changing that loss.' },
    }),
    logisticRegressionNotebookLab,
  ],
}
