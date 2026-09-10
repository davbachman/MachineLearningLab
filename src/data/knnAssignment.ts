import type {
  AssignmentSpec,
  Knn2DQuestionSpec,
  MultipleChoiceQuestionSpec,
  ValidationResult,
} from '../types'
import { knnInteractiveDatasets } from './knnDatasets'
import {
  classifyGridCells,
  classifyKnnPoint,
  findBestK,
} from '../lib/knnMath'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import { knnCodeLab, knnEvaluationNotebookLab } from './codeLabQuestions'

type KnnDatasetId = keyof typeof knnInteractiveDatasets

function validatePredictSequence(datasetId: KnnDatasetId) {
  const dataset = knnInteractiveDatasets[datasetId]
  if (dataset.kind !== 'predictSequence') {
    throw new Error(`Dataset ${datasetId} is not a predict-sequence dataset.`)
  }

  return (submission: unknown): ValidationResult => {
    const payload = submission as { step?: number; predictedLabel?: number }
    const stepIndex = Math.max(0, Math.min(payload.step ?? 0, dataset.kSequence.length - 1))
    const k = dataset.kSequence[stepIndex]
    const targetLabel = classifyKnnPoint(
      dataset.trainingPoints,
      dataset.queryPoint,
      k,
      dataset.metric,
    ).label

    return {
      correct: payload.predictedLabel === targetLabel,
      message:
        payload.predictedLabel === targetLabel
          ? 'Correct! Good Job!'
          : `Look more carefully at the ${k} nearest neighbors around the query point.`,
    }
  }
}

function validateDecisionBoundary(datasetId: KnnDatasetId) {
  const dataset = knnInteractiveDatasets[datasetId]
  if (dataset.kind !== 'decisionBoundary') {
    throw new Error(`Dataset ${datasetId} is not a decision-boundary dataset.`)
  }

  return (submission: unknown): ValidationResult => {
    const payload = submission as { step?: number; cells?: number[] }
    const stepIndex = Math.max(0, Math.min(payload.step ?? 0, dataset.kSequence.length - 1))
    const k = dataset.kSequence[stepIndex]
    const targetCells = classifyGridCells(
      dataset.trainingPoints,
      dataset.bounds,
      dataset.gridColumns,
      dataset.gridRows,
      k,
      dataset.metric,
    )
    const submittedCells = payload.cells ?? []
    const correct =
      submittedCells.length === targetCells.length &&
      submittedCells.every((cell, index) => cell === targetCells[index])

    return {
      correct,
      message: correct
        ? 'Correct! Good Job!'
        : `At least one grid cell still disagrees with the ${k}-NN decision rule.`,
    }
  }
}

function validateBestK(datasetId: KnnDatasetId) {
  const dataset = knnInteractiveDatasets[datasetId]
  if (dataset.kind !== 'bestK') {
    throw new Error(`Dataset ${datasetId} is not a best-k dataset.`)
  }

  const bestChoice = findBestK(
    dataset.trainingPoints,
    dataset.testPoints,
    dataset.kValues,
    dataset.metric,
  )

  return (submission: unknown): ValidationResult => {
    const payload = submission as { k?: number }
    const correct = payload.k === bestChoice.k

    return {
      correct,
      message: correct
        ? 'Correct! Good Job!'
        : 'A different value of k gets higher accuracy on the held-out test points.',
    }
  }
}

function validateScalingTrap(datasetId: KnnDatasetId) {
  const dataset = knnInteractiveDatasets[datasetId]
  if (dataset.kind !== 'scalingTrap') {
    throw new Error(`Dataset ${datasetId} is not a scaling-trap dataset.`)
  }

  const rawLabel = classifyKnnPoint(
    dataset.trainingPoints,
    dataset.queryPoint,
    dataset.k,
    dataset.metric,
  ).label
  const normalizedLabel = classifyKnnPoint(
    dataset.normalizedTrainingPoints,
    dataset.normalizedQueryPoint,
    dataset.k,
    dataset.metric,
  ).label

  return (submission: unknown): ValidationResult => {
    const payload = submission as { step?: 'raw' | 'normalized'; predictedLabel?: number }
    const targetLabel = payload.step === 'normalized' ? normalizedLabel : rawLabel

    return {
      correct: payload.predictedLabel === targetLabel,
      message:
        payload.predictedLabel === targetLabel
          ? 'Correct! Good Job!'
          : payload.step === 'normalized'
            ? 'After normalization, the distance geometry changes. Re-check the closest neighbors.'
            : 'Without normalization, the wide-range feature dominates the distance calculation.',
    }
  }
}

function validateMetricComparison(datasetId: KnnDatasetId) {
  const dataset = knnInteractiveDatasets[datasetId]
  if (dataset.kind !== 'metricComparison') {
    throw new Error(`Dataset ${datasetId} is not a metric-comparison dataset.`)
  }

  const euclideanLabel = classifyKnnPoint(
    dataset.trainingPoints,
    dataset.queryPoint,
    dataset.k,
    dataset.metric,
  ).label
  const manhattanLabel = classifyKnnPoint(
    dataset.trainingPoints,
    dataset.queryPoint,
    dataset.k,
    dataset.comparisonMetric,
  ).label

  return (submission: unknown): ValidationResult => {
    const payload = submission as { step?: 'euclidean' | 'manhattan'; predictedLabel?: number }
    const targetLabel = payload.step === 'manhattan' ? manhattanLabel : euclideanLabel

    return {
      correct: payload.predictedLabel === targetLabel,
      message:
        payload.predictedLabel === targetLabel
          ? 'Correct! Good Job!'
          : payload.step === 'manhattan'
            ? 'Under Manhattan distance, axis-aligned paths matter more than diagonal shortcuts.'
            : 'Under Euclidean distance, the diagonally nearby points matter more than under Manhattan distance.',
    }
  }
}

function validateAdversarialPlacement(datasetId: KnnDatasetId) {
  const dataset = knnInteractiveDatasets[datasetId]
  if (dataset.kind !== 'adversarialPlacement') {
    throw new Error(`Dataset ${datasetId} is not an adversarial-placement dataset.`)
  }

  const originalLabel = classifyKnnPoint(
    dataset.trainingPoints,
    dataset.queryPoint,
    dataset.k,
    dataset.metric,
  ).label

  return (submission: unknown): ValidationResult => {
    const payload = submission as { point?: [number, number] | undefined }
    if (!payload.point) {
      return {
        correct: false,
        message: 'Place the new training point somewhere on the plot before checking.',
      }
    }

    const updatedLabel = classifyKnnPoint(
      [...dataset.trainingPoints, { point: payload.point, label: dataset.examplePoint.label }],
      dataset.queryPoint,
      dataset.k,
      dataset.metric,
    ).label

    return {
      correct: updatedLabel !== originalLabel,
      message:
        updatedLabel !== originalLabel
          ? 'Correct! Good Job!'
          : 'That new point still does not change the majority among the three nearest neighbors.',
    }
  }
}

export const knnAssignment: AssignmentSpec = {
  id: 'knn',
  displayNumber: 3,
  version: 8,
  title: 'k-Nearest Neighbors',
  topic: 'Classification',
  description:
    'Practice local voting, boundary shape, choosing k, train/test evaluation, scaling, distance metrics, and adversarial sensitivity.',
  published: true,
  questions: [
    {
      id: 'knn-predict-sequence',
      kind: 'knn2d',
      title: 'Predict the Classification',
      prompt:
        'Predict the class of the query point for k = 3, then k = 5, then k = n on the same dataset.',
      instructions:
        'The points do not move. Only the size of the neighborhood changes from one step to the next.',
      datasetId: 'predictSequence',
      interactionMode: 'predictSequence',
      hintSchedule: [2, 4, 6],
      hints: [
        'For k = 3, look only at the three closest points to the query point, not the whole cloud.',
        'For k = 5, the local majority changes because more orange points enter the vote.',
        'For k = n, every training point votes, so the global class balance matters most.',
      ],
      validator: validatePredictSequence('predictSequence'),
      reveal: {
        explanation:
          'Small k values focus on very local structure, while larger k values smooth the decision toward the broader class balance. The same query point can change labels as k grows.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies Knn2DQuestionSpec,
    {
      id: 'knn-decision-boundary',
      kind: 'knn2d',
      title: 'Sketch the Decision Boundary',
      prompt:
        'Color each grid cell with the class that k-NN would predict there, first for k = 1 and then for k = 5.',
      instructions:
        'Predict the class at each cell’s center. Hover over or focus a cell to highlight its nearest neighbors, then count their votes and paint the cell. On a touchscreen, enable Inspect neighbors to examine a cell before painting. Your k = 1 coloring carries forward to k = 5; update the cells whose predictions change. After checking, dashed outlines mark cells that need correction.',
      datasetId: 'decisionBoundary',
      interactionMode: 'decisionBoundary',
      hintSchedule: [2, 4],
      hints: [
        'At k = 1, a single outlier can carve out a tiny island of its own class.',
        'At k = 5, the local islands shrink because each cell listens to a larger neighborhood.',
      ],
      validator: validateDecisionBoundary('decisionBoundary'),
      reveal: {
        explanation:
          'The k = 1 boundary hugs individual points closely, including isolated outliers. With k = 5, each location averages over a wider neighborhood, so the boundary becomes much smoother.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies Knn2DQuestionSpec,
    {
      id: 'knn-best-k',
      kind: 'knn2d',
      title: 'Choose the Best k',
      prompt:
        'Move the slider and find the value of k that maximizes accuracy on the held-out test points.',
      instructions:
        'Training points are filled. Test points are shown in a different style. Watch how the boundary and the training/test accuracies move together.',
      datasetId: 'bestK',
      interactionMode: 'bestK',
      hintSchedule: [2, 4],
      hints: [
        'Very small k can fit every training point but still generalize badly to held-out data.',
        'Very large k smooths too aggressively and can blur the messy overlap region.',
      ],
      validator: validateBestK('bestK'),
      reveal: {
        explanation:
          'The best choice of k is the one that generalizes best, not the one that merely memorizes the training set. The test-accuracy curve peaks at an intermediate value here.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies Knn2DQuestionSpec,
    {
      id: 'knn-scaling-trap',
      kind: 'knn2d',
      title: 'The Scaling Trap',
      prompt:
        'Using k = 3, predict the query point before normalization and then again after min-max normalization.',
      instructions:
        'Feature 1 has a much wider numerical range than Feature 2. After the first step, the plot switches to min-max normalized coordinates.',
      datasetId: 'scalingTrap',
      interactionMode: 'scalingTrap',
      hintSchedule: [2, 4],
      hints: [
        'Without scaling, the wide-range horizontal feature dominates the Euclidean distance.',
        'After normalization, vertical differences matter on the same footing as horizontal differences.',
      ],
      validator: validateScalingTrap('scalingTrap'),
      reveal: {
        explanation:
          'KNN depends directly on geometric distance, so feature scale matters immediately. Normalizing the features changes the geometry of the space and can change the predicted label.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies Knn2DQuestionSpec,
    {
      id: 'knn-metric-comparison',
      kind: 'knn2d',
      title: 'Distance Metric Comparison',
      prompt:
        'Using k = 3, predict the query point’s class under Euclidean distance, then repeat under Manhattan distance.',
      instructions:
        'Use k = 3 in both parts. Only the metric changes; the points stay fixed.',
      datasetId: 'metricComparison',
      interactionMode: 'metricComparison',
      hintSchedule: [2, 4],
      hints: [
        'Euclidean distance treats diagonal closeness directly.',
        'Manhattan distance favors axis-aligned travel, so the nearest set can change.',
      ],
      validator: validateMetricComparison('metricComparison'),
      reveal: {
        explanation:
          'Different distance metrics define “near” differently. Even with the same data and the same k, Euclidean and Manhattan distance can produce different nearest neighbors and different class predictions.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies Knn2DQuestionSpec,
    {
      id: 'knn-adversarial-placement',
      kind: 'knn2d',
      title: 'Adversarial Placement',
      prompt:
        'Add one new orange training point that flips the current 3-NN prediction for the query point.',
      instructions:
        'Click on the plot to place the new point. You can reposition it by clicking again before checking your answer.',
      datasetId: 'adversarialPlacement',
      interactionMode: 'adversarialPlacement',
      hintSchedule: [2, 4],
      hints: [
        'The new point has to enter the set of the three nearest neighbors to matter.',
        'You only need to flip a 2-to-1 vote into a 2-to-1 vote for the other class.',
      ],
      validator: validateAdversarialPlacement('adversarialPlacement'),
      reveal: {
        explanation:
          'Because KNN is a local voting rule, a single carefully placed point can change the prediction if it enters the neighborhood that determines the vote.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies Knn2DQuestionSpec,
    {
      id: 'knn-train-test-accuracy',
      kind: 'multipleChoice',
      title: 'Train vs Test Accuracy',
      prompt:
        'Use the prediction table to compute the training and test accuracies, then decide which one better estimates future performance.',
      instructions:
        'Count correct predictions separately on the train rows and the test rows before converting each count into an accuracy.',
      datasetId: 'trainTestAccuracy',
      parts: [
        {
          id: 'train-correct',
          prompt: 'How many training examples are classified correctly?',
          correctOptionId: 'train-7',
          options: [
            {
              id: 'train-6',
              title: '6 correct',
              description: 'This misses one of the correctly predicted training rows.',
            },
            {
              id: 'train-7',
              title: '7 correct',
              description: 'Exactly one of the eight training rows is misclassified.',
            },
            {
              id: 'train-8',
              title: '8 correct',
              description: 'That would mean the model made no training mistakes at all.',
            },
          ],
        },
        {
          id: 'train-accuracy',
          prompt: 'What is the training accuracy?',
          correctOptionId: 'train-87-5',
          options: [
            {
              id: 'train-75',
              title: '75%',
              description: 'This corresponds to 6 correct out of 8 training examples.',
            },
            {
              id: 'train-87-5',
              title: '87.5%',
              description: 'This is 7 correct out of 8 training examples.',
            },
            {
              id: 'train-100',
              title: '100%',
              description: 'This would mean every training example is correct.',
            },
          ],
        },
        {
          id: 'test-correct',
          prompt: 'How many test examples are classified correctly?',
          correctOptionId: 'test-2',
          options: [
            {
              id: 'test-1',
              title: '1 correct',
              description: 'This undercounts the correctly predicted test rows.',
            },
            {
              id: 'test-2',
              title: '2 correct',
              description: 'Two of the four test rows match the true label.',
            },
            {
              id: 'test-3',
              title: '3 correct',
              description: 'That would mean only one test mistake, which is not what the table shows.',
            },
          ],
        },
        {
          id: 'test-accuracy',
          prompt: 'What is the test accuracy?',
          correctOptionId: 'test-50',
          options: [
            {
              id: 'test-25',
              title: '25%',
              description: 'This corresponds to 1 correct out of 4 test examples.',
            },
            {
              id: 'test-50',
              title: '50%',
              description: 'This is 2 correct out of 4 test examples.',
            },
            {
              id: 'test-75',
              title: '75%',
              description: 'This corresponds to 3 correct out of 4 test examples.',
            },
          ],
        },
        {
          id: 'future-performance',
          prompt: 'Which accuracy is the better estimate of performance on future unseen data?',
          correctOptionId: 'use-test',
          options: [
            {
              id: 'use-train',
              title: 'Training accuracy',
              description: 'This is measured on the data the model already saw.',
            },
            {
              id: 'use-test',
              title: 'Test accuracy',
              description: 'This is measured on held-out examples and is meant to estimate future generalization.',
            },
            {
              id: 'average-both',
              title: 'Average the two',
              description: 'Averaging them mixes seen and unseen data instead of focusing on held-out performance.',
            },
          ],
        },
      ],
      hintSchedule: [2, 4],
      hints: [
        'Accuracy means correct predictions divided by the number of examples in that split.',
        'Do not combine train and test rows. Compute the two accuracies separately first.',
        'The held-out test split is the one meant to estimate performance on unseen data.',
      ],
      validator: validateMultipleChoiceSelections(
        {
          'train-correct': 'train-7',
          'train-accuracy': 'train-87-5',
          'test-correct': 'test-2',
          'test-accuracy': 'test-50',
          'future-performance': 'use-test',
        },
        'At least one of those counts or accuracies does not match the table yet.',
      ),
      reveal: {
        explanation:
          'The model gets 7 of 8 training examples correct, so training accuracy is 87.5%. It gets 2 of 4 test examples correct, so test accuracy is 50%. The test accuracy is the better estimate of future performance because it is measured on held-out data.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies MultipleChoiceQuestionSpec,
    knnCodeLab,
    knnEvaluationNotebookLab,
  ],
}
