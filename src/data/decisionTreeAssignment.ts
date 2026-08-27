import type { AssignmentSpec, DecisionTreeQuestionSpec, ValidationResult } from '../types'
import {
  decisionTreeDatasets,
  type DecisionTreeBestSplitDataset,
  type DecisionTreeDepthTwoDataset,
  type DecisionTreeGiniWarmupDataset,
  type DecisionTreeSplitScoreDataset,
  type DecisionTreeVisualDepthTwoDataset,
  type DecisionTreeVisualSplitDataset,
} from './decisionTreeDatasets'
import { evaluateSplit, findBestSplit, giniFromCounts } from '../lib/decisionTreeMath'
import { decisionTreeCodeLab } from './codeLabQuestions'

const NUMERIC_TOLERANCE = 0.015
const SCORE_TIE_TOLERANCE = 1e-9

function almostEqual(actual: number | undefined, expected: number, tolerance = NUMERIC_TOLERANCE) {
  return typeof actual === 'number' && Math.abs(actual - expected) <= tolerance
}

function validateGiniWarmup(datasetId: keyof typeof decisionTreeDatasets) {
  const dataset = decisionTreeDatasets[datasetId] as DecisionTreeGiniWarmupDataset
  const answers = Object.fromEntries(
    dataset.nodes.map((node) => [node.id, giniFromCounts(node.counts)]),
  ) as Record<string, number>

  return (submission: unknown): ValidationResult => {
    const payload = submission as { values?: Record<string, number | undefined> }
    const values = payload.values ?? {}
    const correct = Object.entries(answers).every(([nodeId, answer]) =>
      almostEqual(values[nodeId], answer),
    )

    return {
      correct,
      message: correct
        ? 'Correct! Good Job!'
        : 'At least one node impurity does not match 1 - p(pass)^2 - p(not pass)^2.',
    }
  }
}

function validateSplitScore(datasetId: keyof typeof decisionTreeDatasets) {
  const dataset = decisionTreeDatasets[datasetId] as DecisionTreeSplitScoreDataset
  const splitScore = evaluateSplit(dataset.rows, dataset.split)
  const expectedLeftRowIds = splitScore.left.rows.map((row) => row.id).sort()

  const sameRowIds = (rowIds: string[] | undefined) => {
    const submitted = [...(rowIds ?? [])].sort()
    return (
      submitted.length === expectedLeftRowIds.length &&
      submitted.every((rowId, index) => rowId === expectedLeftRowIds[index])
    )
  }

  return (submission: unknown): ValidationResult => {
    const payload = submission as {
      step?: 'partition' | 'childStats' | 'weightedGini'
      leftRowIds?: string[]
      counts?: {
        left?: { negative?: number; positive?: number }
        right?: { negative?: number; positive?: number }
      }
      values?: { left?: number; right?: number }
      value?: number
    }

    const partitionCorrect = sameRowIds(payload.leftRowIds)
    if (payload.step === 'partition') {
      return {
        correct: partitionCorrect,
        message: partitionCorrect
          ? 'Correct! Good Job!'
          : 'Check the <= rule carefully and select every row sent to the left child.',
      }
    }

    const countsCorrect =
      payload.counts?.left?.negative === splitScore.left.counts.negative &&
      payload.counts?.left?.positive === splitScore.left.counts.positive &&
      payload.counts?.right?.negative === splitScore.right.counts.negative &&
      payload.counts?.right?.positive === splitScore.right.counts.positive
    const childGiniCorrect =
        almostEqual(payload.values?.left, splitScore.left.gini) &&
        almostEqual(payload.values?.right, splitScore.right.gini)

    if (payload.step === 'childStats') {
      const correct = partitionCorrect && countsCorrect && childGiniCorrect
      return {
        correct,
        message: correct
          ? 'Correct! Good Job!'
          : 'Use your row partition to recount both outcomes in each child, then compute each Gini.',
      }
    }

    const correct =
      partitionCorrect &&
      countsCorrect &&
      childGiniCorrect &&
      almostEqual(payload.value, splitScore.weightedGini)
    return {
      correct,
      message: correct
        ? 'Correct! Good Job!'
        : 'Weight each child Gini by the fraction of rows that went to that child.',
    }
  }
}

function validateBestSplit(datasetId: keyof typeof decisionTreeDatasets) {
  const dataset = decisionTreeDatasets[datasetId] as DecisionTreeBestSplitDataset
  const best = findBestSplit(dataset.rows, dataset.candidateSplits)

  return (submission: unknown): ValidationResult => {
    const payload = submission as { splitId?: string }
    const correct = payload.splitId === best.split.id

    return {
      correct,
      message: correct
        ? 'Correct! Good Job!'
        : 'A different candidate has a lower weighted Gini after the split.',
    }
  }
}

function validateVisualSplit(datasetId: keyof typeof decisionTreeDatasets) {
  const dataset = decisionTreeDatasets[datasetId] as DecisionTreeVisualSplitDataset
  const best = findBestSplit(dataset.rows, dataset.candidateSplits)

  return (submission: unknown): ValidationResult => {
    const payload = submission as { featureId?: string; threshold?: number }
    if (!payload.featureId || typeof payload.threshold !== 'number') {
      return {
        correct: false,
        message: 'Choose a split direction and threshold before checking.',
      }
    }

    const result = evaluateSplit(dataset.rows, {
      featureId: payload.featureId,
      threshold: payload.threshold,
    })
    const correct = result.weightedGini <= best.weightedGini + SCORE_TIE_TOLERANCE

    return {
      correct,
      message: correct
        ? 'Correct! Good Job!'
        : 'That line still mixes the two classes more than the best available split.',
    }
  }
}

function validateDepthTwo(datasetId: keyof typeof decisionTreeDatasets) {
  const dataset = decisionTreeDatasets[datasetId] as
    | DecisionTreeDepthTwoDataset
    | DecisionTreeVisualDepthTwoDataset
  const root = findBestSplit(dataset.rows, dataset.rootCandidates)
  const child = findBestSplit(root.right.rows, dataset.rightChildCandidates)

  return (submission: unknown): ValidationResult => {
    const payload = submission as { rootSplitId?: string; rightChildSplitId?: string }
    const correct =
      payload.rootSplitId === root.split.id && payload.rightChildSplitId === child.split.id

    return {
      correct,
      message: correct
        ? 'Correct! Good Job!'
        : 'Choose the lowest-Gini root first, then rescore the child candidates using only rows in the right branch.',
    }
  }
}

export const decisionTreeAssignment: AssignmentSpec = {
  id: 'decision-trees',
  version: 7,
  title: 'Decision Trees',
  topic: 'Classification',
  description:
    'Practice binary decision-tree splits by computing Gini impurity, weighted split scores, and first tree-building choices.',
  published: false,
  questions: [
    {
      id: 'dt-gini-warmup',
      kind: 'decisionTree',
      title: 'Compute Node Gini',
      prompt:
        'Compute the Gini impurity for each node using only the pass and not-pass counts.',
      instructions:
        'Use Gini = 1 - p(pass)^2 - p(not pass)^2. A pure node has Gini 0; a perfectly balanced binary node has Gini 0.5.',
      datasetId: 'giniWarmup',
      interactionMode: 'giniWarmup',
      hintSchedule: [2, 4],
      hints: [
        'Convert the counts to proportions before squaring.',
        'The two mixed 3-to-1 nodes have the same impurity, even though the majority class is different.',
      ],
      validator: validateGiniWarmup('giniWarmup'),
      reveal: {
        explanation:
          'Gini measures how mixed the labels are in a node. Pure nodes have impurity 0, and a 50-50 binary split has impurity 0.5.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies DecisionTreeQuestionSpec,
    {
      id: 'dt-score-one-split',
      kind: 'decisionTree',
      title: 'Score One Split',
      prompt:
        'Trace the proposed threshold through the raw data: route the rows, count both outcomes in each child, and compute the child and weighted Gini values.',
      instructions:
        'Complete all three stages. Rows with study hours less than or equal to the threshold go left; the rest go right.',
      datasetId: 'splitScore',
      interactionMode: 'splitScore',
      hintSchedule: [2, 4],
      hints: [
        'Start with the threshold comparison only. Do not use the outcome column until every row has a side.',
        'The weighted score is not the simple average unless both children contain the same number of rows.',
      ],
      validator: validateSplitScore('splitScore'),
      reveal: {
        explanation:
          'A split is scored by the weighted impurity of its children. Lower weighted Gini means the split has separated the labels more cleanly.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies DecisionTreeQuestionSpec,
    {
      id: 'dt-best-threshold',
      kind: 'decisionTree',
      title: 'Find the Best Threshold',
      prompt:
        'Compare the candidate thresholds for study hours and choose the split with the lowest weighted Gini.',
      datasetId: 'bestThreshold',
      interactionMode: 'bestThreshold',
      hintSchedule: [2, 4],
      hints: [
        'Do not just look for the purest child. Both child sizes matter in the weighted score.',
        'The best threshold makes the left child pure while keeping the right child mostly pass labels.',
      ],
      validator: validateBestSplit('bestThreshold'),
      reveal: {
        explanation:
          'Decision trees greedily choose the candidate split with the lowest weighted child impurity at the current node.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies DecisionTreeQuestionSpec,
    {
      id: 'dt-best-root-split',
      kind: 'decisionTree',
      title: 'Choose the Root Split',
      prompt:
        'Compare split candidates across two features and choose the best root split for the tree.',
      datasetId: 'bestRootSplit',
      interactionMode: 'bestRootSplit',
      hintSchedule: [2, 4],
      hints: [
        'Score each candidate independently; the feature names do not matter except through the labels they separate.',
        'The strongest candidates are close. Keep the child-size weights instead of comparing only the purer child.',
      ],
      validator: validateBestSplit('bestRootSplit'),
      reveal: {
        explanation:
          'The best root split is the candidate with the lowest weighted Gini across all features being considered.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies DecisionTreeQuestionSpec,
    {
      id: 'dt-visual-axis-split',
      kind: 'decisionTree',
      title: 'Place a Visual Split',
      prompt:
        'Place the vertical or horizontal split with the lowest weighted Gini. No split is perfect, and the visually widest gap is not necessarily best.',
      instructions:
        'The line represents x <= threshold or y <= threshold. Split scores remain hidden until you commit an answer.',
      datasetId: 'visualSplit',
      interactionMode: 'visualSplit',
      hintSchedule: [2, 4],
      hints: [
        'For each promising line, count both colors on each side and weight each child impurity by its number of points.',
        'A small pure child can outweigh a more balanced-looking division of the plot.',
      ],
      validator: validateVisualSplit('visualSplit'),
      reveal: {
        explanation:
          'The best vertical cut sits just to the right of x = 2. It isolates three did-not-pass points in a pure child; every more central cut leaves enough mixing to score worse.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies DecisionTreeQuestionSpec,
    {
      id: 'dt-visual-depth-two',
      kind: 'decisionTree',
      title: 'Build a Tree from Geometry',
      prompt:
        'Use the scatterplot to choose the greedy root split, then choose the best split for the root\'s right child.',
      instructions:
        'Choose the root first. The second line is scored using only points with x greater than the chosen root threshold; scores stay hidden until you submit.',
      datasetId: 'visualDepthTwo',
      interactionMode: 'depthTwo',
      hintSchedule: [2, 4],
      hints: [
        'The two strongest root candidates are close: calculate their weighted impurities rather than choosing the line that looks most central.',
        'For the second split, completely ignore every point to the left of the root line.',
      ],
      validator: validateDepthTwo('visualDepthTwo'),
      reveal: {
        explanation:
          'Greedy tree building scores the root by itself. It then restricts attention to the selected branch and scores the next candidates on that subset only.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies DecisionTreeQuestionSpec,
    {
      id: 'dt-depth-two-tree',
      kind: 'decisionTree',
      title: 'Complete a Depth-2 Tree',
      prompt:
        'Choose the best root split, then choose the best split for the impure right child.',
      instructions:
        'Only the right child of the root needs a second split in this dataset.',
      datasetId: 'depthTwo',
      interactionMode: 'depthTwo',
      hintSchedule: [2, 4],
      hints: [
        'After the best root split, the left child is already pure.',
        'For the right child, ignore rows that went left at the root and score the child candidates only on the remaining rows.',
      ],
      validator: validateDepthTwo('depthTwo'),
      reveal: {
        explanation:
          'Tree building is recursive: after choosing the root split, each impure child is treated as its own smaller split-selection problem.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies DecisionTreeQuestionSpec,
    decisionTreeCodeLab,
  ],
}
