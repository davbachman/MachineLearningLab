import type { AssignmentSpec, RandomForestQuestionSpec, ValidationResult } from '../types'
import {
  randomForestDatasets,
  type RandomForestBootstrapDataset,
  type RandomForestFeatureGeometryDataset,
  type RandomForestLabel,
  type RandomForestOobDataset,
  type RandomForestVarianceDataset,
  type RandomForestVoteGeometryDataset,
} from './randomForestDatasets'
import {
  aggregateOobPredictions,
  bootstrapMultiplicities,
  correlatedEnsembleVariance,
  findBestStump,
  oobAccuracy,
  outOfBagIds,
  predictStump,
  trainedStumpFromEvaluation,
  predictForest,
} from '../lib/randomForestMath'
import { randomForestCodeLab } from './codeLabQuestions'

const NUMERIC_TOLERANCE = 0.0015

function sameIds(actual: unknown, expected: string[]) {
  return (
    Array.isArray(actual) &&
    actual.every((value) => typeof value === 'string') &&
    [...actual].sort().join('|') === [...expected].sort().join('|')
  )
}

function sameLabelRecord(
  actual: Record<string, RandomForestLabel | undefined> | undefined,
  expected: Record<string, RandomForestLabel>,
) {
  return Object.entries(expected).every(([id, label]) => actual?.[id] === label)
}

function validateBootstrapAudit(datasetId: keyof typeof randomForestDatasets) {
  const dataset = randomForestDatasets[datasetId] as RandomForestBootstrapDataset
  const rowIds = dataset.rows.map((row) => row.id)
  const expectedCounts = bootstrapMultiplicities(rowIds, dataset.draws)
  const expectedOobIds = outOfBagIds(rowIds, dataset.draws)

  return (submission: unknown): ValidationResult => {
    const payload = submission as {
      multiplicities?: Record<string, number | undefined>
      oobIds?: string[]
    }
    const countsCorrect = Object.entries(expectedCounts).every(
      ([rowId, count]) => payload.multiplicities?.[rowId] === count,
    )
    const oobCorrect = sameIds(payload.oobIds, expectedOobIds)

    return {
      correct: countsCorrect && oobCorrect,
      message:
        countsCorrect && oobCorrect
          ? 'Correct! Good Job!'
          : 'Count every repeated draw, then mark exactly the rows whose count is zero.',
    }
  }
}

function expectedFeatureGeometryAnswers(dataset: RandomForestFeatureGeometryDataset) {
  const selectedSplits: Record<string, string> = {}
  const trainedTrees = new Map(
    dataset.trees.map((tree) => {
      const best = findBestStump(dataset.rows, tree.candidates, tree.draws)
      selectedSplits[tree.id] = best.candidate.id
      return [tree.id, trainedStumpFromEvaluation(best)]
    }),
  )
  const predictions = Object.fromEntries(
    dataset.probes.map((probe) => [
      probe.id,
      Object.fromEntries(
        dataset.trees.map((tree) => [
          tree.id,
          predictStump(trainedTrees.get(tree.id)!, probe),
        ]),
      ),
    ]),
  ) as Record<string, Record<string, RandomForestLabel>>

  return { selectedSplits, predictions }
}

function validateFeatureGeometry(datasetId: keyof typeof randomForestDatasets) {
  const dataset = randomForestDatasets[datasetId] as RandomForestFeatureGeometryDataset
  const expected = expectedFeatureGeometryAnswers(dataset)

  return (submission: unknown): ValidationResult => {
    const payload = submission as {
      selectedSplits?: Record<string, string | undefined>
      predictions?: Record<string, Record<string, RandomForestLabel | undefined>>
    }
    const splitsCorrect = Object.entries(expected.selectedSplits).every(
      ([treeId, splitId]) => payload.selectedSplits?.[treeId] === splitId,
    )
    const predictionsCorrect = Object.entries(expected.predictions).every(([probeId, labels]) =>
      sameLabelRecord(payload.predictions?.[probeId], labels),
    )

    return {
      correct: splitsCorrect && predictionsCorrect,
      message:
        splitsCorrect && predictionsCorrect
          ? 'Correct! Good Job!'
          : 'Score each tree on its own repeated bootstrap rows and only its allowed feature, then trace both probe points.',
    }
  }
}

function validateForestGeometry(datasetId: keyof typeof randomForestDatasets) {
  const dataset = randomForestDatasets[datasetId] as RandomForestVoteGeometryDataset
  const expected = Object.fromEntries(
    dataset.probes.map((probe) => [probe.id, predictForest(dataset.trees, probe)]),
  ) as Record<string, RandomForestLabel>

  return (submission: unknown): ValidationResult => {
    const payload = submission as {
      predictions?: Record<string, RandomForestLabel | undefined>
    }
    const correct = sameLabelRecord(payload.predictions, expected)

    return {
      correct,
      message: correct
        ? 'Correct! Good Job!'
        : 'Trace all five tree rules for each probe and use the majority, not the nearest training point.',
    }
  }
}

function expectedOobAnswers(dataset: RandomForestOobDataset) {
  const aggregated = aggregateOobPredictions(
    dataset.rows.map((row) => row.id),
    dataset.trees,
  )
  return {
    predictions: Object.fromEntries(
      aggregated.map((entry) => [entry.rowId, entry.prediction]),
    ) as Record<string, RandomForestLabel>,
    accuracy: oobAccuracy(dataset.rows, dataset.trees),
  }
}

function validateOobEstimate(datasetId: keyof typeof randomForestDatasets) {
  const dataset = randomForestDatasets[datasetId] as RandomForestOobDataset
  const expected = expectedOobAnswers(dataset)

  return (submission: unknown): ValidationResult => {
    const payload = submission as {
      predictions?: Record<string, RandomForestLabel | undefined>
      accuracy?: number
    }
    const predictionsCorrect = sameLabelRecord(payload.predictions, expected.predictions)
    const accuracyCorrect =
      typeof payload.accuracy === 'number' &&
      Math.abs(payload.accuracy - expected.accuracy) <= NUMERIC_TOLERANCE
    const correct = predictionsCorrect && accuracyCorrect

    return {
      correct,
      message: correct
        ? 'Correct! Good Job!'
        : 'For each row, vote using only nonblank OOB cells, then compare those eight votes with the true labels.',
    }
  }
}

function expectedVarianceAnswers(dataset: RandomForestVarianceDataset) {
  const variances = Object.fromEntries(
    dataset.scenarios.map((scenario) => [
      scenario.id,
      correlatedEnsembleVariance(
        dataset.singleTreeVariance,
        scenario.correlation,
        scenario.treeCount,
      ),
    ]),
  ) as Record<string, number>
  const bestScenarioId = dataset.scenarios.reduce((best, scenario) =>
    variances[scenario.id] < variances[best.id] ? scenario : best,
  ).id

  return { variances, bestScenarioId, interventionId: 'feature-subsampling' }
}

function validateVariance(datasetId: keyof typeof randomForestDatasets) {
  const dataset = randomForestDatasets[datasetId] as RandomForestVarianceDataset
  const expected = expectedVarianceAnswers(dataset)

  return (submission: unknown): ValidationResult => {
    const payload = submission as {
      variances?: Record<string, number | undefined>
      bestScenarioId?: string
      interventionId?: string
    }
    const variancesCorrect = Object.entries(expected.variances).every(
      ([scenarioId, variance]) =>
        typeof payload.variances?.[scenarioId] === 'number' &&
        Math.abs(payload.variances[scenarioId]! - variance) <= NUMERIC_TOLERANCE,
    )
    const correct =
      variancesCorrect &&
      payload.bestScenarioId === expected.bestScenarioId &&
      payload.interventionId === expected.interventionId

    return {
      correct,
      message: correct
        ? 'Correct! Good Job!'
        : 'The correlation term does not vanish as more trees are added; calculate all three values before choosing.',
    }
  }
}

export const randomForestAssignment: AssignmentSpec = {
  id: 'random-forests',
  version: 6,
  title: 'Random Forests',
  topic: 'Ensemble Learning',
  description:
    'Build a forest from bootstrapped, decorrelated trees; aggregate their predictions; and estimate how well the ensemble generalizes.',
  published: false,
  questions: [
    {
      id: 'rf-bootstrap-oob',
      kind: 'randomForest',
      title: 'Audit a Bootstrap Sample',
      prompt:
        'Record how many times every training row appears in the draw, then identify the out-of-bag rows.',
      instructions:
        'A bootstrap sample contains n draws with replacement. Repeated rows count repeatedly; rows with multiplicity zero are OOB for this tree.',
      datasetId: 'bootstrapAudit',
      interactionMode: 'bootstrapAudit',
      hintSchedule: [2, 4],
      hints: [
        'Tally the draw from left to right rather than treating it as a set.',
        'Exactly three row IDs never appear in this sample.',
      ],
      validator: validateBootstrapAudit('bootstrapAudit'),
      reveal: {
        explanation:
          'Bootstrap sampling changes row weights: duplicates influence the fitted tree multiple times, while omitted rows provide out-of-bag evaluation cases.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies RandomForestQuestionSpec,
    {
      id: 'rf-feature-subsampling-geometry',
      kind: 'randomForest',
      title: 'Grow Two Decorrelated Stumps',
      prompt:
        'Each tree gets a different bootstrap sample and is allowed to split on only one randomly selected feature. Choose its lowest-Gini threshold, then trace both probes through both fitted stumps.',
      instructions:
        'Repeated bootstrap IDs count repeatedly. A leaf predicts its majority class; break an exact class tie toward Class 0.',
      datasetId: 'featureSubsamplingGeometry',
      interactionMode: 'featureSubsamplingGeometry',
      hintSchedule: [2, 4],
      hints: [
        'Tree A can draw only a vertical line and Tree B only a horizontal line; score each on its own listed draws.',
        'After choosing a split, recount the bootstrap labels on each side to obtain the two leaf predictions.',
      ],
      validator: validateFeatureGeometry('featureSubsamplingGeometry'),
      reveal: {
        explanation:
          'Random feature subsets force otherwise strong trees to consider different split directions. Their errors become less correlated, which makes averaging more effective.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies RandomForestQuestionSpec,
    {
      id: 'rf-geometric-vote',
      kind: 'randomForest',
      title: 'Map a Forest by Majority Vote',
      prompt:
        'Classify each labeled probe by tracing all five axis-aligned trees and taking their majority vote.',
      instructions:
        'Click a probe in the plot or use the answer buttons. Some trees reverse the usual leaf labels, so the nearest training point is not a reliable shortcut.',
      datasetId: 'forestVoteGeometry',
      interactionMode: 'forestVoteGeometry',
      hintSchedule: [2, 4],
      hints: [
        'Make a five-entry vote tally for one probe at a time.',
        'T4 predicts Class 1 below its horizontal line, while T5 predicts Class 1 to the left of its vertical line.',
      ],
      validator: validateForestGeometry('forestVoteGeometry'),
      reveal: {
        explanation:
          'A forest can form a nontrivial, piecewise axis-aligned boundary even when every constituent model is only a one-split tree.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies RandomForestQuestionSpec,
    {
      id: 'rf-oob-estimate',
      kind: 'randomForest',
      title: 'Compute an OOB Estimate',
      prompt:
        'For each training row, aggregate only the trees for which that row was out of bag. Then compute OOB accuracy.',
      instructions:
        'A blank cell means the tree trained on that row, so including that tree would leak training information into the estimate.',
      datasetId: 'oobEstimate',
      interactionMode: 'oobEstimate',
      hintSchedule: [2, 4],
      hints: [
        'Each row has exactly three nonblank OOB votes.',
        'After voting, compare the eight aggregate predictions against the label column; two are wrong.',
      ],
      validator: validateOobEstimate('oobEstimate'),
      reveal: {
        explanation:
          'OOB prediction evaluates each row only with trees that did not train on it, producing a validation-like estimate without a separate holdout set.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies RandomForestQuestionSpec,
    {
      id: 'rf-variance-correlation',
      kind: 'randomForest',
      title: 'Balance Forest Size and Correlation',
      prompt:
        'Compute the approximate variance of each averaged forest, select the lowest-variance scenario, and choose the intervention that directly reduces tree correlation.',
      instructions:
        'Use Var(average) = rho * sigma² + (1 - rho) * sigma² / T with single-tree variance sigma² = 0.24. Enter decimals to three places.',
      datasetId: 'varianceReduction',
      interactionMode: 'varianceReduction',
      hintSchedule: [2, 4],
      hints: [
        'The rho * sigma² term remains even as T becomes very large.',
        'One scenario with only 16 trees beats the 100-tree forest because its trees are much less correlated.',
      ],
      validator: validateVariance('varianceReduction'),
      reveal: {
        explanation:
          'Adding trees reduces the independent part of variance, but correlated errors create a floor. Feature subsampling targets that correlation directly.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies RandomForestQuestionSpec,
    randomForestCodeLab,
  ],
}
