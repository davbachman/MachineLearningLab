import type {
  RandomForestLabel,
  RandomForestPoint,
  RandomForestSplitCandidate,
  RandomForestStump,
} from '../data/randomForestDatasets'

export interface RandomForestClassCounts {
  negative: number
  positive: number
}

export interface RandomForestStumpEvaluation {
  candidate: RandomForestSplitCandidate
  leftCounts: RandomForestClassCounts
  rightCounts: RandomForestClassCounts
  leftLabel: RandomForestLabel
  rightLabel: RandomForestLabel
  weightedGini: number
}

export interface OobTreePrediction {
  id: string
  predictions: Record<string, RandomForestLabel>
}

export interface OobRowPrediction {
  rowId: string
  votes: RandomForestLabel[]
  prediction: RandomForestLabel | null
}

/**
 * Count bootstrap multiplicities without collapsing repeated draws. Keeping the
 * row IDs in the result makes the value suitable for JSON submission and for a
 * later Gradescope implementation in another language.
 */
export function bootstrapMultiplicities(rowIds: string[], draws: string[]) {
  const counts = Object.fromEntries(rowIds.map((rowId) => [rowId, 0])) as Record<string, number>

  draws.forEach((rowId) => {
    if (!(rowId in counts)) {
      throw new Error(`Unknown bootstrap row ID: ${rowId}`)
    }
    counts[rowId] += 1
  })

  return counts
}

export function outOfBagIds(rowIds: string[], draws: string[]) {
  const selected = new Set(draws)
  return rowIds.filter((rowId) => !selected.has(rowId))
}

export function giniFromBinaryCounts(counts: RandomForestClassCounts) {
  const total = counts.negative + counts.positive
  if (total === 0) {
    return 0
  }

  const negativeRatio = counts.negative / total
  const positiveRatio = counts.positive / total
  return 1 - negativeRatio ** 2 - positiveRatio ** 2
}

export function majorityVote(labels: RandomForestLabel[]): RandomForestLabel {
  const positiveVotes = labels.filter((label) => label === 1).length
  const negativeVotes = labels.length - positiveVotes

  // A fixed class-0 tie break keeps browser and Gradescope results identical.
  return positiveVotes > negativeVotes ? 1 : 0
}

function countLabels(rows: RandomForestPoint[]): RandomForestClassCounts {
  return rows.reduce<RandomForestClassCounts>(
    (counts, row) =>
      row.label === 1
        ? { ...counts, positive: counts.positive + 1 }
        : { ...counts, negative: counts.negative + 1 },
    { negative: 0, positive: 0 },
  )
}

export function expandBootstrapRows(rows: RandomForestPoint[], draws: string[]) {
  const rowsById = new Map(rows.map((row) => [row.id, row]))
  return draws.map((rowId) => {
    const row = rowsById.get(rowId)
    if (!row) {
      throw new Error(`Unknown bootstrap row ID: ${rowId}`)
    }
    return row
  })
}

export function evaluateStump(
  rows: RandomForestPoint[],
  candidate: RandomForestSplitCandidate,
  draws: string[] = rows.map((row) => row.id),
): RandomForestStumpEvaluation {
  const bootstrapRows = expandBootstrapRows(rows, draws)
  const leftRows = bootstrapRows.filter((row) => row[candidate.feature] <= candidate.threshold)
  const rightRows = bootstrapRows.filter((row) => row[candidate.feature] > candidate.threshold)
  const leftCounts = countLabels(leftRows)
  const rightCounts = countLabels(rightRows)
  const total = bootstrapRows.length || 1
  const weightedGini =
    (leftRows.length / total) * giniFromBinaryCounts(leftCounts) +
    (rightRows.length / total) * giniFromBinaryCounts(rightCounts)

  return {
    candidate,
    leftCounts,
    rightCounts,
    leftLabel: majorityVote([
      ...Array.from({ length: leftCounts.negative }, () => 0 as const),
      ...Array.from({ length: leftCounts.positive }, () => 1 as const),
    ]),
    rightLabel: majorityVote([
      ...Array.from({ length: rightCounts.negative }, () => 0 as const),
      ...Array.from({ length: rightCounts.positive }, () => 1 as const),
    ]),
    weightedGini,
  }
}

export function findBestStump(
  rows: RandomForestPoint[],
  candidates: RandomForestSplitCandidate[],
  draws: string[] = rows.map((row) => row.id),
) {
  const [firstCandidate, ...remainingCandidates] = candidates
  if (!firstCandidate) {
    throw new Error('findBestStump requires at least one candidate split.')
  }

  return remainingCandidates.reduce((best, candidate) => {
    const next = evaluateStump(rows, candidate, draws)
    return next.weightedGini < best.weightedGini ? next : best
  }, evaluateStump(rows, firstCandidate, draws))
}

export function trainedStumpFromEvaluation(evaluation: RandomForestStumpEvaluation): RandomForestStump {
  return {
    id: evaluation.candidate.id,
    feature: evaluation.candidate.feature,
    threshold: evaluation.candidate.threshold,
    leftLabel: evaluation.leftLabel,
    rightLabel: evaluation.rightLabel,
  }
}

export function predictStump(
  stump: RandomForestStump,
  point: Pick<RandomForestPoint, 'x' | 'y'>,
): RandomForestLabel {
  return point[stump.feature] <= stump.threshold ? stump.leftLabel : stump.rightLabel
}

export function forestVotes(
  stumps: RandomForestStump[],
  point: Pick<RandomForestPoint, 'x' | 'y'>,
) {
  return stumps.map((stump) => predictStump(stump, point))
}

export function predictForest(
  stumps: RandomForestStump[],
  point: Pick<RandomForestPoint, 'x' | 'y'>,
) {
  return majorityVote(forestVotes(stumps, point))
}

export function aggregateOobPredictions(rowIds: string[], trees: OobTreePrediction[]) {
  return rowIds.map<OobRowPrediction>((rowId) => {
    const votes = trees.flatMap((tree) =>
      tree.predictions[rowId] === undefined ? [] : [tree.predictions[rowId]],
    )
    return {
      rowId,
      votes,
      prediction: votes.length ? majorityVote(votes) : null,
    }
  })
}

export function oobAccuracy(
  rows: Array<Pick<RandomForestPoint, 'id' | 'label'>>,
  trees: OobTreePrediction[],
) {
  const predictions = aggregateOobPredictions(
    rows.map((row) => row.id),
    trees,
  )
  const rowsWithPrediction = predictions.filter(
    (prediction): prediction is OobRowPrediction & { prediction: RandomForestLabel } =>
      prediction.prediction !== null,
  )
  if (!rowsWithPrediction.length) {
    return 0
  }

  const labelsById = new Map(rows.map((row) => [row.id, row.label]))
  const correct = rowsWithPrediction.filter(
    (prediction) => labelsById.get(prediction.rowId) === prediction.prediction,
  ).length
  return correct / rowsWithPrediction.length
}

/**
 * Approximate variance of the average of T exchangeable tree predictions.
 * rho is their pairwise correlation and singleTreeVariance is sigma squared.
 */
export function correlatedEnsembleVariance(
  singleTreeVariance: number,
  correlation: number,
  treeCount: number,
) {
  if (treeCount < 1 || !Number.isInteger(treeCount)) {
    throw new Error('treeCount must be a positive integer.')
  }
  if (correlation < 0 || correlation > 1) {
    throw new Error('correlation must lie between 0 and 1.')
  }

  return (
    correlation * singleTreeVariance +
    ((1 - correlation) * singleTreeVariance) / treeCount
  )
}
