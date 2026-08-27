import type {
  DecisionTreeCounts,
  DecisionTreeRow,
  DecisionTreeSplit,
} from '../data/decisionTreeDatasets'

export interface DecisionTreeNodeEvaluation {
  rows: DecisionTreeRow[]
  counts: DecisionTreeCounts
  gini: number
}

export interface DecisionTreeSplitEvaluation {
  split: DecisionTreeSplit
  parent: DecisionTreeNodeEvaluation
  left: DecisionTreeNodeEvaluation
  right: DecisionTreeNodeEvaluation
  weightedGini: number
  gain: number
}

export function giniFromCounts(counts: DecisionTreeCounts) {
  const total = counts.negative + counts.positive
  if (total === 0) {
    return 0
  }

  const negativeRatio = counts.negative / total
  const positiveRatio = counts.positive / total

  return 1 - negativeRatio ** 2 - positiveRatio ** 2
}

export function countClasses(rows: DecisionTreeRow[]): DecisionTreeCounts {
  return rows.reduce<DecisionTreeCounts>(
    (counts, row) =>
      row.label === 1
        ? { ...counts, positive: counts.positive + 1 }
        : { ...counts, negative: counts.negative + 1 },
    { negative: 0, positive: 0 },
  )
}

export function evaluateNode(rows: DecisionTreeRow[]): DecisionTreeNodeEvaluation {
  const counts = countClasses(rows)

  return {
    rows,
    counts,
    gini: giniFromCounts(counts),
  }
}

export function evaluateSplit(
  rows: DecisionTreeRow[],
  split: DecisionTreeSplit,
): DecisionTreeSplitEvaluation {
  const leftRows = rows.filter((row) => row.features[split.featureId] <= split.threshold)
  const rightRows = rows.filter((row) => row.features[split.featureId] > split.threshold)
  const parent = evaluateNode(rows)
  const left = evaluateNode(leftRows)
  const right = evaluateNode(rightRows)
  const total = rows.length || 1
  const weightedGini = (left.rows.length / total) * left.gini + (right.rows.length / total) * right.gini

  return {
    split,
    parent,
    left,
    right,
    weightedGini,
    gain: parent.gini - weightedGini,
  }
}

export function findBestSplit(
  rows: DecisionTreeRow[],
  candidateSplits: DecisionTreeSplit[],
): DecisionTreeSplitEvaluation {
  const [firstSplit, ...remainingSplits] = candidateSplits
  if (!firstSplit) {
    throw new Error('findBestSplit requires at least one candidate split.')
  }

  return remainingSplits.reduce((best, split) => {
    const next = evaluateSplit(rows, split)
    return next.weightedGini < best.weightedGini ? next : best
  }, evaluateSplit(rows, firstSplit))
}

export function splitLabel(split: DecisionTreeSplit, featureLabel: string) {
  return `${featureLabel} <= ${formatThreshold(split.threshold)}`
}

export function formatThreshold(value: number) {
  return Number.isInteger(value) ? String(value) : value.toFixed(1)
}

export function formatGini(value: number) {
  return value.toFixed(3).replace(/\.?0+$/, '')
}

export function countsLabel(counts: DecisionTreeCounts) {
  return `${counts.positive} passed, ${counts.negative} did not pass`
}
