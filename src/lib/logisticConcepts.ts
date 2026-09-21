import type { Vec2 } from '../types'

export function sigmoid(score: number): number {
  // Equivalent branches avoid overflow for large negative scores.
  return score >= 0 ? 1 / (1 + Math.exp(-score)) : Math.exp(score) / (1 + Math.exp(score))
}

export function binaryCrossEntropy(probability: number, label: 0 | 1): number {
  return label === 1 ? -Math.log(probability) : -Math.log1p(-probability)
}

export const logisticScoreExamples = [
  { id: 'A', score: -2 }, { id: 'B', score: 0 }, { id: 'C', score: 2 },
]

export const logisticBoundaryPoints: Array<{ id: string; point: Vec2 }> = [
  { id: 'A', point: [0.5, 1.25] },
  { id: 'B', point: [3, 0.75] },
  { id: 'C', point: [1, 1.5] },
  { id: 'D', point: [2.5, 1.5] },
]

export const logisticBoundaryScore = ([x1, x2]: Vec2) => x1 + 2 * x2 - 4

export interface LogisticObservation {
  id: string
  probability: number
  label: 0 | 1
}

// Fixed validation observations: moving the threshold never refits their probabilities.
export const logisticThresholdObservations: LogisticObservation[] = [
  { id: 'A', probability: 0.08, label: 0 },
  { id: 'B', probability: 0.17, label: 0 },
  { id: 'C', probability: 0.28, label: 1 },
  { id: 'D', probability: 0.38, label: 0 },
  { id: 'E', probability: 0.47, label: 1 },
  { id: 'F', probability: 0.54, label: 1 },
  { id: 'G', probability: 0.61, label: 0 },
  { id: 'H', probability: 0.72, label: 1 },
  { id: 'I', probability: 0.84, label: 0 },
  { id: 'J', probability: 0.91, label: 1 },
]

export const logisticThresholdValues = Array.from({ length: 19 }, (_, index) => (index + 1) / 20)
export const logisticMinimumRecall = 0.8

export function isLogisticThreshold(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0.05 && value <= 0.95
    && Math.abs(value * 20 - Math.round(value * 20)) < 1e-8
}

export function logisticThresholdMetrics(threshold: number, observations = logisticThresholdObservations) {
  let truePositives = 0, falsePositives = 0, falseNegatives = 0, trueNegatives = 0
  for (const observation of observations) {
    const predictedPositive = observation.probability > threshold
    if (observation.label === 1) {
      if (predictedPositive) truePositives++
      else falseNegatives++
    } else if (predictedPositive) falsePositives++
    else trueNegatives++
  }
  return {
    truePositives, falsePositives, falseNegatives, trueNegatives,
    precision: truePositives + falsePositives === 0 ? null : truePositives / (truePositives + falsePositives),
    recall: truePositives + falseNegatives === 0 ? null : truePositives / (truePositives + falseNegatives),
  }
}

export const logisticBestFeasiblePrecision = Math.max(...logisticThresholdValues.map(threshold => {
  const metrics = logisticThresholdMetrics(threshold)
  return metrics.recall !== null && metrics.recall >= logisticMinimumRecall ? metrics.precision ?? -1 : -1
}))

export const logisticLossLabels: Array<0 | 1> = [1, 1, 0, 0]
export const logisticLossModels = [
  { id: 'A', probabilities: [0.7, 0.7, 0.4, 0.6] },
  { id: 'B', probabilities: [0.95, 0.95, 0.05, 0.99] },
]
