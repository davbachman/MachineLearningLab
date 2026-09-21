export type Vector2 = [number, number]

export function addScaled(a: Vector2, b: Vector2, scale: number): Vector2 {
  return [a[0] + scale * b[0], a[1] + scale * b[1]]
}

export function normalizeChannels(values: number[]): number[] {
  const mean = values.reduce((a, b) => a + b, 0) / values.length
  const variance = values.reduce((sum, x) => sum + (x - mean) ** 2, 0) / values.length
  return values.map(x => variance === 0 ? 0 : (x - mean) / Math.sqrt(variance))
}

/** Matches the course's threshold-based top-k rule, including boundary ties. */
export function samplingProbabilities(logits: number[], temperature: number, topK: number): number[] {
  if (temperature === 0) return logits.map((_, i) => i === logits.indexOf(Math.max(...logits)) ? 1 : 0)
  const threshold = [...logits].sort((a, b) => b - a)[Math.min(topK, logits.length) - 1]
  const max = Math.max(...logits)
  const weights = logits.map(x => x < threshold ? 0 : Math.exp((x - max) / temperature))
  const sum = weights.reduce((a, b) => a + b, 0)
  return weights.map(x => x / sum)
}

export const attentionPattern = [[1, 0, 0, 0], [.4, .6, 0, 0], [.2, .3, .5, 0], [.1, .2, .3, .4]]
export const attentionValues = [2, -1, 4, 0]
export const dictionaryDirections: Vector2[] = [[1, 0], [0, 1], [.6, .8]]
export const sparseTarget: Vector2 = [1.2, 1.6]

export function reconstructDictionary(coefficients: number[]): Vector2 {
  return dictionaryDirections.reduce<Vector2>((sum, direction, i) => addScaled(sum, direction, coefficients[i]), [0, 0])
}

export function sparseMetrics(coefficients: number[], target: Vector2 = sparseTarget) {
  const reconstruction = reconstructDictionary(coefficients)
  return {
    reconstruction,
    mse: ((target[0] - reconstruction[0]) ** 2 + (target[1] - reconstruction[1]) ** 2) / 2,
    l1: coefficients.reduce((sum, x) => sum + Math.abs(x), 0),
    l0: coefficients.filter(x => x > 1e-4).length,
  }
}

export function activationMetrics(features: number[][], threshold = 1e-4) {
  return {
    meanL0: features.reduce((sum, row) => sum + row.filter(x => x > threshold).length, 0) / features.length,
    deadFraction: features[0].filter((_, column) => features.every(row => row[column] <= threshold)).length / features[0].length,
    meanL1: features.reduce((sum, row) => sum + row.reduce((a, b) => a + Math.abs(b), 0), 0) / features.length,
  }
}

export function steerResidual(residual: Vector2, coefficient: number, direction: Vector2, factor: number): Vector2 {
  return addScaled(residual, direction, (factor - 1) * coefficient)
}

export const steeringTrials = [
  { name: 'Baseline', probability: .25, loss: 1.0 },
  { name: 'Moderate', probability: .45, loss: 1.03 },
  { name: 'Strong', probability: .70, loss: 1.90 },
  { name: 'Matched random', probability: .28, loss: 1.04 },
]
