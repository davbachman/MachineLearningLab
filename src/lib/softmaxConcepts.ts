export const softmaxClasses = ['A', 'B', 'C'] as const
export const baseSoftmaxScores = [1.2, 0.6, -0.4]

export function softmaxProbabilities(scores: number[]) {
  const max = Math.max(...scores)
  const weights = scores.map(score => Math.exp(score - max))
  const sum = weights.reduce((total, weight) => total + weight, 0)
  return weights.map(weight => weight / sum)
}

export function transformedScores(shift: number, spread: number) {
  return baseSoftmaxScores.map(score => spread * score + shift)
}

export const multiclassPredictions = [
  { name: 'P', probabilities: [0.38, 0.34, 0.28] },
  { name: 'Q', probabilities: [0.10, 0.65, 0.25] },
]
export const comparisonPredictions = [
  { name: 'U', probabilities: [0.15, 0.60, 0.25] },
  { name: 'V', probabilities: [0.30, 0.60, 0.10] },
]
export const updateProbabilities = [0.50, 0.30, 0.20]
export const categoricalLoss = (probabilities: number[], trueClass: number) => -Math.log(probabilities[trueClass])
export const scoreGradient = (probabilities: number[], trueClass: number) => probabilities.map((p,i) => p - (i===trueClass ? 1 : 0))
export const boundaryScores = (x: number, y: number) => [x, y, 0]
export const boundaryPoints = [{name:'P', x:1, y:1}, {name:'Q', x:-1, y:-1}]
