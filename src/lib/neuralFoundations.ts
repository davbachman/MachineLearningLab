export const rectifier = (x: number) => Math.max(0, x)
export const hingeValue = (x: number, threshold: number) => rectifier(x - threshold)
export const twoBranchValue = (x: number, activated: boolean) => activated ? rectifier(x) + rectifier(-x) : x + (-x)

export function affineRows(input: number[][], kernel: number[][], bias: number[]) {
  return input.map(row => bias.map((b, j) => row.reduce((sum, x, i) => sum + x * kernel[i][j], b)))
}

export function sharedBranchTrace(a: number, b: number) {
  return { product: a * b, value: a * b + a, productPath: b, directPath: 1, da: b + 1, db: a }
}
export const logRoot = (u: number) => Math.log(Math.sqrt(u))
export const logRootDerivative = (u: number) => 1 / (2 * u)
export const denseParameterCount = (widths: number[]) => widths.slice(1).reduce((sum, width, i) => sum + widths[i] * width + width, 0)

export function squaredErrorGrid(predictions: number[], targets: number[]) {
  return predictions.map(p => targets.map(t => (p - t) ** 2))
}
export function pairedMse(predictions: number[], targets: number[]) {
  return predictions.reduce((sum, p, i) => sum + (p - targets[i]) ** 2, 0) / predictions.length
}
export function allPairsMse(predictions: number[], targets: number[]) {
  const grid = squaredErrorGrid(predictions, targets)
  return grid.flat().reduce((a, b) => a + b, 0) / (predictions.length * targets.length)
}

export function exponentialMoments(gradients: number[], beta: number) {
  let m = 0
  return gradients.map(g => { m = beta * m + (1 - beta) * g; return m })
}
export function batchGroups(order: number[], size: number) {
  if (!Number.isInteger(size) || size < 1) throw new Error('Batch size must be a positive integer')
  return Array.from({ length: Math.ceil(order.length / size) }, (_, i) => order.slice(i * size, (i + 1) * size))
}
export function dropoutValues(values: number[], keep: boolean[], rate: number, training: boolean) {
  return training ? values.map((v, i) => keep[i] ? v / (1 - rate) : 0) : [...values]
}

/** Notebook 15 calls this Conv; its kernel is intentionally not flipped. */
export function validCorrelation(image: number[][], kernel: number[][]) {
  const rows = image.length - kernel.length + 1
  const cols = image[0].length - kernel[0].length + 1
  return Array.from({ length: rows }, (_, r) => Array.from({ length: cols }, (_, c) =>
    kernel.reduce((sum, row, kr) => sum + row.reduce((s, value, kc) => s + value * image[r + kr][c + kc], 0), 0),
  ))
}
export function maxPool(image: number[][], pool: [number, number]) {
  return Array.from({ length: Math.floor(image.length / pool[0]) }, (_, r) =>
    Array.from({ length: Math.floor(image[0].length / pool[1]) }, (_, c) =>
      Math.max(...image.slice(r * pool[0], (r + 1) * pool[0]).flatMap(row => row.slice(c * pool[1], (c + 1) * pool[1]))),
    ),
  )
}
export const contrastImage = [[1, 2, 0, 1], [3, 4, 1, 0], [0, 2, 5, 1], [1, 0, 2, 3]]
export const contrastKernel = [[-1, 1], [-1, 1]]
export const poolingImage = [[1, 5, 2, 4, 99], [3, 0, 6, 1, 99], [99, 99, 99, 99, 99]]
