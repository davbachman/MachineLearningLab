/** Small deterministic experiments; none of these panels train a model. */
export const edgeImage = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
export const edgeKernel = [[-1, 1], [-1, 1]]
export function filterAt(image: number[][], kernel: number[][], row: number, column: number) {
  return kernel.reduce((sum, values, i) => sum + values.reduce((s, v, j) => s + v * image[row + i][column + j], 0), 0)
}
export const poolingImage = [[1, 3, 2, 0], [4, 2, 5, 1], [0, 1, 2, 2], [3, 1, 0, 6]]
export function poolTwo(image: number[][], mode: 'max' | 'mean') {
  return [0, 2].map(row => [0, 2].map(column => {
    const values = [image[row][column], image[row][column + 1], image[row + 1][column], image[row + 1][column + 1]]
    return mode === 'max' ? Math.max(...values) : values.reduce((a, b) => a + b, 0) / 4
  }))
}
export function convSize(input: number, kernel: number, padding = 0, stride = 1) {
  return Math.floor((input + 2 * padding - kernel) / stride) + 1
}
export function transposeConvSize(input: number, kernel: number, padding: number, stride: number, outputPadding: number) {
  return (input - 1) * stride - 2 * padding + kernel + outputPadding
}
export const cleanPixels = [0, .2, .8, 1]
export const noisyPixels = [0, .2, 0, 1]
export const restoredPixels = [0, .3, .6, .9]
export function mse(prediction: number[], target: number[]) {
  return prediction.reduce((sum, value, i) => sum + (value - target[i]) ** 2, 0) / target.length
}
export function pairCompression(pixels: number[]) {
  const code = [(pixels[0] + pixels[1]) / 2, (pixels[2] + pixels[3]) / 2]
  return { code, reconstruction: [code[0], code[0], code[1], code[1]] }
}
export function simpleTokens(text: string) {
  return text.match(/[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]/g) ?? []
}
export const windowStream = ['<bos>', 'Ava', 'found', 'a', 'blue', 'kite', '.', '<eos>']
export function shiftedWindow(start: number, length = 4) {
  return { inputs: windowStream.slice(start, start + length), targets: windowStream.slice(start + 1, start + length + 1) }
}
export function normalizedWeights(scores: number[], allowed?: boolean[]) {
  const active = scores.filter((_, index) => allowed?.[index] ?? true)
  if (!active.length) throw new Error('At least one key must be allowed.')
  const maximum = Math.max(...active)
  const exponentials = scores.map((value, index) => (allowed?.[index] ?? true) ? Math.exp(value - maximum) : 0)
  const total = exponentials.reduce((a, b) => a + b, 0)
  return exponentials.map(value => value / total)
}
export const memoryKeys = [[1, 0], [0, 1], [1, 1]]
export const memoryValues = [2, 8, 5]
export function retrieve(query: number[], values = memoryValues) {
  const scores = memoryKeys.map(key => key.reduce((s, v, i) => s + v * query[i], 0) / Math.sqrt(2))
  const weights = normalizedWeights(scores)
  return { scores, weights, result: weights.reduce((s, w, i) => s + w * values[i], 0) }
}
export const matrixQueries = [[1, 0], [0, 1], [1, 1]]
export const attentionMatrix = matrixQueries.map(query => retrieve(query).weights)
export const causalScores = [[2, 1, 0, -1], [0, 2, 1, 0], [1, 0, 2, 1], [0, 1, 0, 2]]
export function causalRow(row: number, applyMask = true) {
  return normalizedWeights(causalScores[row], causalScores[row].map((_, column) => !applyMask || column <= row))
}
export function headShapes(batch: number, sequence: number, channels: number, heads: number) {
  if (channels % heads) throw new Error('Channels must divide evenly between heads.')
  return { headSize: channels / heads, split: [batch, heads, sequence, channels / heads], weights: [batch, heads, sequence, sequence], output: [batch, sequence, channels] }
}
