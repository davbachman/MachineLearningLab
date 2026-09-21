// A constant-prediction regression model: every observation is predicted as w.
// Each warm-up uses L = half the mean squared error and a batch-MEAN gradient.
export const batchTargets = [0, 1, 2, 3, 5, 6, 7, 8]
export const batchSizes = [1, 2, 4, 8] as const
export type TeachingBatchSize = typeof batchSizes[number]
export const batchLearningRate = 0.4
export const batchEpochs = 4
export const batchStart = 0
export const batchOrders = [
  [0, 6, 3, 5, 1, 7, 2, 4],
  [7, 2, 5, 0, 4, 1, 6, 3],
  [1, 4, 7, 2, 6, 0, 3, 5],
  [5, 0, 6, 3, 7, 2, 4, 1],
]
export const allBatchRows = batchTargets.map((_, i) => i)
export const batchMean = (rows: number[]) => rows.reduce((sum, row) => sum + batchTargets[row], 0) / rows.length
export const batchLoss = (w: number, rows = allBatchRows) => rows.reduce((sum, row) => sum + (w - batchTargets[row]) ** 2, 0) / (2 * rows.length)
export const batchGradient = (w: number, rows = allBatchRows) => w - batchMean(rows)
export const batchOptimum = batchMean(allBatchRows)

export interface BatchStep {
  step: number
  epoch: number
  rowsProcessed: number
  rows: number[]
  before: number
  w: number
  fullLossBefore: number
  fullLoss: number
  batchLossBefore: number
  batchLossAfter: number
}

export function batchTrajectory(size: TeachingBatchSize, start = batchStart, rate = batchLearningRate, orders = batchOrders): BatchStep[] {
  let w = start
  let rowsProcessed = 0
  const result: BatchStep[] = []
  for (let epoch = 0; epoch < orders.length; epoch++) {
    const order = orders[epoch]
    for (let j = 0; j < order.length; j += size) {
      const rows = order.slice(j, j + size)
      const before = w
      w = before - rate * batchGradient(before, rows)
      rowsProcessed += rows.length
      result.push({ step: result.length + 1, epoch: epoch + 1, rowsProcessed, rows, before, w,
        fullLossBefore: batchLoss(before), fullLoss: batchLoss(w),
        batchLossBefore: batchLoss(before, rows), batchLossAfter: batchLoss(w, rows) })
    }
  }
  return result
}

export function updatesPerEpoch(examples: number, size: number) {
  return Math.ceil(examples / size)
}

export const shuffleSchedules = [
  { name: 'A', rows: [0, 1, 2, 3, 4, 5, 6, 7] },
  { name: 'B', rows: batchOrders[0] },
]
