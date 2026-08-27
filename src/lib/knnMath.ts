import type { DistanceMetric, LabeledPoint2D, Vec2 } from '../types'

export interface KnnNeighbor {
  index: number
  label: number
  point: Vec2
  distance: number
}

export interface KnnPrediction {
  label: number
  neighbors: KnnNeighbor[]
}

export interface PlotBounds {
  minX: number
  maxX: number
  minY: number
  maxY: number
}

export interface MinMaxNormalizer {
  min: Vec2
  max: Vec2
}

export function distanceBetween(a: Vec2, b: Vec2, metric: DistanceMetric) {
  const dx = Math.abs(a[0] - b[0])
  const dy = Math.abs(a[1] - b[1])
  return metric === 'manhattan' ? dx + dy : Math.hypot(dx, dy)
}

export function classifyKnnPoint(
  trainingPoints: LabeledPoint2D[],
  queryPoint: Vec2,
  k: number,
  metric: DistanceMetric,
): KnnPrediction {
  const orderedNeighbors = trainingPoints
    .map<KnnNeighbor>((entry, index) => ({
      index,
      label: entry.label,
      point: entry.point,
      distance: distanceBetween(entry.point, queryPoint, metric),
    }))
    .sort((left, right) => left.distance - right.distance || left.index - right.index)

  const effectiveK = Math.max(1, Math.min(k, orderedNeighbors.length))
  const neighbors = orderedNeighbors.slice(0, effectiveK)
  const voteTable = new Map<number, { count: number; totalDistance: number }>()

  for (const neighbor of neighbors) {
    const current = voteTable.get(neighbor.label) ?? { count: 0, totalDistance: 0 }
    voteTable.set(neighbor.label, {
      count: current.count + 1,
      totalDistance: current.totalDistance + neighbor.distance,
    })
  }

  let bestLabel = neighbors[0]?.label ?? 0
  let bestCount = -1
  let bestDistance = Number.POSITIVE_INFINITY

  for (const [label, summary] of voteTable.entries()) {
    if (
      summary.count > bestCount ||
      (summary.count === bestCount && summary.totalDistance < bestDistance - 1e-9) ||
      (summary.count === bestCount &&
        Math.abs(summary.totalDistance - bestDistance) <= 1e-9 &&
        label < bestLabel)
    ) {
      bestLabel = label
      bestCount = summary.count
      bestDistance = summary.totalDistance
    }
  }

  return {
    label: bestLabel,
    neighbors,
  }
}

export function computeAccuracy(
  trainingPoints: LabeledPoint2D[],
  evaluationPoints: LabeledPoint2D[],
  k: number,
  metric: DistanceMetric,
) {
  const correctCount = evaluationPoints.reduce((total, point) => {
    const prediction = classifyKnnPoint(trainingPoints, point.point, k, metric)
    return total + (prediction.label === point.label ? 1 : 0)
  }, 0)

  return evaluationPoints.length > 0 ? correctCount / evaluationPoints.length : 0
}

export function findBestK(
  trainingPoints: LabeledPoint2D[],
  evaluationPoints: LabeledPoint2D[],
  kValues: number[],
  metric: DistanceMetric,
) {
  return kValues.reduce(
    (best, currentK) => {
      const accuracy = computeAccuracy(trainingPoints, evaluationPoints, currentK, metric)
      if (
        accuracy > best.accuracy + 1e-9 ||
        (Math.abs(accuracy - best.accuracy) <= 1e-9 && currentK < best.k)
      ) {
        return {
          k: currentK,
          accuracy,
        }
      }

      return best
    },
    {
      k: kValues[0] ?? 1,
      accuracy: Number.NEGATIVE_INFINITY,
    },
  )
}

export function createGridCenters(bounds: PlotBounds, columns: number, rows: number) {
  return Array.from({ length: columns * rows }, (_, cellIndex) => {
    const columnIndex = cellIndex % columns
    const rowIndex = Math.floor(cellIndex / columns)
    const x0 = bounds.minX + (columnIndex / columns) * (bounds.maxX - bounds.minX)
    const x1 = bounds.minX + ((columnIndex + 1) / columns) * (bounds.maxX - bounds.minX)
    const y0 = bounds.minY + (rowIndex / rows) * (bounds.maxY - bounds.minY)
    const y1 = bounds.minY + ((rowIndex + 1) / rows) * (bounds.maxY - bounds.minY)

    return [(x0 + x1) / 2, (y0 + y1) / 2] as Vec2
  })
}

export function classifyGridCells(
  trainingPoints: LabeledPoint2D[],
  bounds: PlotBounds,
  columns: number,
  rows: number,
  k: number,
  metric: DistanceMetric,
) {
  return createGridCenters(bounds, columns, rows).map(
    (center) => classifyKnnPoint(trainingPoints, center, k, metric).label,
  )
}

export function fitMinMaxNormalizer(points: Vec2[]): MinMaxNormalizer {
  const xs = points.map((point) => point[0])
  const ys = points.map((point) => point[1])

  return {
    min: [Math.min(...xs), Math.min(...ys)],
    max: [Math.max(...xs), Math.max(...ys)],
  }
}

export function applyMinMaxNormalizer(point: Vec2, normalizer: MinMaxNormalizer): Vec2 {
  const [minX, minY] = normalizer.min
  const [maxX, maxY] = normalizer.max

  return [
    (point[0] - minX) / Math.max(maxX - minX, 1e-9),
    (point[1] - minY) / Math.max(maxY - minY, 1e-9),
  ]
}

export function normalizeLabeledPoints(
  points: LabeledPoint2D[],
  normalizer: MinMaxNormalizer,
): LabeledPoint2D[] {
  return points.map((entry) => ({
    ...entry,
    point: applyMinMaxNormalizer(entry.point, normalizer),
  }))
}
