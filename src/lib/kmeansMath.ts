import type { DistanceMetric, Vec2 } from '../types'

export function squaredEuclideanDistance(point: Vec2, centroid: Vec2) {
  return (point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2
}

export function manhattanDistance(point: Vec2, centroid: Vec2) {
  return Math.abs(point[0] - centroid[0]) + Math.abs(point[1] - centroid[1])
}

export function distanceForMetric(point: Vec2, centroid: Vec2, metric: DistanceMetric) {
  return metric === 'manhattan'
    ? manhattanDistance(point, centroid)
    : squaredEuclideanDistance(point, centroid)
}

export function assignPointsToCentroids(
  points: Vec2[],
  centroids: Vec2[],
  metric: DistanceMetric = 'euclidean',
) {
  return points.map((point) =>
    centroids.reduce(
      (best, centroid, index) => {
        const distance = distanceForMetric(point, centroid, metric)
        return distance < best.distance ? { index, distance } : best
      },
      { index: 0, distance: Number.POSITIVE_INFINITY },
    ).index,
  )
}

export function computeCentroidsFromAssignments(
  points: Vec2[],
  assignments: number[],
  k: number,
  fallbackCentroids?: Vec2[],
) {
  return Array.from({ length: k }, (_, clusterIndex) => {
    const clusterPoints = points.filter((_, pointIndex) => assignments[pointIndex] === clusterIndex)
    if (!clusterPoints.length) {
      return fallbackCentroids?.[clusterIndex] ?? ([0, 0] as Vec2)
    }

    return [
      clusterPoints.reduce((sum, point) => sum + point[0], 0) / clusterPoints.length,
      clusterPoints.reduce((sum, point) => sum + point[1], 0) / clusterPoints.length,
    ] as Vec2
  })
}

export function withinClusterSumOfSquares(points: Vec2[], assignments: number[], centroids: Vec2[]) {
  return points.reduce(
    (sum, point, index) => sum + squaredEuclideanDistance(point, centroids[assignments[index]]),
    0,
  )
}

export function nearestCentroidObjective(points: Vec2[], centroids: Vec2[]) {
  const assignments = assignPointsToCentroids(points, centroids, 'euclidean')
  return withinClusterSumOfSquares(points, assignments, centroids)
}

export function runLloydToConvergence(
  points: Vec2[],
  initialCentroids: Vec2[],
  metric: DistanceMetric = 'euclidean',
  maxIterations = 20,
) {
  let centroids = initialCentroids.map((centroid) => [...centroid] as Vec2)
  let assignments = assignPointsToCentroids(points, centroids, metric)

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const nextCentroids = computeCentroidsFromAssignments(points, assignments, centroids.length, centroids)
    const nextAssignments = assignPointsToCentroids(points, nextCentroids, metric)

    const centroidsUnchanged = nextCentroids.every(
      (centroid, index) =>
        Math.abs(centroid[0] - centroids[index][0]) < 1e-9 &&
        Math.abs(centroid[1] - centroids[index][1]) < 1e-9,
    )
    const assignmentsUnchanged = nextAssignments.every(
      (assignment, index) => assignment === assignments[index],
    )

    centroids = nextCentroids
    assignments = nextAssignments

    if (centroidsUnchanged && assignmentsUnchanged) {
      break
    }
  }

  return {
    centroids,
    assignments,
  }
}

function permute(values: number[]): number[][] {
  if (values.length <= 1) {
    return [values]
  }

  return values.flatMap((value, index) => {
    const rest = values.filter((_, restIndex) => restIndex !== index)
    return permute(rest).map((suffix: number[]) => [value, ...suffix])
  })
}

export function assignmentsMatchUpToPermutation(
  actual: number[],
  expected: number[],
  k: number,
) {
  const permutations = permute(Array.from({ length: k }, (_, index) => index))
  return permutations.some((mapping) =>
    actual.every((assignment, index) => mapping[assignment] === expected[index]),
  )
}

export function centroidsMatchUpToPermutation(
  actual: Vec2[],
  expected: Vec2[],
  tolerance: number,
) {
  const permutations = permute(Array.from({ length: expected.length }, (_, index) => index))

  return permutations.some((mapping) =>
    actual.every((centroid, index) => {
      const target = expected[mapping[index]]
      return (
        Math.abs(centroid[0] - target[0]) <= tolerance &&
        Math.abs(centroid[1] - target[1]) <= tolerance
      )
    }),
  )
}

export function formatObjective(value: number) {
  return value < 100 ? value.toFixed(2) : value.toFixed(1)
}
