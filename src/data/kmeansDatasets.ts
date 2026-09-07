import type { DistanceMetric, Vec2 } from '../types'
import {
  assignPointsToCentroids,
  computeCentroidsFromAssignments,
  withinClusterSumOfSquares,
} from '../lib/kmeansMath'

export interface KmeansInteractiveDataset {
  id: string
  label: string
  points: Vec2[]
  k: number
  metric: DistanceMetric
  initialCentroids?: Vec2[]
  fixedCentroids?: Vec2[]
  targetCentroids?: Vec2[]
  targetAssignments?: number[]
  comparisonAssignments?: number[]
  comparisonMetric?: DistanceMetric
  obviousAssignments?: number[]
  displayAssignments?: number[]
  comparisonCentroids?: Vec2[]
  comparisonLabel?: string
  objectiveBefore?: number
  objectiveAfter?: number
  iterationAssignments?: number[][]
  iterationCentroids?: Vec2[][]
  objectiveSequence?: number[]
  regionVisibility?: 'never' | 'resolved' | 'always'
}

const baseClusterPoints: Vec2[] = [
  [-5, -1],
  [-4, 1],
  [-3, 0],
  [-4, -2],
  [3, 3],
  [4, 2],
  [5, 4],
  [4, 5],
]

const baseAssignments = [0, 0, 0, 0, 1, 1, 1, 1]
const baseCentroids = computeCentroidsFromAssignments(baseClusterPoints, baseAssignments, 2)

const outlierPoints: Vec2[] = [...baseClusterPoints, [9, 6]]
const outlierAssignments = assignPointsToCentroids(outlierPoints, baseCentroids, 'euclidean')
const outlierCentroids = computeCentroidsFromAssignments(outlierPoints, outlierAssignments, 2, baseCentroids)

const multiIterationPoints: Vec2[] = [
  [-6, 3],
  [-5, 4],
  [-4, 2],
  [-5, 1],
  [-1, -5],
  [0, -4],
  [1, -3],
  [0, -6],
  [5, 3],
  [6, 2],
  [7, 4],
  [6, 5],
  [-1, 1],
  [2, -1],
  [2, 2],
]

const multiIterationInitialCentroids: Vec2[] = [
  [-7, 2],
  [-2, -6],
  [7, 2],
]

function buildLloydSequence(points: Vec2[], initialCentroids: Vec2[], k: number) {
  const iterationAssignments: number[][] = []
  const iterationCentroids: Vec2[][] = []
  const objectiveSequence: number[] = []
  let currentCentroids = initialCentroids

  for (let iteration = 0; iteration < 8; iteration += 1) {
    const assignments = assignPointsToCentroids(points, currentCentroids, 'euclidean')
    const nextCentroids = computeCentroidsFromAssignments(points, assignments, k, currentCentroids)

    iterationAssignments.push(assignments)
    iterationCentroids.push(nextCentroids)
    objectiveSequence.push(withinClusterSumOfSquares(points, assignments, currentCentroids))

    if (
      nextCentroids.every(
        (centroid, index) =>
          Math.abs(centroid[0] - currentCentroids[index][0]) < 1e-9 &&
          Math.abs(centroid[1] - currentCentroids[index][1]) < 1e-9,
      )
    ) {
      break
    }

    currentCentroids = nextCentroids
  }

  return {
    iterationAssignments,
    iterationCentroids,
    objectiveSequence,
  }
}

const multiIterationSequence = buildLloydSequence(multiIterationPoints, multiIterationInitialCentroids, 3)

const badInitializationPoints: Vec2[] = [
  [-8, 0],
  [-7, 1],
  [-7, -1],
  [-6, 0],
  [-2, 5],
  [-1, 6],
  [0, 5],
  [-1, 4],
  [5, 0],
  [6, 1],
  [6, -1],
  [7, 0],
  [8, 0],
]
const badInitializationObviousAssignments = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]

const metricComparisonCentroids: Vec2[] = [
  [-5, -4],
  [-1, -1],
]
const metricComparisonPoints: Vec2[] = [
  [-8, -4],
  [-7, -5],
  [-6, -4],
  [-2, -1],
  [-1, -2],
  [0, -1],
  [-1, -8],
  [0, -9],
]

export const kmeansInteractiveDatasets: Record<string, KmeansInteractiveDataset> = {
  fullIteration: {
    id: 'fullIteration',
    label: "Run Lloyd's algorithm until it converges",
    points: multiIterationPoints,
    k: 3,
    metric: 'euclidean',
    initialCentroids: multiIterationInitialCentroids,
    iterationAssignments: multiIterationSequence.iterationAssignments,
    iterationCentroids: multiIterationSequence.iterationCentroids,
    targetAssignments:
      multiIterationSequence.iterationAssignments[multiIterationSequence.iterationAssignments.length - 1],
    targetCentroids:
      multiIterationSequence.iterationCentroids[multiIterationSequence.iterationCentroids.length - 1],
    objectiveSequence: multiIterationSequence.objectiveSequence,
    objectiveBefore: multiIterationSequence.objectiveSequence[0],
    objectiveAfter:
      multiIterationSequence.objectiveSequence[multiIterationSequence.objectiveSequence.length - 1],
    regionVisibility: 'resolved',
  },
  badInitialization: {
    id: 'badInitialization',
    label: 'Pick a bad initialization for k = 3',
    points: badInitializationPoints,
    k: 3,
    metric: 'euclidean',
    obviousAssignments: badInitializationObviousAssignments,
    regionVisibility: 'never',
  },
  metricComparison: {
    id: 'metricComparison',
    label: 'Euclidean vs Manhattan assignments',
    points: metricComparisonPoints,
    k: 2,
    metric: 'euclidean',
    fixedCentroids: metricComparisonCentroids,
    targetAssignments: assignPointsToCentroids(metricComparisonPoints, metricComparisonCentroids, 'euclidean'),
    comparisonAssignments: assignPointsToCentroids(metricComparisonPoints, metricComparisonCentroids, 'manhattan'),
    comparisonMetric: 'manhattan',
    regionVisibility: 'never',
  },
  outlierShift: {
    id: 'outlierShift',
    label: 'Outlier sensitivity',
    points: outlierPoints,
    k: 2,
    metric: 'euclidean',
    initialCentroids: baseCentroids,
    displayAssignments: outlierAssignments,
    targetCentroids: outlierCentroids,
    comparisonCentroids: baseCentroids,
    comparisonLabel: 'Centroids without the outlier',
    regionVisibility: 'always',
  },
}
