import type { DistanceMetric, LabeledPoint2D, Vec2 } from '../types'
import {
  applyMinMaxNormalizer,
  fitMinMaxNormalizer,
  normalizeLabeledPoints,
  type PlotBounds,
} from '../lib/knnMath'

export interface KnnClassSpec {
  label: string
  color: string
}

interface KnnBaseDataset {
  id: string
  label: string
  classes: KnnClassSpec[]
  trainingPoints: LabeledPoint2D[]
  bounds: PlotBounds
  boundaryResolution?: number
}

export interface KnnPredictSequenceDataset extends KnnBaseDataset {
  kind: 'predictSequence'
  metric: DistanceMetric
  queryPoint: Vec2
  kSequence: number[]
}

export interface KnnDecisionBoundaryDataset extends KnnBaseDataset {
  kind: 'decisionBoundary'
  metric: DistanceMetric
  gridColumns: number
  gridRows: number
  kSequence: number[]
}

export interface KnnBestKDataset extends KnnBaseDataset {
  kind: 'bestK'
  metric: DistanceMetric
  testPoints: LabeledPoint2D[]
  kValues: number[]
  initialK: number
}

export interface KnnScalingTrapDataset extends KnnBaseDataset {
  kind: 'scalingTrap'
  metric: DistanceMetric
  queryPoint: Vec2
  k: number
  featureLabels: [string, string]
  normalizedTrainingPoints: LabeledPoint2D[]
  normalizedQueryPoint: Vec2
  normalizedBounds: PlotBounds
}

export interface KnnMetricComparisonDataset extends KnnBaseDataset {
  kind: 'metricComparison'
  metric: DistanceMetric
  comparisonMetric: DistanceMetric
  queryPoint: Vec2
  k: number
}

export interface KnnAdversarialPlacementDataset extends KnnBaseDataset {
  kind: 'adversarialPlacement'
  metric: DistanceMetric
  queryPoint: Vec2
  k: number
  examplePoint: LabeledPoint2D
}

export type KnnInteractiveDataset =
  | KnnPredictSequenceDataset
  | KnnDecisionBoundaryDataset
  | KnnBestKDataset
  | KnnScalingTrapDataset
  | KnnMetricComparisonDataset
  | KnnAdversarialPlacementDataset

export const defaultKnnClasses: KnnClassSpec[] = [
  { label: 'Teal', color: '#1c7b8a' },
  { label: 'Orange', color: '#cc7c31' },
]

const scalingTrainingPoints: LabeledPoint2D[] = [
  { point: [4, 0.91], label: 0 },
  { point: [10, 0.84], label: 0 },
  { point: [16, 0.88], label: 0 },
  { point: [24, 0.88], label: 0 },
  { point: [72, 0.12], label: 1 },
  { point: [78, 0.18], label: 1 },
  { point: [85, 0.15], label: 1 },
  { point: [92, 0.2], label: 1 },
]

const scalingQueryPoint: Vec2 = [60, 0.86]
const scalingNormalizer = fitMinMaxNormalizer(scalingTrainingPoints.map((entry) => entry.point))

export const knnInteractiveDatasets: Record<string, KnnInteractiveDataset> = {
  predictSequence: {
    id: 'predictSequence',
    kind: 'predictSequence',
    label: 'One query point, three different choices of k',
    classes: defaultKnnClasses,
    trainingPoints: [
      { point: [-0.5, 0.25], label: 0 },
      { point: [0.2, -0.55], label: 0 },
      { point: [-3, 3], label: 0 },
      { point: [0.65, 0.15], label: 1 },
      { point: [1.0, -0.1], label: 1 },
      { point: [0.1, 1.15], label: 1 },
      { point: [2.3, 2.1], label: 1 },
      { point: [2.8, -2.0], label: 1 },
      { point: [3.2, 0.5], label: 1 },
    ],
    queryPoint: [0, 0],
    metric: 'euclidean',
    kSequence: [3, 5, 9],
    bounds: {
      minX: -4.2,
      maxX: 4.2,
      minY: -3.2,
      maxY: 3.6,
    },
  },
  decisionBoundary: {
    id: 'decisionBoundary',
    kind: 'decisionBoundary',
    label: 'Paint the class of each cell in the grid',
    classes: defaultKnnClasses,
    trainingPoints: [
      { point: [-3.75, 2.25], label: 0 },
      { point: [-2.25, 0.75], label: 0 },
      { point: [-3, -0.75], label: 0 },
      { point: [-2.25, -2.25], label: 0 },
      { point: [-0.9, 2.25], label: 0 },
      { point: [0.45, 0.75], label: 0 },
      { point: [2.25, 2.25], label: 1 },
      { point: [3.75, 0.75], label: 1 },
      { point: [2.25, -0.75], label: 1 },
      { point: [2.25, -2.25], label: 1 },
      { point: [3.75, -2.25], label: 1 },
      { point: [0.75, -0.75], label: 1 },
    ],
    metric: 'euclidean',
    gridColumns: 6,
    gridRows: 6,
    kSequence: [1, 5],
    bounds: {
      minX: -4.5,
      maxX: 4.5,
      minY: -4.5,
      maxY: 4.5,
    },
  },
  bestK: {
    id: 'bestK',
    kind: 'bestK',
    label: 'Use the slider to balance fit and generalization',
    classes: defaultKnnClasses,
    trainingPoints: [
      { point: [-0.32, -1.99], label: 0 },
      { point: [-3.42, -1.91], label: 0 },
      { point: [-2.27, -0.94], label: 0 },
      { point: [-1.24, -0.73], label: 0 },
      { point: [0.21, 3.1], label: 0 },
      { point: [0.99, -0.12], label: 0 },
      { point: [1.05, 1.96], label: 0 },
      { point: [-0.13, 0.29], label: 0 },
      { point: [1.06, 1.85], label: 1 },
      { point: [2.24, -0.21], label: 1 },
      { point: [1.4, 0.66], label: 1 },
      { point: [1.44, 0.77], label: 1 },
      { point: [2.26, -1.08], label: 1 },
      { point: [0.06, -1.71], label: 1 },
      { point: [0.01, -1.88], label: 1 },
      { point: [1.17, -0.61], label: 1 },
      { point: [0.2, 0.1], label: 0 },
      { point: [0.7, 0.2], label: 1 },
    ],
    testPoints: [
      { point: [-1.97, -0.49], label: 0 },
      { point: [-1.59, -1.34], label: 0 },
      { point: [-2.62, -1.21], label: 0 },
      { point: [-0.89, 2.22], label: 0 },
      { point: [-0.79, 3.67], label: 0 },
      { point: [0.47, -0.51], label: 0 },
      { point: [1.65, 2.46], label: 1 },
      { point: [1.32, -0.13], label: 1 },
      { point: [2.02, -0.09], label: 1 },
      { point: [-0.21, -2.56], label: 1 },
      { point: [2.86, -2.74], label: 1 },
      { point: [0.91, -2.12], label: 1 },
      { point: [0.1, 0.25], label: 0 },
      { point: [0.8, -0.1], label: 1 },
    ],
    metric: 'euclidean',
    kValues: Array.from({ length: 11 }, (_, index) => index + 1),
    initialK: 1,
    boundaryResolution: 20,
    bounds: {
      minX: -4.2,
      maxX: 3.4,
      minY: -3.1,
      maxY: 4,
    },
  },
  scalingTrap: {
    id: 'scalingTrap',
    kind: 'scalingTrap',
    label: 'The same points before and after feature normalization',
    classes: defaultKnnClasses,
    trainingPoints: scalingTrainingPoints,
    normalizedTrainingPoints: normalizeLabeledPoints(scalingTrainingPoints, scalingNormalizer),
    queryPoint: scalingQueryPoint,
    normalizedQueryPoint: applyMinMaxNormalizer(scalingQueryPoint, scalingNormalizer),
    metric: 'euclidean',
    k: 3,
    featureLabels: ['Feature 1', 'Feature 2'],
    bounds: {
      minX: 0,
      maxX: 100,
      minY: 0,
      maxY: 1,
    },
    normalizedBounds: {
      minX: -0.05,
      maxX: 1.05,
      minY: -0.05,
      maxY: 1.05,
    },
    boundaryResolution: 20,
  },
  metricComparison: {
    id: 'metricComparison',
    kind: 'metricComparison',
    label: 'Euclidean distance versus Manhattan distance',
    classes: defaultKnnClasses,
    trainingPoints: [
      { point: [1.4, 1.4], label: 0 },
      { point: [1.5, -1.3], label: 0 },
      { point: [-1.35, 1.45], label: 0 },
      { point: [0, 2.2], label: 1 },
      { point: [2.2, 0], label: 1 },
      { point: [-2.2, 0], label: 1 },
      { point: [0, -3], label: 1 },
    ],
    queryPoint: [0, 0],
    metric: 'euclidean',
    comparisonMetric: 'manhattan',
    k: 3,
    bounds: {
      minX: -3.2,
      maxX: 3.2,
      minY: -3.4,
      maxY: 3,
    },
    boundaryResolution: 18,
  },
  adversarialPlacement: {
    id: 'adversarialPlacement',
    kind: 'adversarialPlacement',
    label: 'Add one new point that flips the current vote',
    classes: defaultKnnClasses,
    trainingPoints: [
      { point: [-0.9, 0], label: 0 },
      { point: [0.75, -0.42], label: 0 },
      { point: [-1.8, 1.4], label: 0 },
      { point: [1.9, 1.5], label: 0 },
      { point: [0.62, 0.62], label: 1 },
      { point: [1.8, -1.3], label: 1 },
      { point: [-1.7, -1.5], label: 1 },
    ],
    queryPoint: [0, 0],
    metric: 'euclidean',
    k: 3,
    examplePoint: {
      point: [0.06, 0.12],
      label: 1,
    },
    bounds: {
      minX: -2.6,
      maxX: 2.6,
      minY: -2.2,
      maxY: 2.2,
    },
    boundaryResolution: 18,
  },
}
