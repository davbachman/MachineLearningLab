import type { DistanceMetric, Vec2 } from '../types'
import { assignPointsToCentroids } from '../lib/kmeansMath'

export interface MultipleChoiceTableDataset {
  id: string
  kind: 'table'
  headers: string[]
  rows: Array<Array<string | number>>
  caption: string
}

export interface MultipleChoicePointCloudScenariosDataset {
  id: string
  kind: 'pointCloudScenarios'
  caption: string
  points: Vec2[]
  scenarios: {
    id: string
    title: string
    centroids: Vec2[]
  }[]
}

export interface MultipleChoiceMetricComparisonDataset {
  id: string
  kind: 'metricComparison'
  caption: string
  points: Vec2[]
  centroids: Vec2[]
  euclideanAssignments: number[]
  manhattanAssignments: number[]
}

export interface MultipleChoiceElbowDataset {
  id: string
  kind: 'elbowChart'
  caption: string
  values: {
    k: number
    objective: number
  }[]
}

export type MultipleChoiceDataset =
  | MultipleChoiceTableDataset
  | MultipleChoicePointCloudScenariosDataset
  | MultipleChoiceMetricComparisonDataset
  | MultipleChoiceElbowDataset

const houseFeatureRows = [
  [210000, 2],
  [285000, 3],
  [360000, 4],
  [435000, 5],
]

const trainTestAccuracyRows: Array<Array<string | number>> = [
  [1, 'train', 'teal', 'teal'],
  [2, 'train', 'teal', 'teal'],
  [3, 'train', 'teal', 'orange'],
  [4, 'train', 'orange', 'orange'],
  [5, 'train', 'orange', 'orange'],
  [6, 'train', 'orange', 'orange'],
  [7, 'train', 'teal', 'teal'],
  [8, 'train', 'orange', 'orange'],
  [9, 'test', 'teal', 'orange'],
  [10, 'test', 'orange', 'orange'],
  [11, 'test', 'teal', 'teal'],
  [12, 'test', 'orange', 'teal'],
]

const initializationPoints: Vec2[] = [
  [-5, 0],
  [-4, 1],
  [-4, -1],
  [4, 4],
  [5, 4],
  [4, 5],
  [4, -4],
  [5, -4],
  [4, -5],
]

const metricCentroids: Vec2[] = [
  [-5, -4],
  [1, 4],
]
const metricPoints: Vec2[] = [
  [-6, -4],
  [-5, -5],
  [-4, -4],
  [-1, 4],
  [1, 3],
  [2, 4],
  [4, -4],
  [5, -4],
]

export const multipleChoiceDatasets: Record<string, MultipleChoiceDataset> = {
  housePrices: {
    id: 'housePrices',
    kind: 'table',
    headers: ['House price ($)', 'Bedrooms'],
    rows: houseFeatureRows,
    caption:
      'Suppose you are trying to predict one house value from nearby house prices and bedroom counts. These are the two centered features you feed into PCA.',
  },
  trainTestAccuracy: {
    id: 'trainTestAccuracy',
    kind: 'table',
    headers: ['ID', 'Split', 'True label', 'Model prediction'],
    rows: trainTestAccuracyRows,
    caption:
      'A classifier has already made the predictions shown here. Use the split column to compute train and test accuracy separately.',
  },
  initSensitivity: {
    id: 'initSensitivity',
    kind: 'pointCloudScenarios',
    caption:
      'Both panels show the same dataset with different starting centroids for k = 3. Compare how well each initialization covers the visible regions.',
    points: initializationPoints,
    scenarios: [
      {
        id: 'A',
        title: 'Initialization A',
        centroids: [
          [-5.9, 1.4],
          [2.9, 3.1],
          [5.7, -2.6],
        ],
      },
      {
        id: 'B',
        title: 'Initialization B',
        centroids: [
          [-5.6, 2.5],
          [2.3, 4.1],
          [4.1, 0.8],
        ],
      },
    ],
  },
  metricComparison: {
    id: 'metricComparison',
    kind: 'metricComparison',
    caption:
      'The panel starts with Euclidean assignments. After the answer is checked or revealed, the Manhattan result appears for comparison.',
    points: metricPoints,
    centroids: metricCentroids,
    euclideanAssignments: assignPointsToCentroids(metricPoints, metricCentroids, 'euclidean'),
    manhattanAssignments: assignPointsToCentroids(metricPoints, metricCentroids, 'manhattan' as DistanceMetric),
  },
  elbowCurve: {
    id: 'elbowCurve',
    kind: 'elbowChart',
    caption:
      'Within-cluster sum of squares drops as k increases. Look for the point where the large gains start to level off.',
    values: [
      { k: 1, objective: 175.4 },
      { k: 2, objective: 36.7 },
      { k: 3, objective: 15.0 },
      { k: 4, objective: 10.0 },
    ],
  },
}
