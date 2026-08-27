import type { Vec2, Vec3 } from '../types'
import {
  add3,
  centerPoints2D,
  centerPoints3D,
  covariance2D,
  cross3,
  normalize3,
  principalDirectionFromPoints2D,
  seededGaussian,
  topTwoPrincipalComponents3D,
} from '../lib/pcaMath'

export interface Pca2DDataset {
  id: string
  label: string
  points: Vec2[]
  scoring: 'covariance' | 'secondMoment'
  answerDirection: Vec2
  comparisonDirection?: Vec2
  comparisonLabel?: string
}

export interface Pca3DDataset {
  id: string
  label: string
  points: Vec3[]
  answerFirst: Vec3
  answerSecond: Vec3
}

export interface HandCalculationDataset {
  id: string
  headers: string[]
  rows: number[][]
  centeredRows: number[][]
  covariance: number[][]
  firstDirection: Vec2
}

function generate2DCloud(options: {
  seed: number
  count: number
  majorScale: number
  minorScale: number
  rotationRadians: number
  translation?: Vec2
}) {
  const gaussian = seededGaussian(options.seed)
  const translation = options.translation ?? [0, 0]
  const cosine = Math.cos(options.rotationRadians)
  const sine = Math.sin(options.rotationRadians)

  return Array.from({ length: options.count }, () => {
    const major = gaussian() * options.majorScale
    const minor = gaussian() * options.minorScale

    return [
      translation[0] + major * cosine - minor * sine,
      translation[1] + major * sine + minor * cosine,
    ] as Vec2
  })
}

function generate3DCloud() {
  const gaussian = seededGaussian(314)
  const first = normalize3([1, 1.2, 0.55])
  const helper = normalize3([-0.72, 0.3, 0.88])
  const second = normalize3([
    helper[0] - first[0] * (first[0] * helper[0] + first[1] * helper[1] + first[2] * helper[2]),
    helper[1] - first[1] * (first[0] * helper[0] + first[1] * helper[1] + first[2] * helper[2]),
    helper[2] - first[2] * (first[0] * helper[0] + first[1] * helper[1] + first[2] * helper[2]),
  ])
  const third = normalize3(cross3(first, second))

  const points = Array.from({ length: 34 }, () => {
    const coordinates = [
      gaussian() * 3.4,
      gaussian() * 1.4,
      gaussian() * 0.45,
    ] as Vec3

    return add3(
      add3(
        [first[0] * coordinates[0], first[1] * coordinates[0], first[2] * coordinates[0]],
        [second[0] * coordinates[1], second[1] * coordinates[1], second[2] * coordinates[1]],
      ),
      [third[0] * coordinates[2], third[1] * coordinates[2], third[2] * coordinates[2]],
    )
  })

  return centerPoints3D(points)
}

const nonCenteredBase = generate2DCloud({
  seed: 7,
  count: 24,
  majorScale: 2.5,
  minorScale: 0.55,
  rotationRadians: 0.97,
  translation: [4.4, -3.6],
})

const centeredBase = centerPoints2D(nonCenteredBase)
const nearCircular = centerPoints2D(
  generate2DCloud({
    seed: 19,
    count: 24,
    majorScale: 1.5,
    minorScale: 1.18,
    rotationRadians: 0.43,
  }),
)

const outlierCore = centerPoints2D(
  generate2DCloud({
    seed: 33,
    count: 22,
    majorScale: 1.45,
    minorScale: 0.4,
    rotationRadians: 1.47,
  }),
)
const outlierPulled = centerPoints2D([...outlierCore, [5.3, 1.1]])

const pca3DPoints = generate3DCloud()
const pca3DAnswer = topTwoPrincipalComponents3D(pca3DPoints)

export const pca2dDatasets: Record<string, Pca2DDataset> = {
  uncentered: {
    id: 'uncentered',
    label: 'Uncentered elongated cloud',
    points: nonCenteredBase,
    scoring: 'secondMoment',
    answerDirection: principalDirectionFromPoints2D(nonCenteredBase, 'secondMoment'),
  },
  centered: {
    id: 'centered',
    label: 'Centered elongated cloud',
    points: centeredBase,
    scoring: 'covariance',
    answerDirection: principalDirectionFromPoints2D(centeredBase, 'covariance'),
  },
  nearCircular: {
    id: 'nearCircular',
    label: 'Centered almost-circular cloud',
    points: nearCircular,
    scoring: 'covariance',
    answerDirection: principalDirectionFromPoints2D(nearCircular, 'covariance'),
  },
  outlierSensitive: {
    id: 'outlierSensitive',
    label: 'Outlier-sensitive centered cloud',
    points: outlierPulled,
    scoring: 'covariance',
    answerDirection: principalDirectionFromPoints2D(outlierPulled, 'covariance'),
    comparisonDirection: principalDirectionFromPoints2D(outlierCore, 'covariance'),
    comparisonLabel: 'PC1 without the outlier',
  },
}

export const pca3dDatasets: Record<string, Pca3DDataset> = {
  centered3d: {
    id: 'centered3d',
    label: 'Centered 3D cloud',
    points: pca3DPoints,
    answerFirst: pca3DAnswer.first,
    answerSecond: pca3DAnswer.second,
  },
}

const handCalculationRows = [
  [0, 1],
  [2, 5],
  [4, 9],
  [6, 13],
]
const handCentered = centerPoints2D(handCalculationRows as Vec2[])

export const handCalculationDatasets: Record<string, HandCalculationDataset> = {
  workedExample: {
    id: 'workedExample',
    headers: ['x1', 'x2'],
    rows: handCalculationRows,
    centeredRows: handCentered,
    covariance: covariance2D(handCalculationRows as Vec2[]),
    firstDirection: [1, 2],
  },
}
