import type { Vec2, Vec3 } from '../types'

export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

export function radians(degrees: number) {
  return (degrees * Math.PI) / 180
}

export function degrees(radiansValue: number) {
  return (radiansValue * 180) / Math.PI
}

export function add2(a: Vec2, b: Vec2): Vec2 {
  return [a[0] + b[0], a[1] + b[1]]
}

export function subtract2(a: Vec2, b: Vec2): Vec2 {
  return [a[0] - b[0], a[1] - b[1]]
}

export function scale2(v: Vec2, scalar: number): Vec2 {
  return [v[0] * scalar, v[1] * scalar]
}

export function dot2(a: Vec2, b: Vec2) {
  return a[0] * b[0] + a[1] * b[1]
}

export function norm2(v: Vec2) {
  return Math.hypot(v[0], v[1])
}

export function normalize2(v: Vec2): Vec2 {
  const length = norm2(v)
  if (length === 0) {
    return [1, 0]
  }
  return [v[0] / length, v[1] / length]
}

export function rotate2(v: Vec2, angleRadians: number): Vec2 {
  const cosine = Math.cos(angleRadians)
  const sine = Math.sin(angleRadians)
  return [v[0] * cosine - v[1] * sine, v[0] * sine + v[1] * cosine]
}

export function mean2(points: Vec2[]): Vec2 {
  const total = points.reduce<Vec2>((accumulator, point) => add2(accumulator, point), [0, 0])
  return [total[0] / points.length, total[1] / points.length]
}

export function centerPoints2D(points: Vec2[]) {
  const center = mean2(points)
  return points.map((point) => subtract2(point, center))
}

export function standardizePoints2D(points: Vec2[]) {
  const centered = centerPoints2D(points)
  const scales = centered.reduce<Vec2>(
    (accumulator, point) => [accumulator[0] + point[0] ** 2, accumulator[1] + point[1] ** 2],
    [0, 0],
  )
  const std = [
    Math.sqrt(scales[0] / centered.length) || 1,
    Math.sqrt(scales[1] / centered.length) || 1,
  ] as Vec2

  return centered.map((point) => [point[0] / std[0], point[1] / std[1]] as Vec2)
}

export function secondMoment2D(points: Vec2[]) {
  const matrix = [
    [0, 0],
    [0, 0],
  ]

  for (const [x, y] of points) {
    matrix[0][0] += x * x
    matrix[0][1] += x * y
    matrix[1][0] += x * y
    matrix[1][1] += y * y
  }

  return matrix.map((row) => row.map((value) => value / points.length))
}

export function covariance2D(points: Vec2[]) {
  return secondMoment2D(centerPoints2D(points))
}

export function largestEigenvector2x2(matrix: number[][]): Vec2 {
  const a = matrix[0][0]
  const b = matrix[0][1]
  const d = matrix[1][1]

  if (Math.abs(b) < 1e-12) {
    return a >= d ? [1, 0] : [0, 1]
  }

  const trace = a + d
  const delta = Math.sqrt((a - d) ** 2 + 4 * b * b)
  const lambda = (trace + delta) / 2
  const vector: Vec2 = [b, lambda - a]
  return normalize2(vector)
}

export function principalDirectionFromPoints2D(
  points: Vec2[],
  scoring: 'covariance' | 'secondMoment',
) {
  const matrix = scoring === 'covariance' ? covariance2D(points) : secondMoment2D(points)
  return largestEigenvector2x2(matrix)
}

export function signInvariantAngle2(a: Vec2, b: Vec2) {
  const normalizedA = normalize2(a)
  const normalizedB = normalize2(b)
  const cosine = clamp(Math.abs(dot2(normalizedA, normalizedB)), -1, 1)
  return Math.acos(cosine)
}

export function projectPointOntoLine2D(point: Vec2, direction: Vec2): Vec2 {
  const unitDirection = normalize2(direction)
  const scalar = dot2(point, unitDirection)
  return scale2(unitDirection, scalar)
}

export function projectScalarOntoLine2D(point: Vec2, direction: Vec2) {
  return dot2(point, normalize2(direction))
}

export function explainedVarianceForLine(
  points: Vec2[],
  direction: Vec2,
  scoring: 'covariance' | 'secondMoment',
) {
  const prepared = scoring === 'covariance' ? centerPoints2D(points) : points
  const unitDirection = normalize2(direction)
  const squaredSum = prepared.reduce((accumulator, point) => {
    const scalar = dot2(point, unitDirection)
    return accumulator + scalar * scalar
  }, 0)

  return squaredSum / prepared.length
}

export function maxAbsCoordinate2(points: Vec2[]) {
  return points.reduce((maximum, point) => {
    return Math.max(maximum, Math.abs(point[0]), Math.abs(point[1]))
  }, 0)
}

export function add3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

export function subtract3(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

export function scale3(v: Vec3, scalar: number): Vec3 {
  return [v[0] * scalar, v[1] * scalar, v[2] * scalar]
}

export function dot3(a: Vec3, b: Vec3) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

export function cross3(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

export function norm3(v: Vec3) {
  return Math.hypot(v[0], v[1], v[2])
}

export function normalize3(v: Vec3): Vec3 {
  const length = norm3(v)
  if (length === 0) {
    return [1, 0, 0]
  }
  return [v[0] / length, v[1] / length, v[2] / length]
}

export function mean3(points: Vec3[]): Vec3 {
  const total = points.reduce<Vec3>((accumulator, point) => add3(accumulator, point), [0, 0, 0])
  return [total[0] / points.length, total[1] / points.length, total[2] / points.length]
}

export function centerPoints3D(points: Vec3[]) {
  const center = mean3(points)
  return points.map((point) => subtract3(point, center))
}

export function covariance3D(points: Vec3[]) {
  const centered = centerPoints3D(points)
  const matrix = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ]

  for (const [x, y, z] of centered) {
    matrix[0][0] += x * x
    matrix[0][1] += x * y
    matrix[0][2] += x * z
    matrix[1][0] += x * y
    matrix[1][1] += y * y
    matrix[1][2] += y * z
    matrix[2][0] += x * z
    matrix[2][1] += y * z
    matrix[2][2] += z * z
  }

  return matrix.map((row) => row.map((value) => value / centered.length))
}

function identityMatrix(size: number): number[][] {
  return Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, column) => (row === column ? 1 : 0) as number),
  )
}

export function jacobiEigenDecomposition(matrix: number[][]) {
  const size = matrix.length
  const values = matrix.map((row) => [...row])
  const vectors = identityMatrix(size)

  for (let iteration = 0; iteration < 60; iteration += 1) {
    let p = 0
    let q = 1
    let max = Math.abs(values[p][q])

    for (let row = 0; row < size; row += 1) {
      for (let column = row + 1; column < size; column += 1) {
        const candidate = Math.abs(values[row][column])
        if (candidate > max) {
          max = candidate
          p = row
          q = column
        }
      }
    }

    if (max < 1e-10) {
      break
    }

    const app = values[p][p]
    const aqq = values[q][q]
    const apq = values[p][q]
    const angle = 0.5 * Math.atan2(2 * apq, aqq - app)
    const cosine = Math.cos(angle)
    const sine = Math.sin(angle)

    for (let column = 0; column < size; column += 1) {
      const vip = vectors[column][p]
      const viq = vectors[column][q]
      vectors[column][p] = cosine * vip - sine * viq
      vectors[column][q] = sine * vip + cosine * viq
    }

    for (let column = 0; column < size; column += 1) {
      if (column !== p && column !== q) {
        const aip = values[column][p]
        const aiq = values[column][q]
        values[column][p] = cosine * aip - sine * aiq
        values[p][column] = values[column][p]
        values[column][q] = sine * aip + cosine * aiq
        values[q][column] = values[column][q]
      }
    }

    values[p][p] = cosine * cosine * app - 2 * sine * cosine * apq + sine * sine * aqq
    values[q][q] = sine * sine * app + 2 * sine * cosine * apq + cosine * cosine * aqq
    values[p][q] = 0
    values[q][p] = 0
  }

  const eigenpairs = values.map((row, index) => ({
    value: row[index],
    vector: normalize3([vectors[0][index], vectors[1][index], vectors[2][index]]),
  }))

  eigenpairs.sort((left, right) => right.value - left.value)

  return {
    eigenvalues: eigenpairs.map((pair) => pair.value),
    eigenvectors: eigenpairs.map((pair) => pair.vector),
  }
}

export function topTwoPrincipalComponents3D(points: Vec3[]) {
  const decomposition = jacobiEigenDecomposition(covariance3D(points))
  return {
    first: decomposition.eigenvectors[0],
    second: decomposition.eigenvectors[1],
    eigenvalues: decomposition.eigenvalues,
  }
}

export function projectPointOntoPlaneBasis(point: Vec3, first: Vec3, second: Vec3): Vec2 {
  return [dot3(point, normalize3(first)), dot3(point, normalize3(second))]
}

export function projectPointOntoPlane3D(point: Vec3, first: Vec3, second: Vec3): Vec3 {
  const planeCoordinates = projectPointOntoPlaneBasis(point, first, second)
  return add3(scale3(normalize3(first), planeCoordinates[0]), scale3(normalize3(second), planeCoordinates[1]))
}

export function explainedVarianceForPlane(points: Vec3[], first: Vec3, second: Vec3) {
  const unitFirst = normalize3(first)
  const unitSecond = normalize3(second)
  const squaredSum = centerPoints3D(points).reduce((accumulator, point) => {
    const firstCoordinate = dot3(point, unitFirst)
    const secondCoordinate = dot3(point, unitSecond)
    return accumulator + firstCoordinate * firstCoordinate + secondCoordinate * secondCoordinate
  }, 0)

  return squaredSum / points.length
}

export function directionFromSpherical(azimuth: number, elevation: number): Vec3 {
  const cosineElevation = Math.cos(elevation)
  return normalize3([
    cosineElevation * Math.cos(azimuth),
    Math.sin(elevation),
    cosineElevation * Math.sin(azimuth),
  ])
}

export function sphericalFromDirection(direction: Vec3) {
  const unit = normalize3(direction)
  return {
    azimuth: Math.atan2(unit[2], unit[0]),
    elevation: Math.asin(clamp(unit[1], -1, 1)),
  }
}

export function orthonormalBasisAround(first: Vec3, roll: number) {
  const unitFirst = normalize3(first)
  const fallback = Math.abs(unitFirst[1]) > 0.94 ? ([1, 0, 0] as Vec3) : ([0, 1, 0] as Vec3)
  const baseOne = normalize3(cross3(unitFirst, fallback))
  const baseTwo = normalize3(cross3(unitFirst, baseOne))
  const second = normalize3(add3(scale3(baseOne, Math.cos(roll)), scale3(baseTwo, Math.sin(roll))))
  return { baseOne, baseTwo, second }
}

export function rollForSecondDirection(first: Vec3, second: Vec3) {
  const { baseOne, baseTwo } = orthonormalBasisAround(first, 0)
  return Math.atan2(dot3(normalize3(second), baseTwo), dot3(normalize3(second), baseOne))
}

export function signInvariantAngle3(a: Vec3, b: Vec3) {
  const normalizedA = normalize3(a)
  const normalizedB = normalize3(b)
  const cosine = clamp(Math.abs(dot3(normalizedA, normalizedB)), -1, 1)
  return Math.acos(cosine)
}

export function pointsAlmostEqual(expected: number[], actual: number[], tolerance: number) {
  if (expected.length !== actual.length) {
    return false
  }

  return expected.every((value, index) => Math.abs(value - actual[index]) <= tolerance)
}

export function matrixAlmostEqual(expected: number[][], actual: number[][], tolerance: number) {
  if (expected.length !== actual.length) {
    return false
  }

  return expected.every((row, rowIndex) => pointsAlmostEqual(row, actual[rowIndex] ?? [], tolerance))
}

function mulberry32(seed: number) {
  let current = seed >>> 0
  return () => {
    current += 0x6d2b79f5
    let value = current
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}

export function seededGaussian(seed: number) {
  const random = mulberry32(seed)
  return () => {
    const u = Math.max(random(), 1e-9)
    const v = Math.max(random(), 1e-9)
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }
}

export function formatNumber(value: number) {
  const rounded = Math.abs(value) < 1e-9 ? 0 : value
  return Number.isInteger(rounded) ? `${rounded}` : rounded.toFixed(2)
}
