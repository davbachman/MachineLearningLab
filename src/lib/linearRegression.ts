import type { Vec2, Vec3 } from '../types'

export interface PlaneCoefficients {
  slope1: number
  slope2: number
  intercept: number
}

export function planeResidualSumOfSquares(points: Vec3[], coefficients: PlaneCoefficients) {
  const { slope1, slope2, intercept } = coefficients
  return points.reduce((sum, [x1, x2, y]) => sum + (y - (slope1 * x1 + slope2 * x2 + intercept)) ** 2, 0)
}

export function fitLeastSquaresPlane(points: Vec3[]): PlaneCoefficients & { rss: number } {
  const means = [0, 1, 2].map((axis) => points.reduce((sum, point) => sum + point[axis], 0) / points.length)
  let s11 = 0, s12 = 0, s22 = 0, t1 = 0, t2 = 0
  for (const [x1, x2, y] of points) {
    const u = x1 - means[0], v = x2 - means[1], w = y - means[2]
    s11 += u * u
    s12 += u * v
    s22 += v * v
    t1 += u * w
    t2 += v * w
  }
  const determinant = s11 * s22 - s12 * s12
  if (Math.abs(determinant) < 1e-12) throw new Error('Plane fitting needs two independent features.')
  const slope1 = (t1 * s22 - t2 * s12) / determinant
  const slope2 = (t2 * s11 - t1 * s12) / determinant
  const intercept = means[2] - slope1 * means[0] - slope2 * means[1]
  const coefficients = { slope1, slope2, intercept }
  return { ...coefficients, rss: planeResidualSumOfSquares(points, coefficients) }
}

export function residualSumOfSquares(points: Vec2[], slope: number, intercept: number) {
  return points.reduce((sum, [x, y]) => sum + (y - (slope * x + intercept)) ** 2, 0)
}

export function fitLeastSquaresLine(points: Vec2[]) {
  const meanX = points.reduce((sum, [x]) => sum + x, 0) / points.length
  const meanY = points.reduce((sum, [, y]) => sum + y, 0) / points.length
  const xx = points.reduce((sum, [x]) => sum + (x - meanX) ** 2, 0)
  const xy = points.reduce((sum, [x, y]) => sum + (x - meanX) * (y - meanY), 0)
  const slope = xy / xx
  const intercept = meanY - slope * meanX
  return { slope, intercept, rss: residualSumOfSquares(points, slope, intercept) }
}
