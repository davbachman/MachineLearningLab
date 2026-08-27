import { describe, expect, it } from 'vitest'
import {
  centerPoints2D,
  covariance2D,
  dot3,
  principalDirectionFromPoints2D,
  signInvariantAngle2,
  standardizePoints2D,
  topTwoPrincipalComponents3D,
} from '../lib/pcaMath'
import type { Vec2, Vec3 } from '../types'

describe('pcaMath', () => {
  it('centers 2D points to mean zero', () => {
    const centered = centerPoints2D([
      [2, 4],
      [4, 6],
      [6, 8],
    ] as Vec2[])

    const mean = centered.reduce<Vec2>((accumulator, point) => [accumulator[0] + point[0], accumulator[1] + point[1]], [0, 0])
    expect(mean[0]).toBeCloseTo(0)
    expect(mean[1]).toBeCloseTo(0)
  })

  it('standardizes centered data to unit-scale feature variance', () => {
    const standardized = standardizePoints2D([
      [-2, -200],
      [-1, -100],
      [1, 100],
      [2, 200],
    ] as Vec2[])
    const covariance = covariance2D(standardized)
    expect(covariance[0][0]).toBeCloseTo(1, 6)
    expect(covariance[1][1]).toBeCloseTo(1, 6)
  })

  it('finds the diagonal principal direction for a rank-one cloud', () => {
    const direction = principalDirectionFromPoints2D([
      [-3, -3],
      [-1, -1],
      [1, 1],
      [3, 3],
    ] as Vec2[], 'covariance')

    expect(signInvariantAngle2(direction, [1, 1])).toBeLessThan(1e-6)
  })

  it('returns orthogonal leading components in 3D', () => {
    const points = [
      [4, 2, 1],
      [2, 1, 0.2],
      [-2, -1, -0.3],
      [-4, -2, -1],
      [3, 1.6, 0.4],
    ] as Vec3[]

    const { first, second } = topTwoPrincipalComponents3D(points)
    expect(Math.abs(dot3(first, second))).toBeLessThan(1e-6)
  })
})
