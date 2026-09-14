import { describe, expect, it } from 'vitest'
import { planeFitQuestion } from '../data/planeFitQuestion'
import { fitLeastSquaresPlane, planeResidualSumOfSquares } from '../lib/linearRegression'

describe('regression plane fitting', () => {
  it('recovers a plane with independent features and nonzero feature means', () => {
    const points: [number, number, number][] = [[1, 2, 1], [2, 2, 3], [1, 3, -2], [3, 4, -1]]
    const fit = fitLeastSquaresPlane(points)
    expect(fit.slope1).toBeCloseTo(2)
    expect(fit.slope2).toBeCloseTo(-3)
    expect(fit.intercept).toBeCloseTo(5)
    expect(fit.rss).toBeCloseTo(0)
    expect(planeResidualSumOfSquares(points, { ...fit, intercept: 6 })).toBeCloseTo(4)
  })

  it('accepts slider-rounded optima and enforces the five-percent RSS tolerance', () => {
    const fit = fitLeastSquaresPlane(planeFitQuestion.points)
    const rounded = { slope1: +fit.slope1.toFixed(2), slope2: +fit.slope2.toFixed(2), intercept: +fit.intercept.toFixed(2) }
    expect(planeFitQuestion.validator(rounded).correct).toBe(true)
    const offset = Math.sqrt(fit.rss * 0.05 / planeFitQuestion.points.length)
    expect(planeFitQuestion.validator({ ...fit, intercept: fit.intercept + offset * 0.99 }).correct).toBe(true)
    expect(planeFitQuestion.validator({ ...fit, intercept: fit.intercept + offset * 1.01 }).correct).toBe(false)
  })

  it('ignores claimed RSS and rejects missing or nonfinite coefficients', () => {
    expect(planeFitQuestion.validator({ ...planeFitQuestion.initialCoefficients, rss: 0 }).correct).toBe(false)
    for (const answer of [null, {}, { slope1: 0, slope2: 0 }, { slope1: NaN, slope2: 0, intercept: 0 }, { slope1: 0, slope2: Infinity, intercept: 0 }]) {
      expect(planeFitQuestion.validator(answer).correct).toBe(false)
    }
  })
})
