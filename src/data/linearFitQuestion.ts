import type { LinearFitQuestionSpec, Vec2 } from '../types'
import { fitLeastSquaresLine, residualSumOfSquares } from '../lib/linearRegression'

const points: Vec2[] = [
  [-4, -1], [-3, 0.8], [-2, -0.4], [-1, 2.7], [0, 2],
  [1, 4.5], [2, 3.4], [3, 6.6], [4, 6],
]
const rssToleranceRatio = 1.05
const minimum = fitLeastSquaresLine(points).rss

export const linearFitQuestion: LinearFitQuestionSpec = {
  id: 'linear-regression-fit-line',
  kind: 'linearFit',
  title: 'Find the Line with the Smallest RSS',
  prompt: 'Adjust the line to minimize the residual sum of squares (RSS).',
  instructions: 'Drag either open handle on the line, or use the slope and intercept sliders. The purple vertical segments show residuals: observed y minus predicted y. RSS is the sum of their squares. A line is accepted when its RSS is within 5% of the minimum possible RSS.',
  points,
  initialSlope: -0.4,
  initialIntercept: 0.5,
  rssToleranceRatio,
  hintSchedule: [2, 4],
  hints: [
    'Start by matching the overall upward trend. Then move the line up or down and watch RSS.',
    'Try small slope adjustments, then small intercept adjustments. Keep changes that lower RSS; the best line need not pass through any particular point.',
  ],
  validator: (submission) => {
    if (!submission || typeof submission !== 'object') return { correct: false }
    const { slope, intercept } = submission as { slope?: unknown; intercept?: unknown }
    if (typeof slope !== 'number' || !Number.isFinite(slope) ||
        typeof intercept !== 'number' || !Number.isFinite(intercept)) {
      return { correct: false, message: 'Choose a finite slope and intercept.' }
    }
    const rss = residualSumOfSquares(points, slope, intercept)
    const correct = rss <= minimum * rssToleranceRatio + 1e-10
    return { correct, message: correct
      ? 'Good work! Your line is within 5% of the minimum RSS.'
      : 'Try to lower RSS further by adjusting the slope and intercept.' }
  },
  reveal: { explanation: 'Least squares minimizes the sum of squared vertical residuals. A few large errors can contribute more than many small ones, so the best line balances errors across all the observations.' },
}
