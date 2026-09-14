import type { PlaneFitQuestionSpec, Vec3 } from '../types'
import { fitLeastSquaresPlane, planeResidualSumOfSquares } from '../lib/linearRegression'

// Each observation is [x1, x2, y]; the third coordinate is the response.
const points: Vec3[] = [
  [-2.4, -2.4, 1.72], [-2.4, -0.8, -0.32], [-2.4, 0.8, 0.04], [-2.4, 2.4, -2.30],
  [-0.8, -2.4, 1.70], [-0.8, -0.8, 2.16], [-0.8, 0.8, -0.08], [-0.8, 2.4, 0.38],
  [0.8, -2.4, 3.68], [0.8, -0.8, 1.84], [0.8, 0.8, 2.20], [0.8, 2.4, -0.04],
  [2.4, -2.4, 3.76], [2.4, -0.8, 4.12], [2.4, 0.8, 2.08], [2.4, 2.4, 2.44],
]
const rssToleranceRatio = 1.05
const minimum = fitLeastSquaresPlane(points).rss

export const planeFitQuestion: PlaneFitQuestionSpec = {
  id: 'linear-regression-fit-plane',
  kind: 'planeFit',
  title: 'Find the Plane with the Smallest RSS',
  prompt: 'Fit a regression plane y = a·x₁ + b·x₂ + c to the observations by minimizing RSS.',
  instructions: 'Use the two slope sliders to tilt the plane and the intercept slider to raise or lower it. Drag the plot to rotate the view; scroll or pinch to zoom. Purple segments run from each observation to its predicted y on the plane, holding x₁ and x₂ fixed. Your RSS must be within 5% of the minimum possible value.',
  points,
  initialCoefficients: { slope1: -0.3, slope2: 0.4, intercept: 0 },
  rssToleranceRatio,
  hintSchedule: [2, 4, 6],
  hints: [
    'Rotate the view to look along each feature axis. How does y tend to change as x₁ or x₂ increases?',
    'Adjust one slope at a time and keep changes that lower RSS. Then adjust the intercept and repeat.',
    'Residuals are differences in y at fixed feature values. They are not perpendicular distances to the tilted plane.',
  ],
  validator: (submission) => {
    if (!submission || typeof submission !== 'object') return { correct: false }
    const { slope1, slope2, intercept } = submission as { slope1?: unknown; slope2?: unknown; intercept?: unknown }
    if (typeof slope1 !== 'number' || !Number.isFinite(slope1) ||
        typeof slope2 !== 'number' || !Number.isFinite(slope2) ||
        typeof intercept !== 'number' || !Number.isFinite(intercept)) {
      return { correct: false, message: 'Choose finite values for both slopes and the intercept.' }
    }
    const rss = planeResidualSumOfSquares(points, { slope1, slope2, intercept })
    const correct = rss <= minimum * rssToleranceRatio + 1e-10
    return { correct, message: correct
      ? 'Good work! Your plane is within 5% of the minimum RSS.'
      : 'Try to lower RSS further by adjusting both slopes and the intercept.' }
  },
  reveal: { explanation: 'With two features, the regression prediction is a plane. Least squares minimizes squared errors in the response y while holding x₁ and x₂ fixed. This differs from PCA, which minimizes perpendicular distances to a plane.' },
}
