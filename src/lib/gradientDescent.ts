import type { Vec2 } from '../types'

// A fixed, strictly convex teaching loss; coordinates represent model parameters.
export const descentStart: Vec2 = [-2, 1]
export const descentSteps = 20
export const descentTargetLoss = 1.05
export const formatLoss = (loss: number) => loss >= 10000 ? loss.toExponential(2) : loss.toFixed(4)
export const descentLoss = ([a, b]: Vec2) => 1 + (a - 1) ** 2 + 4 * (b + 1) ** 2
export const descentGradient = ([a, b]: Vec2): Vec2 => [2 * (a - 1), 8 * (b + 1)]

export function descentTrajectory(learningRate: number, steps = descentSteps) {
  let point: Vec2 = [...descentStart]
  const trajectory = [{ point, loss: descentLoss(point) }]
  for (let i = 0; i < steps; i++) {
    const gradient = descentGradient(point)
    point = [point[0] - learningRate * gradient[0], point[1] - learningRate * gradient[1]]
    trajectory.push({ point, loss: descentLoss(point) })
  }
  return trajectory
}

export function isDescentLearningRate(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0.01 && value <= 0.30 &&
    Math.abs(value * 100 - Math.round(value * 100)) < 1e-8
}

export const descentRuns = [
  { name: 'A', rate: 0.01 }, { name: 'B', rate: 0.12 }, { name: 'C', rate: 0.255 },
].map(run => ({ ...run, trajectory: descentTrajectory(run.rate) }))

export const singleParameterLoss = (w: number) => (w - 1) ** 2 + 1
export const nonconvexLoss = (w: number) => w ** 4 / 4 - w ** 3 / 3 - w ** 2 + 3
export const nonconvexGradient = (w: number) => w * (w - 2) * (w + 1)

// Equal-length arrows at the start point: to the minimizer, uphill, tangent, downhill.
export const descentArrows = [
  { label: 'A', vector: [3, -2] as Vec2 },
  { label: 'B', vector: [-6, 16] as Vec2 },
  { label: 'C', vector: [16, 6] as Vec2 },
  { label: 'D', vector: [6, -16] as Vec2 },
]
