import type { Vec2 } from '../types'
import { polynomialExamplePoints } from '../data/polynomialVisualData'

// Every model minimizes training MSE + lambda * sum(w_j^2), j = 1,...,8.
// Features are (x/2)^j; the intercept is unpenalized. Coefficients were obtained
// from (X^T X / n + lambda D)w = X^T y / n, D = diag(0,1,...,1).
export const ridgeStrengths = [0, 0.001, 0.01, 0.1, 1, 10] as const
export const ridgeTrainingPoints: Vec2[] = polynomialExamplePoints
export const ridgeValidationPoints: Vec2[] = [
  [-1.9,3.627],[-1.5,2.525],[-1.1,1.877],[-0.7,1.263],[-0.3,1.103],
  [0.1,0.957],[0.5,1.255],[0.9,1.547],[1.3,2.243],[1.7,2.983],[1.9,3.547],
]
const coefficients = [
  [0.96302663046306713,-1.2633443060110461,9.2102442432290328,10.152088714371413,-40.809338390572655,-23.496718103403317,67.962336373832244,15.04229881489754,-33.405172494215137],
  [1.215606630405172,-0.058179732497133506,1.2121333440585853,-0.74445507215132023,0.25794202280297146,-0.30096392915632547,0.63714946414839124,1.4893218587216472,0.60294705889285949],
  [1.2638806008172858,-0.27116675533218038,0.89364231055378318,-0.16530485601184075,0.64313461549636619,0.18686489201605649,0.58191727049571818,0.58383724327567621,0.52125623565839496],
  [1.4124498902754086,-0.14918028127020488,0.58784630239091606,0.03143414767131416,0.56896774717169585,0.13241985317354385,0.54565034317549233,0.20413785564707607,0.52484422951273746],
  [1.7907537081749174,-0.010047589602517127,0.21603386901239124,0.019027001589281133,0.22534691408282229,0.032642469268771686,0.22270559845990218,0.041295816954597005,0.21888801020899107],
  [1.9995740548173184,0.00040118130221476999,0.030439444016851346,0.0032375316025709771,0.032039765286063004,0.0045402820727648155,0.031780375009971072,0.0053655705465166775,0.031313707801582435],
]

export function isRidgeStrengthIndex(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 && value < ridgeStrengths.length
}

export function ridgePrediction(weights: number[], x: number): number {
  return weights.reduceRight((value, weight) => value * (x / 2) + weight, 0)
}

function mse(weights: number[], points: Vec2[]) {
  return points.reduce((sum, [x,y]) => sum + (ridgePrediction(weights, x) - y) ** 2, 0) / points.length
}

export const ridgeFits = coefficients.map((weights, index) => ({
  strength: ridgeStrengths[index], weights,
  trainingMSE: mse(weights, ridgeTrainingPoints), validationMSE: mse(weights, ridgeValidationPoints),
  squaredWeights: weights.slice(1).reduce((sum, w) => sum + w ** 2, 0),
}))
export const minimumRidgeValidationMSE = Math.min(...ridgeFits.map(fit => fit.validationMSE))
export const bestRidgeStrengthIndex = ridgeFits.findIndex(fit => fit.validationMSE === minimumRidgeValidationMSE)

export const penaltyExamplePoints: Vec2[] = [[-1.5,-2.9],[-0.5,0.7],[0.5,1.3],[1.5,4.9]]
export const penaltyCandidates = [2.4, 1.2, 0].map((slope, i) => {
  const trainingMSE = penaltyExamplePoints.reduce((sum,[x,y]) => sum + (1 + slope*x - y)**2, 0) / penaltyExamplePoints.length
  return { name: ['A','B','C'][i], slope, trainingMSE, penalty: slope**2, objective: trainingMSE+slope**2 }
})

// Exact minimizers for the explicitly stated independent-coordinate example.
export const unregularizedWeights = [2, -1, 0.4]
export function penaltyPath(kind: 'l1' | 'l2', strength: number) {
  return unregularizedWeights.map(w => kind === 'l1'
    ? Math.sign(w) * Math.max(Math.abs(w) - strength, 0)
    : w / (1 + strength))
}
