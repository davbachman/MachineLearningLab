import type { PolynomialDegreeQuestionSpec, Vec2 } from '../types'
import { polynomialExamplePoints } from './polynomialVisualData'

export const trainingPoints = polynomialExamplePoints
export const validationPoints: Vec2[] = [
  [-1.9, 3.627], [-1.5, 2.525], [-1.1, 1.877], [-0.7, 1.263], [-0.3, 1.103],
  [0.1, 0.957], [0.5, 1.255], [0.9, 1.547], [1.3, 2.243], [1.7, 2.983], [1.9, 3.547],
]

// Constant-first coefficients from least squares on trainingPoints ONLY.
// Precomputed with numpy.polynomial.polynomial.polyfit for deterministic browser/grader results.
export const degreeCoefficients = [
  [2.0341351983333333, 0.00914645251324086],
  [1.00010090430511, 0.009146452513241173, 0.6562140704530488],
  [1.0001009043051097, -0.29425233891560754, 0.6562140704530487, 0.10797427570725066],
  [1.2475025929445351, -0.294252338915607, 0.11477324364882238, 0.10797427570725053, 0.1368139558358162],
  [1.2475025929445351, 0.08939146059335949, 0.11477324364882231, -0.29189389166621277, 0.1368139558358162, 0.07946285305662909],
  [1.14800975251582, 0.08939146059335448, 0.5999116358435334, -0.2918938916662133, -0.19691161564259893, 0.07946285305662928, 0.05538512915946206],
  [1.1480097525158173, -0.6316721530043731, 0.599911635843538, 1.2690110892941124, -0.19691161564260096, -0.7342724407301967, 0.05538512915946222, 0.11751795949122412],
  [0.9630266304646826, -0.6316721530043778, 2.3025610607922964, 1.2690110892941138, -2.550583649390112, -0.7342724407301953, 1.0619115058322943, 0.11751795949122372, -0.13048895505438357],
]

export function predictPolynomial(coefficients: number[], x: number) {
  return coefficients.reduceRight((value, coefficient) => value * x + coefficient, 0)
}

export function isPolynomialDegree(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 1 && value <= degreeCoefficients.length
}

export function polynomialFit(degree: number) {
  if (!isPolynomialDegree(degree)) throw new Error('Degree must be an integer from 1 to 8.')
  const coefficients = degreeCoefficients[degree - 1]
  const mse = (points: Vec2[]) => points.reduce((sum, [x, y]) => sum + (y - predictPolynomial(coefficients, x)) ** 2, 0) / points.length
  return { coefficients, trainingMse: mse(trainingPoints), validationMse: mse(validationPoints) }
}

export function bestPolynomialDegree(target: 'training' | 'validation') {
  const key = target === 'training' ? 'trainingMse' : 'validationMse'
  return degreeCoefficients.map((_, i) => i + 1).reduce((best, degree) =>
    polynomialFit(degree)[key] < polynomialFit(best)[key] ? degree : best, 1)
}

export function polynomialDegreeQuestion(target: 'training' | 'validation'): PolynomialDegreeQuestionSpec {
  return {
    id: `polynomial-${target}-degree`, kind: 'polynomialDegree', target, initialDegree: 1,
    title: target === 'training' ? 'Minimize Training Error' : 'Minimize Validation Error',
    prompt: `Adjust the degree slider to find the model with the smallest ${target} MSE among degrees 1–8. Then check your degree.`,
    instructions: 'The training points are the same as in Question 1; held-out validation points are now shown too. Each degree selects a least-squares polynomial fitted using only the training points. Both datasets stay fixed as you move the slider.',
    hints: [`Compare the ${target} readout at every degree. Smaller MSE is better.`, target === 'training' ? 'More flexible polynomials can follow the training points more closely.' : 'A closer fit to training points does not guarantee better predictions on held-out points.'],
    hintSchedule: [2, 4],
    validator: (submission) => {
      const degree = submission && typeof submission === 'object' && 'degree' in submission ? submission.degree : null
      if (!isPolynomialDegree(degree)) return { correct: false, message: 'Choose a degree from 1 to 8.' }
      const correct = degree === bestPolynomialDegree(target)
      return { correct, message: correct ? `Correct! This degree gives the smallest ${target} MSE among the available models.` : `Another degree has lower ${target} MSE. Explore the slider and compare that readout.` }
    },
    reveal: { explanation: target === 'training'
      ? 'Degree 8 minimizes training MSE among these eight models. Extra powers let the model follow more of the training-specific detail; training error alone does not tell us whether that detail will help on unseen data.'
      : 'Degree 2 minimizes validation MSE here. Higher degrees continue to reduce training error but increase validation error: they fit training-specific noise that does not generalize. Validation selects the degree, but the coefficients are still fitted on training data only.' },
  }
}
