import type { AssignmentSpec } from '../types'
import { linearFitQuestion } from './linearFitQuestion'
import { planeFitQuestion } from './planeFitQuestion'
import { linearRegressionNotebookLab } from './regressionCodeLabQuestions'
import { polynomialRegressionAssignment } from './polynomialAssignment'
import { gradientDescentAssignment } from './gradientDescentAssignment'
import { batchGradientDescentAssignment } from './batchGradientAssignment'
import { regularizationAssignment } from './regularizationAssignment'
import { logisticRegressionAssignment } from './logisticAssignment'
import { softmaxAssignment } from './softmaxAssignment'

export { polynomialRegressionAssignment, gradientDescentAssignment, batchGradientDescentAssignment,
  regularizationAssignment, logisticRegressionAssignment, softmaxAssignment }

export const linearRegressionAssignment: AssignmentSpec = {
  id: 'linear-regression', version: 2, displayNumber: 4, published: true,
  title: 'Linear Regression', topic: 'Regression',
  description: 'Fit a line and a plane by minimizing RSS, then trace the normal-equation implementation in the notebook lab.',
  questions: [linearFitQuestion, planeFitQuestion, linearRegressionNotebookLab],
}

export const notebookAssignments = [
  linearRegressionAssignment, polynomialRegressionAssignment, gradientDescentAssignment,
  batchGradientDescentAssignment, regularizationAssignment, logisticRegressionAssignment, softmaxAssignment,
]
