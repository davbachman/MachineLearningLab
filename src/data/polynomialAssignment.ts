import type { AssignmentSpec, MultipleChoiceQuestionSpec } from '../types'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import { polynomialRegressionNotebookLab, polynomialEvaluationNotebookLab } from './regressionCodeLabQuestions'

function question(input: Omit<MultipleChoiceQuestionSpec, 'kind' | 'validator' | 'hintSchedule'>): MultipleChoiceQuestionSpec {
  return { ...input, kind: 'multipleChoice', hintSchedule: [2, 4],
    validator: validateMultipleChoiceSelections(Object.fromEntries(input.parts.map(p => [p.id, p.correctOptionId])), 'Check each selection against the displayed data and the notebook implementation.') }
}

export const polynomialRegressionAssignment: AssignmentSpec = {
  id: 'polynomial-regression', version: 2, displayNumber: 5, published: true,
  title: 'Polynomial Regression and Overfitting', topic: 'Model Flexibility and Generalization',
  description: 'Build polynomial features with from-scratch classes, compare training and validation errors, and trace the combined notebook one calculation at a time.',
  questions: [
    question({
      id: 'polynomial-curves', title: 'Compare the Fitted Curves',
      prompt: 'These three models use degrees 1, 2, and 8, in an unknown order.',
      instructions: 'Compare the overall shape and the deviations from the training observations. All three models were fitted by least squares on the same data.',
      datasetId: 'polynomialCurves',
      parts: [{ id: 'line', prompt: 'Which model must have degree 1?', correctOptionId: 'B', options: [
        { id: 'A', title: 'Model A', description: '' }, { id: 'B', title: 'Model B', description: '' }, { id: 'C', title: 'Model C', description: '' },
      ] }, { id: 'generalization', prompt: 'Model C follows the training observations most closely. What can you conclude from these plots alone?', correctOptionId: 'need-validation', options: [
        { id: 'need-validation', title: 'It fits these training observations well; held-out errors are needed to assess whether it generalizes better.', description: '' },
        { id: 'best', title: 'It must make the best predictions on unseen observations.', description: '' },
        { id: 'always', title: 'Every curve with visible wiggles must be overfitting.', description: '' },
      ] }],
      hints: ['Degree 1 means a straight line in the original input.', 'Training fit and prediction on unseen observations answer different questions.'],
      reveal: { explanation: 'B is degree 1, A is degree 2, and C is degree 8. The more flexible model can follow training-specific detail. The next error-curve problem supplies the held-out evidence needed to judge generalization.' },
    }),
    question({
      id: 'polynomial-powers', title: 'Turn One Feature into Several',
      prompt: 'Construct polynomial features from a one-dimensional array of scaled inputs.',
      instructions: 'Use the from-scratch PolynomialFeatures class in 5PolynomialRegression.ipynb. Observations remain rows.',
      datasetId: 'polynomialFeatureTrace',
      parts: [{ id: 'row', prompt: 'What is the first row when degree=3 and include_bias=False?', correctOptionId: 'powers', options: [
        { id: 'powers', title: '[-2, 4, -8]', description: '' },
        { id: 'bias', title: '[1, -2, 4]', description: '' },
        { id: 'cube', title: '[-8]', description: '' },
      ] }, { id: 'shape', prompt: 'For all four observations, what is the feature-matrix shape when degree=3 and include_bias=True?', correctOptionId: 'four-four', options: [
        { id: 'four-four', title: '(4, 4)', description: '' },
        { id: 'four-three', title: '(4, 3)', description: '' },
        { id: 'three-four', title: '(3, 4)', description: '' },
      ] }],
      hints: ['List the powers starting at one when bias is absent.', 'Including bias adds the power-zero column.'],
      reveal: { explanation: 'For -2, powers one through three are -2, 4, and -8. With bias, four rows each contain powers zero through three, giving shape (4, 4).' },
    }),
    question({
      id: 'polynomial-normal-equation', title: 'Same Solver, Different Features',
      prompt: 'The fitted function can be curved even though we still use LinearRegression from Homework 4.',
      instructions: 'The custom model adds a column of ones internally. It does not have a fit_intercept argument.',
      datasetId: 'polynomialModelReference',
      parts: [{ id: 'linear', prompt: 'In what sense is this still a linear model?', correctOptionId: 'coefficients', options: [
        { id: 'coefficients', title: 'Predictions are linear in the fitted coefficients, not necessarily in x.', description: '' },
        { id: 'straight', title: 'Predictions must form a straight line in x.', description: '' },
        { id: 'loss', title: 'The squared-error loss has been replaced with a linear loss.', description: '' },
      ] }, { id: 'bias', prompt: 'Why does the notebook use include_bias=False when fitting this custom LinearRegression?', correctOptionId: 'duplicate', options: [
        { id: 'duplicate', title: 'A bias feature would duplicate the ones column already added inside fit.', description: '' },
        { id: 'no-offset', title: 'The model must have intercept zero.', description: '' },
        { id: 'overfit-proof', title: 'Removing the bias guarantees that high-degree models cannot overfit.', description: '' },
      ] }],
      hints: ['Treat x and x² as two already-computed input columns.', 'Two identical columns are linearly dependent.'],
      reveal: { explanation: 'Polynomial regression is least squares on engineered powers. Adding the constant feature twice makes Xnew.T @ Xnew singular; the custom model already supplies one intercept column.' },
    }),
    question({
      id: 'polynomial-validation-curve', title: 'Read the Generalization Curve',
      prompt: 'Compare training and validation MSE as polynomial degree increases.',
      instructions: 'These rounded values come from the combined notebook with random_state=42. All degrees use the same split and training-fitted scaler. The table supplies precise values for close points.',
      datasetId: 'polynomialErrors',
      parts: [{ id: 'degree', prompt: 'Which degree would you choose by minimum validation MSE?', correctOptionId: 'four', options: [
        { id: 'four', title: '4', description: '' }, { id: 'eight', title: '8', description: '' }, { id: 'one', title: '1', description: '' },
      ] }, { id: 'evidence', prompt: 'What happens when degree rises from 4 to 5?', correctOptionId: 'overfit', options: [
        { id: 'overfit', title: 'Training MSE decreases, but validation MSE increases: added flexibility hurts generalization here.', description: '' },
        { id: 'both-better', title: 'Both errors decrease, so generalization improves.', description: '' },
        { id: 'always-bad', title: 'This proves every degree-5 polynomial overfits every dataset.', description: '' },
      ] }],
      hints: ['Choose using the validation curve, not the training curve.', 'Compare the same two degrees in both columns.'],
      reveal: { explanation: 'Degree 4 has the smallest validation MSE. From 4 to 5, training MSE falls from about 9.68 to 7.52 while validation MSE rises from about 24.33 to 61.86. This is evidence of overfitting in this experiment, not a universal rule about degree.' },
    }),
    question({
      id: 'polynomial-evaluation', title: 'Keep Evaluation Honest',
      prompt: 'Distinguish fitting, degree selection, and final evaluation.',
      instructions: 'The notebook reserves 20% for validation and fits the scaler and every model on the training set only.',
      datasetId: 'polynomialEvaluationReference',
      parts: [{ id: 'scaler', prompt: 'How should validation inputs be scaled?', correctOptionId: 'training', options: [
        { id: 'training', title: 'Use the mean and standard deviation learned from Xtrain.', description: '' },
        { id: 'validation', title: 'Fit a new scaler on Xval.', description: '' },
        { id: 'together', title: 'Refit the scaler on the combined training and validation inputs.', description: '' },
      ] }, { id: 'final-test', prompt: 'After choosing degree using validation MSE, what is needed for a final independent performance estimate?', correctOptionId: 'fresh', options: [
        { id: 'fresh', title: 'A separate test set that was not used to choose the degree.', description: '' },
        { id: 'minimum', title: 'Report the smallest validation MSE as an independent test score.', description: '' },
        { id: 'train', title: 'Report the selected model’s training MSE.', description: '' },
      ] }],
      hints: ['Prediction must use the coordinate system learned during fitting.', 'The choice of degree has already used information from the validation set.'],
      reveal: { explanation: 'Fit preprocessing on training data, then reuse it unchanged. Validation data guide degree selection, so a separate untouched test set is needed for final independent evaluation.' },
    }),
    polynomialRegressionNotebookLab,
    polynomialEvaluationNotebookLab,
  ],
}
