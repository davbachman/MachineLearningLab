import type { AssignmentSpec, MultipleChoicePart, MultipleChoiceQuestionSpec } from '../../src/types'
import { validateMultipleChoiceSelections } from '../../src/lib/questionValidation'
import { gradientDescentNotebookLab } from '../../src/data/regressionCodeLabQuestions'

// Archived answer key: keep previously downloaded submissions gradeable.
type Choice = [id: string, title: string, description: string]

function part(
  id: string,
  prompt: string,
  correctOptionId: string,
  choices: Choice[],
): MultipleChoicePart {
  return {
    id,
    prompt,
    correctOptionId,
    options: choices.map(([optionId, title, description]) => ({
      id: optionId,
      title,
      description,
    })),
  }
}

function notebookQuestion(input: {
  id: string
  title: string
  prompt: string
  instructions: string
  datasetId: string
  parts: MultipleChoicePart[]
  hints: string[]
  explanation: string
}): MultipleChoiceQuestionSpec {
  const { explanation, ...question } = input
  return {
    ...question,
    kind: 'multipleChoice',
    hintSchedule: [2, 4],
    validator: validateMultipleChoiceSelections(
      Object.fromEntries(input.parts.map((questionPart) => [questionPart.id, questionPart.correctOptionId])),
      'At least one selection does not match the displayed notebook code.',
    ),
    reveal: { explanation },
    successCopy: 'Correct! Good Job!',
  }
}


const gradientDescentQuestions: MultipleChoiceQuestionSpec[] = [
  notebookQuestion({
    id: 'gradient-function',
    title: 'Read the Two-Variable Gradient',
    prompt: 'Match the update expressions in `fGD` to the gradient of the displayed function.',
    instructions: 'Both new coordinates are calculated before either current coordinate is replaced.',
    datasetId: 'gradientDescentNotebook',
    parts: [
      part('gradient-x', 'What is the x component of the gradient?', 'gx', [
        ['gx', '`2*x + 2*y - 4`', 'Differentiate x², 2xy, and -4x with respect to x.'],
        ['gy', '`2*x + 4*y - 4`', 'This is the y component.'],
        ['value', '`x**2 + 2*x*y - 4*x`', 'This is not a derivative.'],
      ]),
      part('update-timing', 'Which values are used to compute `x_new` and `y_new` during one epoch?', 'same-old-pair', [
        ['same-old-pair', 'Both use the same old x and y', 'Assignments to x and y happen after both new values are computed.'],
        ['updated-x', '`y_new` uses the newly updated x', 'x is not replaced until the next line after `y_new`.'],
        ['updated-y', '`x_new` uses the newly updated y', 'No update occurs before `x_new` is calculated.'],
      ]),
    ],
    hints: ['Differentiate term by term.', 'Read the four assignment lines in exact order.'],
    explanation:
      'The gradient is `(2x + 2y - 4, 2x + 4y - 4)`. The implementation performs a simultaneous-style update because both new coordinates use the previous pair.',
  }),
  notebookQuestion({
    id: 'gradient-regressor-gradients',
    title: 'Trace GDRegressor Gradients',
    prompt: 'Determine the residual and gradient shapes in the full-dataset update.',
    instructions: 'Let X have shape `(n, m)` and y have shape `(n,)`.',
    datasetId: 'gradientDescentNotebook',
    parts: [
      part('residual-definition', 'How are residuals defined?', 'prediction-minus-y', [
        ['prediction-minus-y', '`self.predict(X) - y`', 'This is the exact notebook line.'],
        ['y-minus-prediction', '`y - self.predict(X)`', 'That reverses the gradient sign.'],
        ['squared-residuals', '`(self.predict(X) - y)**2`', 'The gradient uses unsquared residuals.'],
      ]),
      part('gradient-shapes', 'What are the coefficient-gradient and intercept-gradient shapes?', 'm-and-scalar', [
        ['m-and-scalar', '`(m,)` and scalar', '`X.T @ residuals` produces one value per feature; `mean` produces one value.'],
        ['n-and-m', '`(n,)` and `(m,)`', 'The gradient is with respect to parameters, not observations.'],
        ['scalar-scalar', 'both scalar', 'Multiple feature coefficients need separate gradient entries.'],
      ]),
    ],
    hints: ['Write the shapes in `(m,n) @ (n,)`.', 'The intercept is one shared parameter.'],
    explanation:
      'Predicted-minus-actual residuals form an n-vector. Multiplication by X transpose produces one gradient per feature, while their mean is the scalar intercept gradient.',
  }),
  notebookQuestion({
    id: 'gradient-hyperparameters',
    title: 'Interpret Scaling and Hyperparameters',
    prompt: 'Connect notebook preprocessing and constructor arguments to gradient-descent behavior.',
    instructions: 'The optimizer follows a fixed learning rate for a fixed number of iterations.',
    datasetId: 'gradientDescentNotebook',
    parts: [
      part('why-scale', 'Why is displacement standardized before fitting `mpg_mod`?', 'balanced-gradient-scale', [
        ['balanced-gradient-scale', 'To put feature magnitudes on a scale suitable for stable updates', 'Very large feature values can create very large coefficient gradients.'],
        ['make-target-bool', 'To convert mpg to Boolean labels', 'This remains a regression problem.'],
        ['normal-equation', 'To make `np.linalg.inv` possible', 'GDRegressor does not use a matrix inverse.'],
      ]),
      part('large-lr', 'What can happen if the learning rate is too large?', 'overshoot-diverge', [
        ['overshoot-diverge', 'Updates can overshoot and diverge', 'The step size can repeatedly cross or move away from the minimum.'],
        ['more-exact', 'The normal equation becomes more exact', 'No normal equation is used.'],
        ['fewer-features', 'Feature columns are automatically removed', 'Learning rate does not perform feature selection.'],
      ]),
    ],
    hints: ['Inspect the multiplication inside `coef_grad`.', 'Think of repeatedly stepping past a minimum.'],
    explanation:
      'Scaling moderates gradient magnitudes, making one learning rate more useful. A learning rate that is too large can prevent convergence regardless of the maximum iteration count.',
  }),
  notebookQuestion({
    id: 'gradient-one-update',
    title: 'Calculate One Full-Gradient Update',
    prompt: 'Execute one update of `GDRegressor` on a tiny dataset.',
    instructions:
      'Use `X = [[0], [1]]`, `y = [1, 3]`, learning rate 0.2, and the notebook initial coefficient/intercept of 1.',
    datasetId: 'gradientDescentNotebook',
    parts: [
      part('initial-residuals', 'What are the residuals before the update?', 'zero-neg-one', [
        ['zero-neg-one', '`[0, -1]`', 'Initial predictions are `[1, 2]`.'],
        ['zero-one', '`[0, 1]`', 'This uses actual minus predicted.'],
        ['one-one', '`[1, 1]`', 'It ignores the intercept in the predictions.'],
      ]),
      part('updated-parameters', 'What are the coefficient and intercept after the update?', 'both-one-point-one', [
        ['both-one-point-one', '`coef = [1.1]`, `intercept = 1.1`', 'Both gradients are -0.5, so subtracting 0.2 times the gradient adds 0.1.'],
        ['both-point-nine', '`coef = [0.9]`, `intercept = 0.9`', 'This follows the reversed residual convention.'],
        ['coef-one-intercept', '`coef = [1.0]`, `intercept = 1.1`', 'The second row gives a nonzero coefficient gradient.'],
      ]),
    ],
    hints: ['First compute `X @ [1] + 1`.', 'Use `(X.T @ residuals) / 2` and `mean(residuals)`.'],
    explanation:
      'The initial predictions are 1 and 2, so the residuals are 0 and -1. Both gradients equal -0.5, and the learning-rate step raises each parameter from 1 to 1.1.',
  }),
]


export const gradientDescentVersion1Assignment: AssignmentSpec = {
  id: 'gradient-descent', version: 1, displayNumber: 6, title: 'Gradient Descent',
  topic: 'Optimization', description: 'Original code-focused assignment.', published: true,
  questions: [...gradientDescentQuestions, gradientDescentNotebookLab],
}
