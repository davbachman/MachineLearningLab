import type {
  AssignmentSpec,
  MultipleChoicePart,
  MultipleChoiceQuestionSpec,
} from '../types'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import {
  gradientDescentNotebookLab,
  linearRegressionNotebookLab,
  overfittingNotebookLab,
  polynomialRegressionNotebookLab,
} from './regressionCodeLabQuestions'
import {
  batchGradientDescentNotebookLab,
  logisticRegressionNotebookLab,
  regularizationNotebookLab,
  softmaxNotebookLab,
} from './classificationCodeLabQuestions'

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

const linearRegressionQuestions: MultipleChoiceQuestionSpec[] = [
  notebookQuestion({
    id: 'linear-regression-endpoints',
    title: 'Read the Endpoint Model',
    prompt:
      'Trace how `MaxMinLinearRegression.fit` chooses two observations and turns them into a line.',
    instructions:
      'The indices are selected from X first; the corresponding y values are then read at those same indices.',
    datasetId: 'linearRegressionNotebook',
    parts: [
      part('selected-rows', 'Which two rows define this model?', 'x-extremes', [
        ['x-extremes', 'The rows with minimum and maximum X', '`np.argmin(X)` and `np.argmax(X)` select the endpoint rows.'],
        ['y-extremes', 'The rows with minimum and maximum y', 'The target values do not choose the rows.'],
        ['first-last', 'The first and last stored rows', 'Array order matters only if it happens to match the X extremes.'],
      ]),
      part('intercept', 'After computing the slope, how is the intercept obtained?', 'through-min', [
        ['through-min', '`ymin - coef * Xmin`', 'This makes the fitted line pass through the selected minimum-X observation.'],
        ['mean-y', '`y.mean()`', 'The endpoint model does not center the target values.'],
        ['through-max-wrong', '`ymax + coef * Xmax`', 'The sign would not make the line pass through the maximum-X observation.'],
      ]),
    ],
    hints: [
      'Follow `indmin` and `indmax` into all four scalar values.',
      'Substitute Xmin into `coef * x + intercept` and require the result to equal ymin.',
    ],
    explanation:
      'The endpoint implementation chooses rows solely from the smallest and largest X values, computes the rise over run, and chooses an intercept that makes the line pass through `(Xmin, ymin)`.',
  }),
  notebookQuestion({
    id: 'linear-regression-normal-equation',
    title: 'Build the Normal-Equation Design Matrix',
    prompt:
      'Read the shape changes and matrix products in `LinearRegression.fit`.',
    instructions: 'Let the input feature matrix X have shape `(n, m)`.',
    datasetId: 'linearRegressionNotebook',
    parts: [
      part('design-shape', 'What is the shape of `Xnew`?', 'n-m-plus-one', [
        ['n-m-plus-one', '`(n, m + 1)`', 'The extra leading column stores ones for the intercept.'],
        ['n-plus-one-m', '`(n + 1, m)`', 'No observation row is added.'],
        ['m-n', '`(m, n)`', 'The feature matrix is not transposed when it is constructed.'],
      ]),
      part('ones-purpose', 'Why does the first column remain all ones?', 'intercept-coefficient', [
        ['intercept-coefficient', 'It makes the intercept one entry of `coeffs`', 'Multiplying the ones column by `coeffs[0]` adds the same value to every prediction.'],
        ['avoid-inverse', 'It prevents use of a matrix inverse', 'The implementation still calls `np.linalg.inv`.'],
        ['normalize-target', 'It scales y to unit length', 'The target array is not normalized by this operation.'],
      ]),
    ],
    hints: [
      'Read both arguments passed to `np.ones`.',
      'Imagine multiplying one row `[1, x1, x2]` by the coefficient vector.',
    ],
    explanation:
      'The notebook augments every observation with a leading 1, so the first normal-equation coefficient is the intercept and the remaining entries are feature coefficients.',
  }),
  notebookQuestion({
    id: 'linear-regression-prediction-shapes',
    title: 'Respect Prediction Shapes',
    prompt:
      'Determine which arrays represent observations and features in the notebook’s prediction calls.',
    instructions:
      'For a one-feature model, a batch of q observations must have shape `(q, 1)`.',
    datasetId: 'linearRegressionNotebook',
    parts: [
      part('two-observations', 'Which input correctly represents two one-feature observations, 200 and 220?', 'column-batch', [
        ['column-batch', '`np.array([[200], [220]])`', 'There are two rows and one feature column.'],
        ['flat-two', '`np.array([200, 220])`', 'This has two entries on one axis and does not align with a one-entry coefficient vector.'],
        ['row-two', '`np.array([[200, 220]])`', 'This represents one observation with two features.'],
      ]),
      part('dwg-shape', 'If each source array has length n, what is the shape of `DWG = np.array([disp, wt, gr]).T`?', 'n-three', [
        ['n-three', '`(n, 3)`', 'The transpose turns the three source rows into three feature columns.'],
        ['three-n', '`(3, n)`', 'That is the shape before `.T`.'],
        ['n-one', '`(n, 1)`', 'All three features remain present.'],
      ]),
    ],
    hints: [
      'Rows represent observations and columns represent features.',
      'Construct the array before and after applying `.T`.',
    ],
    explanation:
      'The prediction method uses matrix multiplication against one coefficient per feature. A batch therefore needs one row per observation; transposing the three source arrays creates the `(n, 3)` feature matrix used by the final model.',
  }),
  notebookQuestion({
    id: 'linear-regression-losses',
    title: 'Interpret RSS and MSE',
    prompt:
      'Compare the two loss calculations and the effect of expanding the feature matrix.',
    instructions: 'Both losses use the same squared residuals.',
    datasetId: 'linearRegressionNotebook',
    parts: [
      part('rss-mse', 'How are RSS and MSE related for n observations?', 'rss-n-mse', [
        ['rss-n-mse', '`RSS = n * MSE`', 'MSE is the mean of the n squared residuals.'],
        ['same-always', '`RSS = MSE`', 'They coincide only in the special case n = 1.'],
        ['rss-sqrt', '`RSS = sqrt(MSE)`', 'Neither notebook expression takes this square root.'],
      ]),
      part('extra-features', 'What can be said about training RSS after adding weight and gear ratio to displacement and refitting least squares?', 'cannot-increase', [
        ['cannot-increase', 'It cannot be larger than the best displacement-only training RSS', 'The expanded model can reproduce the old solution by assigning zero to the added coefficients.'],
        ['must-increase', 'It must increase because there are more coefficients', 'More available coefficients do not force a worse optimum.'],
        ['test-guarantee', 'Its test RSS must also decrease', 'A training improvement does not guarantee better generalization.'],
      ]),
    ],
    hints: [
      'Compare `.sum()` with `.mean()` on the same array.',
      'Ask whether the larger model can represent every prediction made by the smaller model.',
    ],
    explanation:
      'MSE divides RSS by the observation count. Adding columns enlarges the set of linear predictions, so the optimized training RSS cannot increase, although held-out performance can still worsen.',
  }),
]

const polynomialRegressionQuestions: MultipleChoiceQuestionSpec[] = [
  notebookQuestion({
    id: 'polynomial-scaling',
    title: 'Trace StandardScaler',
    prompt: 'Read what the notebook scaler learns and how it reverses a transformation.',
    instructions: 'For a matrix, `axis=0` computes one value per feature column.',
    datasetId: 'polynomialRegressionNotebook',
    parts: [
      part('learned-values', 'What does `fit` store for an `(n, m)` matrix?', 'm-means-stds', [
        ['m-means-stds', 'm column means and m column standard deviations', 'Both reductions use `axis=0`.'],
        ['n-row-values', 'n row means and n row standard deviations', 'That would require `axis=1`.'],
        ['one-global-pair', 'one global mean and standard deviation', 'The axis argument prevents a global reduction.'],
      ]),
      part('inverse', 'Which expression reverses `transform`?', 'multiply-add', [
        ['multiply-add', '`X * self.std + self.mean`', 'Undo division first, then restore the mean.'],
        ['subtract-divide', '`(X - self.mean) / self.std`', 'That applies the forward transformation again.'],
        ['add-divide', '`(X + self.mean) / self.std`', 'The operations are not inverted in reverse order.'],
      ]),
    ],
    hints: ['Track the output shape of an `axis=0` reduction.', 'Reverse subtraction/division in reverse order.'],
    explanation:
      'The scaler stores column statistics. Its inverse multiplies a standardized value by the stored standard deviation and then adds the stored mean.',
  }),
  notebookQuestion({
    id: 'polynomial-feature-shapes',
    title: 'Build Polynomial Features',
    prompt: 'Determine the columns produced by both branches of `PolynomialFeatures.fit_transform`.',
    instructions: 'The notebook accepts a one-dimensional input X.',
    datasetId: 'polynomialRegressionNotebook',
    parts: [
      part('degree-three-no-bias', 'For degree 3 with `include_bias=False`, what columns are returned?', 'powers-one-three', [
        ['powers-one-three', '`X, X**2, X**3`', 'The loop stores powers `i + 1` for i = 0, 1, 2.'],
        ['powers-zero-two', '`1, X, X**2`', 'That is the include-bias convention for degree 2.'],
        ['only-cube', '`X**3` only', 'Every power through the requested degree is included.'],
      ]),
      part('degree-three-bias-shape', 'For n values, degree 3, and `include_bias=True`, what is the output shape?', 'n-four', [
        ['n-four', '`(n, 4)`', 'The columns are powers 0, 1, 2, and 3.'],
        ['n-three', '`(n, 3)`', 'This omits the bias column.'],
        ['four-n', '`(4, n)`', 'Observations remain rows.'],
      ]),
    ],
    hints: ['List the loop indices in each branch.', '`X**0` is the bias column of ones.'],
    explanation:
      'Without bias the implementation returns degree columns for powers 1 through degree. With bias it returns degree + 1 columns beginning with power zero.',
  }),
  notebookQuestion({
    id: 'polynomial-linear-model',
    title: 'Interpret the Engineered Model',
    prompt: 'Explain why the notebook can fit a curved function with `LinearRegression`.',
    instructions: 'Distinguish linearity in the coefficients from linearity in the original input.',
    datasetId: 'polynomialRegressionNotebook',
    parts: [
      part('nonlinearity-source', 'Where does the nonlinearity in displacement come from?', 'engineered-powers', [
        ['engineered-powers', 'The feature matrix contains powers of scaled displacement', 'The model is linear in coefficients but nonlinear in the original scalar input.'],
        ['changed-loss', 'LinearRegression switches to a nonlinear loss', 'The loss and solver remain ordinary least squares.'],
        ['inverse-scaler', 'The inverse scaler curves the target', 'The target mpg is not inverse-transformed.'],
      ]),
      part('scale-first', 'Why does the notebook scale displacement before taking high powers?', 'control-magnitudes', [
        ['control-magnitudes', 'To keep polynomial feature magnitudes manageable', 'Centered, unit-scale values avoid enormous raw powers.'],
        ['sort-values', 'To sort observations by displacement', 'Sorting is performed separately with `np.argsort`.'],
        ['add-bias', 'To create the intercept column', 'Scaling does not add a column.'],
      ]),
    ],
    hints: ['Write a prediction as c1*x + c2*x².', 'Compare 400³ with a standardized value near 1 cubed.'],
    explanation:
      'Linear regression can weight nonlinear basis functions such as x² and x³. Scaling before constructing them improves numerical behavior without changing which row is which.',
  }),
  notebookQuestion({
    id: 'polynomial-code-trace',
    title: 'Trace a Feature Matrix',
    prompt: 'Mentally execute the notebook class on a two-value input.',
    instructions: 'Use `X = np.array([2, 3])` and degree 2.',
    datasetId: 'polynomialRegressionNotebook',
    parts: [
      part('bias-output', 'What does `PolynomialFeatures(2, include_bias=True).fit_transform(X)` return?', 'bias-matrix', [
        ['bias-matrix', '`[[1, 2, 4], [1, 3, 9]]`', 'The columns are X**0, X**1, and X**2.'],
        ['no-bias-matrix', '`[[2, 4], [3, 9]]`', 'This is the no-bias branch.'],
        ['transposed-matrix', '`[[1, 1], [2, 3], [4, 9]]`', 'The implementation keeps observations in rows.'],
      ]),
      part('column-index', 'In `scaled_disp2[:, 1]`, what does column 1 contain?', 'squared-scaled', [
        ['squared-scaled', '`scaled_disp**2`', 'Zero-based column 1 is the second engineered feature.'],
        ['raw-disp', 'unscaled displacement', 'The feature generator receives `scaled_disp`.'],
        ['bias', 'ones', 'The notebook created `quad` with the default `include_bias=False`.'],
      ]),
    ],
    hints: ['Evaluate powers 0, 1, and 2 row by row.', 'Check how `quad` was constructed before `scaled_disp2`.'],
    explanation:
      'The include-bias branch begins with ones. The notebook’s `quad` object omits bias, so its two columns are scaled displacement and its square.',
  }),
]

const overfittingQuestions: MultipleChoiceQuestionSpec[] = [
  notebookQuestion({
    id: 'overfitting-split',
    title: 'Read the Actual Train/Test Split',
    prompt: 'Compare the prose near the split with the executable argument in the notebook.',
    instructions: 'Answer from the displayed Python call, not from the nearby prose.',
    datasetId: 'overfittingNotebook',
    parts: [
      part('split-fractions', 'What fractions does `test_size=0.8` request?', 'twenty-eighty', [
        ['twenty-eighty', '20% training and 80% testing', 'The explicit size reserves 80% for the test arrays.'],
        ['eighty-twenty', '80% training and 20% testing', 'That would use `test_size=0.2` or `train_size=0.8`.'],
        ['eighty-eighty', '80% in both sets', 'The returned sets are disjoint.'],
      ]),
      part('repeatability', 'Why can the exact rows change between fresh notebook runs?', 'no-random-state', [
        ['no-random-state', '`random_state` is not supplied', 'The split is not fixed by the call.'],
        ['sorted-input', 'The input was sorted first', 'Sorting alone does not randomize or stabilize the random split.'],
        ['test-large', 'The test set is larger', 'Its size does not determine repeatability.'],
      ]),
    ],
    hints: ['The named argument describes the test arrays.', 'Look for a seed or `random_state` in the function call.'],
    explanation:
      'The executable code reserves 80% for testing even though the prose says 80/20 in the opposite direction. With no `random_state`, the membership is not reproducible across fresh runs.',
  }),
  notebookQuestion({
    id: 'overfitting-preprocessing',
    title: 'Avoid Preprocessing Leakage',
    prompt: 'Trace which data determine the scaling statistics.',
    instructions: 'Fitting learns parameters; transforming applies already learned parameters.',
    datasetId: 'overfittingNotebook',
    parts: [
      part('scaler-fit', 'Which array is used in `disp_scaler.fit(...)`?', 'xtrain', [
        ['xtrain', '`Xtrain` only', 'Only the training displacement values determine mean and scale.'],
        ['xtest', '`Xtest` only', 'Test values are transformed but not used to fit.'],
        ['all-disp', 'the complete `disp` array', 'That would leak test-distribution information into preprocessing.'],
      ]),
      part('same-scaler', 'Why is the fitted training scaler also used on Xtest?', 'same-coordinate-system', [
        ['same-coordinate-system', 'To put train and test rows in the same learned coordinate system', 'The model expects features transformed with its training parameters.'],
        ['force-zero-test-mean', 'To guarantee Xtest has mean zero', 'A test set need not have zero mean under training statistics.'],
        ['increase-degree', 'To add polynomial columns', 'PolynomialFeatures performs that separate step.'],
      ]),
    ],
    hints: ['Locate the single `.fit` call.', 'The test set should simulate future unseen observations.'],
    explanation:
      'The notebook correctly learns scaling parameters from Xtrain and reuses them for Xtest, preventing test-set information from influencing model preparation.',
  }),
  notebookQuestion({
    id: 'overfitting-degree-eight',
    title: 'Count Degree-Eight Features',
    prompt: 'Read the polynomial and linear-model configuration together.',
    instructions: '`PolynomialFeatures` counts the bias as degree zero.',
    datasetId: 'overfittingNotebook',
    parts: [
      part('feature-count', 'How many columns does degree 8 produce with `include_bias=True` for one input feature?', 'nine-columns', [
        ['nine-columns', '9 columns', 'The powers are 0 through 8.'],
        ['eight-columns', '8 columns', 'This omits the requested bias column.'],
        ['sixteen-columns', '16 columns', 'The train/test split does not double the feature count.'],
      ]),
      part('no-intercept', 'Why is `LinearRegression(fit_intercept=False)` paired with those features?', 'bias-is-intercept', [
        ['bias-is-intercept', 'The power-zero column already represents an intercept', 'A separate fitted intercept would duplicate the constant term.'],
        ['prevent-overfit', 'It guarantees the model cannot overfit', 'A degree-eight model can still overfit.'],
        ['required-shape', 'The nine-column matrix otherwise has the wrong shape', 'The matrix shape is acceptable either way.'],
      ]),
    ],
    hints: ['List powers beginning at zero.', 'A coefficient multiplying an all-ones column is a constant offset.'],
    explanation:
      'Including bias creates nine columns, with the first all ones. Setting `fit_intercept=False` avoids fitting a second constant term.',
  }),
  notebookQuestion({
    id: 'overfitting-metrics-loop',
    title: 'Audit the Metrics Loop',
    prompt: 'Identify what the cell records and what happens if the notebook is run exactly as supplied.',
    instructions: 'Names must have been defined or imported before Python can call them.',
    datasetId: 'overfittingNotebook',
    parts: [
      part('first-failure', 'What happens at `poly_model = PolynomialRegression(d+1)` in this notebook?', 'name-error', [
        ['name-error', 'A NameError occurs because `PolynomialRegression` is not defined or imported', 'The imports include `PolynomialFeatures`, not a `PolynomialRegression` class.'],
        ['sklearn-model', 'It constructs sklearn LinearRegression automatically', 'Python does not infer this alias.'],
        ['degree-error', 'It raises an error only when d reaches 8', 'The undefined name fails on the first iteration.'],
      ]),
      part('best-degree-rule', 'If the loop were made executable, which expression selects the degree with minimum test MSE?', 'test-argmin-plus-one', [
        ['test-argmin-plus-one', '`metrics[:, 1].argmin() + 1`', 'Column 1 is test MSE and row 0 represents degree 1.'],
        ['train-argmin', '`metrics[:, 0].argmin()`', 'This uses training MSE and returns a zero-based row.'],
        ['test-argmax', '`metrics[:, 1].argmax() + 1`', 'The objective is to minimize MSE.'],
      ]),
    ],
    hints: ['Search the earlier cells for a definition of `PolynomialRegression`.', 'Map row d to polynomial degree d + 1.'],
    explanation:
      'As supplied, the metrics loop cannot run because `PolynomialRegression` is undefined. Once a working model is substituted, the second metrics column is test MSE and its zero-based minimum position must be shifted by one to obtain the degree.',
  }),
]

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

const batchGradientDescentQuestions: MultipleChoiceQuestionSpec[] = [
  notebookQuestion({
    id: 'batch-shuffling',
    title: 'Trace Epoch Shuffling',
    prompt: 'Read how the notebook constructs reproducible shuffled batches.',
    instructions: 'The same `indices` array indexes both X and y.',
    datasetId: 'batchGradientDescentNotebook',
    parts: [
      part('seed-effect', 'What does `np.random.seed(i)` accomplish inside the epoch loop?', 'reproducible-by-epoch', [
        ['reproducible-by-epoch', 'Each epoch’s shuffle is reproducible and depends on i', 'Fresh runs use the same permutation sequence.'],
        ['same-every-epoch', 'Every epoch receives exactly the same permutation', 'The seed changes with i.'],
        ['no-shuffle', 'It disables random shuffling', '`np.random.shuffle` still permutes the indices.'],
      ]),
      part('pairing', 'Why do features and targets stay correctly paired?', 'same-permutation', [
        ['same-permutation', 'Both arrays are indexed by the same shuffled indices', 'Each target moves with its observation row.'],
        ['sorted-target', 'y is sorted after shuffling', 'No sorting occurs.'],
        ['seed-target', 'A separate target seed repairs the order', 'There is only one index permutation.'],
      ]),
    ],
    hints: ['Compare the seeds in epochs 0 and 1.', 'Read the right sides of `X_shuffle` and `y_shuffle`.'],
    explanation:
      'Seeding with the epoch index yields deterministic but different epoch permutations. Applying one permutation to both arrays preserves row-target correspondence.',
  }),
  notebookQuestion({
    id: 'batch-slices',
    title: 'Count Batch Updates',
    prompt: 'Interpret the inner range and its final possibly short slice.',
    instructions: 'Python slices stop at the array end even when the requested endpoint is larger.',
    datasetId: 'batchGradientDescentNotebook',
    parts: [
      part('updates-per-epoch', 'For n rows and batch size b, how many updates occur per epoch?', 'ceiling', [
        ['ceiling', '`ceil(n / b)`', 'Every nonempty slice causes one update, including a final partial batch.'],
        ['floor', '`floor(n / b)`', 'This drops the final partial batch.'],
        ['n', 'n updates', 'That is true only for batch size 1.'],
      ]),
      part('notebook-count', 'For 100,000 rows and batch size 4,150, how many updates occur per epoch?', 'twenty-five', [
        ['twenty-five', '25', 'Twenty-four full starts cover through index 95,449, followed by one final slice.'],
        ['twenty-four', '24', 'This omits the partial final batch.'],
        ['four-thousand', '4,150', 'That is the batch size, not the update count.'],
      ]),
    ],
    hints: ['List starts 0, b, 2b, ... below n.', 'Use a ceiling because 100000 is not divisible by 4150.'],
    explanation:
      'The range visits every valid batch start. A non-divisible row count creates one shorter final batch, so the update count is the ceiling of n divided by batch size.',
  }),
  notebookQuestion({
    id: 'batch-gradient-scaling',
    title: 'Read the Notebook’s Gradient Scaling',
    prompt: 'Distinguish the active denominator from the commented alternative.',
    instructions: 'Answer from the uncommented code.',
    datasetId: 'batchGradientDescentNotebook',
    parts: [
      part('denominator', 'What divides the active coefficient and intercept sums?', 'full-length', [
        ['full-length', '`len(X)`', 'Both active gradient lines use the full dataset length.'],
        ['batch-length', '`len(X_batch)`', 'Those batch-mean alternatives are commented out.'],
        ['epochs', '`self.max_iter`', 'Epoch count does not normalize a batch gradient.'],
      ]),
      part('relative-scale', 'For a full batch of size b < n, how does this compare with dividing by b?', 'b-over-n', [
        ['b-over-n', 'The active gradient is `b/n` times the batch-mean gradient', 'The same numerator is divided by n instead of b.'],
        ['n-over-b', 'The active gradient is `n/b` times larger', 'Dividing by the larger n makes it smaller.'],
        ['identical', 'They are always identical', 'They coincide only when b = n.'],
      ]),
    ],
    hints: ['Ignore the lines beginning with `#`.', 'Compare `S/n` with `S/b`.'],
    explanation:
      'The supplied implementation scales each batch contribution by the entire dataset length. Relative to a batch mean, that shrinks a full b-row batch gradient by b/n.',
  }),
  notebookQuestion({
    id: 'batch-special-cases',
    title: 'Connect Batch Size to GD and SGD',
    prompt: 'Identify limiting cases of the notebook’s batch loop.',
    instructions: 'Keep the other constructor arguments fixed.',
    datasetId: 'batchGradientDescentNotebook',
    parts: [
      part('batch-one', 'What does `batch_size=1` represent?', 'stochastic', [
        ['stochastic', 'One stochastic update per shuffled observation', 'Each slice contains one row.'],
        ['full-batch', 'One full-dataset update per epoch', 'That requires a batch at least as large as n.'],
        ['no-updates', 'No updates because the batch is too small', 'Every one-row slice is nonempty.'],
      ]),
      part('batch-at-least-n', 'What happens when `batch_size >= len(X)`?', 'one-update', [
        ['one-update', 'There is one full-dataset update per epoch', 'Only the j = 0 slice is produced.'],
        ['n-updates', 'There are n updates per epoch', 'The range step is the large batch size.'],
        ['empty', 'The first slice is empty', 'The slice begins at zero and includes all rows.'],
      ]),
    ],
    hints: ['Write the starts produced by `range(0, n, batch_size)`.', 'A slice from zero past the end returns all available rows.'],
    explanation:
      'A one-row batch gives stochastic updates; a batch at least as large as the data yields a single full-gradient-style update during each epoch.',
  }),
]

const regularizationQuestions: MultipleChoiceQuestionSpec[] = [
  notebookQuestion({
    id: 'regularization-penalties',
    title: 'Compare L1 and L2 Penalty Gradients',
    prompt: 'Trace the lambdas selected by the `penalty` constructor argument.',
    instructions: 'Use the exact NumPy functions in the supplied class.',
    datasetId: 'regularizationNotebook',
    parts: [
      part('gradients', 'Which pair matches the implementation?', 'l1-sign-l2-value', [
        ['l1-sign-l2-value', 'L1 uses `sign(x)`; L2 uses `x`', 'Those are the two assigned lambdas.'],
        ['l1-x-l2-sign', 'L1 uses `x`; L2 uses `sign(x)`', 'This reverses the branches.'],
        ['both-square', 'Both use `x**2`', 'The derivative, not the penalty value, is used.'],
      ]),
      part('l1-at-zero', 'What does the notebook’s L1 gradient return at exactly zero?', 'zero', [
        ['zero', '0', '`np.sign(0)` is zero.'],
        ['one', '1', 'That would come from the commented alternative `2*(x>0)-1` only for positive values.'],
        ['undefined', 'It raises an exception', 'NumPy defines the sign output at zero.'],
      ]),
    ],
    hints: ['Read which lambda is active rather than commented.', 'Evaluate `np.sign(0)`.'],
    explanation:
      'The L1 branch uses NumPy sign, which is -1, 0, or 1. The L2 branch uses the current parameter value, so larger magnitudes receive larger shrinkage.',
  }),
  notebookQuestion({
    id: 'regularization-update',
    title: 'Parse the Regularized Update',
    prompt: 'Read the placement of learning rate and alpha in the two update statements.',
    instructions: 'Python evaluates both additive terms before subtracting them.',
    datasetId: 'regularizationNotebook',
    parts: [
      part('alpha-scaling', 'Which expression is subtracted from a coefficient?', 'exact-update', [
        ['exact-update', '`lr * coef_grad + alpha * penalty_grad(coef)`', 'Only the data-gradient term is multiplied by learning rate.'],
        ['lr-all', '`lr * (coef_grad + alpha * penalty_grad(coef))`', 'That would also scale the penalty term by lr.'],
        ['alpha-data', '`alpha * coef_grad + lr * penalty_grad(coef)`', 'The hyperparameters are exchanged.'],
      ]),
      part('intercept-penalty', 'Does this implementation regularize the intercept?', 'yes', [
        ['yes', 'Yes, with the same selected penalty gradient', 'The intercept update includes `self.alpha*penalty_grad(self.intercept)`.'],
        ['no', 'No, only coefficients are penalized', 'That is common elsewhere but not what this notebook does.'],
        ['l1-only', 'Only when penalty is L1', 'Both branches define `penalty_grad` used by the intercept line.'],
      ]),
    ],
    hints: ['Notice where the multiplication by `self.lr` ends.', 'Read the complete intercept update line.'],
    explanation:
      'The notebook adds an alpha-scaled penalty gradient outside the learning-rate product and applies the same form to the intercept as to every coefficient.',
  }),
  notebookQuestion({
    id: 'regularization-polynomial',
    title: 'Use L2 Against Polynomial Overfitting',
    prompt: 'Interpret the degree-12 experiment and its expected loss tradeoff.',
    instructions: 'The polynomial transformer is configured without a bias column.',
    datasetId: 'regularizationNotebook',
    parts: [
      part('degree-columns', 'How many feature columns are produced from one input feature?', 'twelve', [
        ['twelve', '12', 'The columns contain powers 1 through 12.'],
        ['thirteen', '13', 'There is no power-zero bias column.'],
        ['sixteen', '16', 'Sixteen is the original observation count.'],
      ]),
      part('expected-tradeoff', 'What comparison does the notebook expect after adding L2 regularization?', 'train-up-test-down', [
        ['train-up-test-down', 'Higher training MSE but lower testing MSE', 'Some training fit is traded for better generalization.'],
        ['both-zero', 'Both MSE values become zero', 'Regularization does not guarantee exact predictions.'],
        ['train-down-test-up', 'Lower training MSE but higher testing MSE', 'That is the opposite of the stated purpose in this experiment.'],
      ]),
    ],
    hints: ['Check `include_bias=False`.', 'Regularization deliberately restricts an overly flexible fit.'],
    explanation:
      'The degree-12 no-bias matrix has twelve columns. L2 shrinkage can worsen the fit to training rows while reducing variance enough to improve test error.',
  }),
  notebookQuestion({
    id: 'regularization-feature-selection',
    title: 'Trace the Feature-Selection Slice',
    prompt: 'Read how the California housing matrix is scaled and reduced.',
    instructions: 'Column indices follow the DataFrame order from MedInc through Longitude.',
    datasetId: 'regularizationNotebook',
    parts: [
      part('scaler-data', 'Which data determine the scaling parameters?', 'train-only', [
        ['train-only', '`X_train` only', 'The scaler is fit before transforming either split.'],
        ['test-only', '`X_test` only', 'The test set is transformed but never fitted.'],
        ['both', 'The concatenated train and test matrices', 'The code never concatenates them.'],
      ]),
      part('selected-features', 'Which named features are kept by `[:, [0, 1]]`?', 'medinc-houseage', [
        ['medinc-houseage', 'MedInc and HouseAge', 'They are the first two feature columns in the selected DataFrame range.'],
        ['latitude-longitude', 'Latitude and Longitude', 'Those are the final two columns.'],
        ['medinc-longitude', 'MedInc and Longitude', 'That would use indices 0 and 7.'],
      ]),
    ],
    hints: ['Find the one `.fit` call on `Xscaler`.', 'Read the California housing column order shown by the DataFrame.'],
    explanation:
      'Training rows alone determine scaling. The hard-coded two-column slice retains the first two housing features, MedInc and HouseAge, for the smaller model.',
  }),
]

const logisticRegressionQuestions: MultipleChoiceQuestionSpec[] = [
  notebookQuestion({
    id: 'logistic-sigmoid',
    title: 'Trace Sigmoid and the Class Rule',
    prompt: 'Convert a linear score into a probability and then into a Boolean prediction.',
    instructions: 'The comparison in `predict` is strictly greater than 0.5.',
    datasetId: 'logisticRegressionNotebook',
    parts: [
      part('zero-score', 'What probability is produced when t = 0?', 'half', [
        ['half', '0.5', '`exp(0)` is 1, so the sigmoid denominator is 2.'],
        ['zero', '0', 'The sigmoid never maps a finite score to exactly zero.'],
        ['one', '1', 'That is only approached for very large positive scores.'],
      ]),
      part('boundary-class', 'What class does the notebook predict when the probability is exactly 0.5?', 'false', [
        ['false', '`False`', 'The expression `0.5 > 0.5` is false.'],
        ['true', '`True`', 'That would require `>= 0.5`.'],
        ['error', 'A tie error', 'NumPy Boolean comparison handles equality directly.'],
      ]),
    ],
    hints: ['Substitute zero into `1/(1+exp(-t))`.', 'Pay attention to `>` versus `>=`.'],
    explanation:
      'The sigmoid maps zero to exactly 0.5, and the implementation assigns that boundary probability to False because its comparison is strict.',
  }),
  notebookQuestion({
    id: 'logistic-gradient',
    title: 'Connect Logistic and Linear Gradients',
    prompt: 'Identify what changed when the SGD regressor became a classifier.',
    instructions: 'The coefficient and intercept gradient formulas retain their earlier structure.',
    datasetId: 'logisticRegressionNotebook',
    parts: [
      part('residual', 'What is one logistic residual?', 'prob-minus-y', [
        ['prob-minus-y', '`predicted_probability - y`', 'The notebook subtracts the 0/1 target from `predict_proba`.'],
        ['class-minus-y', '`predicted_class - y`', 'The gradient uses probabilities, not thresholded classes.'],
        ['score-minus-y', '`linear_score - y`', 'The sigmoid is applied before residuals are formed.'],
      ]),
      part('coef-gradient', 'What is the coefficient-gradient expression for a batch?', 'xt-residual-mean', [
        ['xt-residual-mean', '`X_batch.T @ residuals / len(X_batch)`', 'This is the exact active line.'],
        ['residual-xt', '`residuals.T @ X_batch.T`', 'Those dimensions do not produce one gradient per feature.'],
        ['mean-only', '`np.mean(residuals)`', 'That is the intercept gradient.'],
      ]),
    ],
    hints: ['Look at the first line inside the batch loop that differs from regression.', 'The feature matrix weights each residual in the coefficient gradient.'],
    explanation:
      'Logistic regression changes residuals to probability minus target. Once those residuals are available, the coefficient and intercept gradients have the same matrix/mean forms as linear regression.',
  }),
  notebookQuestion({
    id: 'logistic-evaluation',
    title: 'Distinguish Accuracy and Log Loss',
    prompt: 'Compare the notebook’s `score` and `NegLogLoss` methods.',
    instructions: 'Accuracy uses classes; log loss uses probabilities.',
    datasetId: 'logisticRegressionNotebook',
    parts: [
      part('score', 'What does `score` return?', 'fraction-correct', [
        ['fraction-correct', 'The fraction of Boolean predictions equal to y', 'Boolean equality is averaged across rows.'],
        ['mean-probability', 'The mean predicted probability', '`score` calls `predict`, not `predict_proba` directly.'],
        ['negative-loss', 'The negative mean log loss', 'That has its own method.'],
      ]),
      part('confident-wrong', 'How does log loss treat an increasingly confident wrong prediction?', 'larger-penalty', [
        ['larger-penalty', 'Its penalty grows', 'The assigned correct-class probability approaches zero and its negative log grows.'],
        ['same-penalty', 'Its penalty is unchanged once the class is wrong', 'That describes 0/1 error, not log loss.'],
        ['smaller-penalty', 'Its penalty shrinks', 'Confidence in the wrong class moves the loss in the opposite direction.'],
      ]),
    ],
    hints: ['Read the return line of each method.', 'Consider `-log(p)` as the correct-class probability p approaches zero.'],
    explanation:
      'Accuracy records only whether the thresholded class is right. Negative mean log loss retains confidence information and strongly penalizes wrong predictions made with high confidence.',
  }),
  notebookQuestion({
    id: 'logistic-iris-shapes',
    title: 'Read the Iris Feature Matrix',
    prompt: 'Trace the two selected columns and the decision-boundary expression.',
    instructions: 'The notebook predicts whether each row is virginica.',
    datasetId: 'logisticRegressionNotebook',
    parts: [
      part('x-shape', 'What is the shape of X after selecting Petal Length and Petal Width?', 'one-fifty-two', [
        ['one-fifty-two', '`(150, 2)`', 'There are 150 iris rows and two selected feature columns.'],
        ['two-one-fifty', '`(2, 150)`', 'Observations remain rows.'],
        ['one-fifty-four', '`(150, 4)`', 'The other two iris measurements were not selected.'],
      ]),
      part('boundary-slope', 'What slope does `func` use for the 0.5 decision boundary?', 'negative-ratio', [
        ['negative-ratio', '`-coef[0] / coef[1]`', 'Setting the linear score to zero and solving for feature 2 gives this slope.'],
        ['positive-ratio', '`coef[0] / coef[1]`', 'This misses the sign introduced when moving the first term.'],
        ['intercept-only', '`-intercept`', 'The coefficients determine the slope.'],
      ]),
    ],
    hints: ['Count the DataFrame columns inside the double brackets.', 'At probability 0.5, the linear score t equals zero.'],
    explanation:
      'The model uses two petal measurements for each of 150 flowers. Its probability boundary occurs where the linear score is zero, giving slope `-coef[0]/coef[1]`.',
  }),
]

const softmaxQuestions: MultipleChoiceQuestionSpec[] = [
  notebookQuestion({
    id: 'softmax-one-hot',
    title: 'Trace One-Hot Encoding',
    prompt: 'Read what `fit` learns and why the same fitted encoder is reused.',
    instructions: '`np.unique` determines the stored category order.',
    datasetId: 'softmaxNotebook',
    parts: [
      part('category-order', "For `y = ['b', 'a', 'c', 'b']`, what is `encoder.categories`?", 'a-b-c', [
        ['a-b-c', "`['a', 'b', 'c']`", '`np.unique` returns the distinct strings in sorted order.'],
        ['b-a-c', "`['b', 'a', 'c']`", 'First-appearance order is not retained by `np.unique` here.'],
        ['zero-one-two', '`[0, 1, 2]`', 'The stored values are the original category labels.'],
      ]),
      part('reuse', 'Why call `encoder.transform(z)` instead of fitting a new encoder to z?', 'same-columns', [
        ['same-columns', 'To preserve the training category columns and their order', 'The model output columns correspond to the fitted training categories.'],
        ['fewer-rows', 'To make z contain fewer observations', 'Encoding does not change row count.'],
        ['scale-labels', 'To standardize string magnitudes', 'One-hot encoding is not numeric scaling.'],
      ]),
    ],
    hints: ['Recall the ordering behavior of `np.unique`.', 'Each probability column must keep one stable meaning.'],
    explanation:
      'The encoder stores sorted unique training labels. Reusing it ensures every later one-hot or probability column refers to the same category even if a new array omits one category.',
  }),
  notebookQuestion({
    id: 'softmax-shapes',
    title: 'Check Softmax Parameter Shapes',
    prompt: 'Follow matrix multiplication through a model with n observations, m features, and k classes.',
    instructions: 'Broadcasting adds one k-entry intercept row to every observation.',
    datasetId: 'softmaxNotebook',
    parts: [
      part('coefficient-shape', 'What is the shape of `self.coef`?', 'm-k', [
        ['m-k', '`(m, k)`', '`(n, m) @ (m, k)` produces one k-score row per observation.'],
        ['k-m', '`(k, m)`', 'Its inner dimension would not align with X as written.'],
        ['n-k', '`(n, k)`', 'That is the score/probability shape, not a parameter shape.'],
      ]),
      part('probability-shape', 'What is the shape of `predict_proba(X)`?', 'n-k', [
        ['n-k', '`(n, k)`', 'Each observation receives one probability per class.'],
        ['n-m', '`(n, m)`', 'Feature count does not determine output class count.'],
        ['k-only', '`(k,)`', 'That would describe only one observation.'],
      ]),
    ],
    hints: ['Apply the matrix multiplication shape rule.', 'There is one output row per input row.'],
    explanation:
      'An m-by-k coefficient matrix turns each m-feature row into k class scores, so both scores and probabilities have shape n by k.',
  }),
  notebookQuestion({
    id: 'softmax-normalization',
    title: 'Normalize and Predict Classes',
    prompt: 'Read the axis used by softmax and the mapping from a maximum column to a label.',
    instructions: 'Each row represents one observation.',
    datasetId: 'softmaxNotebook',
    parts: [
      part('normalization-axis', 'Why does the denominator sum with `axis=1`?', 'row-sum-one', [
        ['row-sum-one', 'It makes each observation’s class probabilities sum to 1', 'Axis 1 reduces across the class columns within each row.'],
        ['column-sum-one', 'It makes each class column sum to 1 across the dataset', 'That would normalize across observations.'],
        ['single-number', 'It produces one denominator for the whole matrix', 'An axis-specific reduction returns one value per row.'],
      ]),
      part('prediction-map', 'After `argmax(probs, axis=1)`, how are class labels recovered?', 'categories-index', [
        ['categories-index', '`self.encoder.categories[indices]`', 'The maximum column index selects its stored category.'],
        ['return-indices', 'The numeric indices are returned directly', 'The notebook maps them back to original labels.'],
        ['refit-encoder', 'The encoder is refit on the indices', 'Prediction does not alter the encoder.'],
      ]),
    ],
    hints: ['Axis 1 runs horizontally across class columns.', 'The encoder records what each column means.'],
    explanation:
      'Softmax normalizes each score row independently. The largest probability column is an integer position, which is mapped back through the fitted category array.',
  }),
  notebookQuestion({
    id: 'softmax-loss-gradient',
    title: 'Audit the Multiclass Loss and Intercept Gradient',
    prompt: 'Compare the shapes implied by categorical cross entropy with the exact intercept-gradient line.',
    instructions: 'Let residuals have shape `(batch_size, k)`.',
    datasetId: 'softmaxNotebook',
    parts: [
      part('loss-selection', 'Why does `Y * np.log(probs)` retain only the assigned class contribution in each row?', 'one-hot-mask', [
        ['one-hot-mask', 'Y is one-hot, so all non-assigned class entries are multiplied by zero', 'Only the true-category column contains a one.'],
        ['argmax-first', '`argmax` runs before multiplication', 'CEloss does not call argmax.'],
        ['logs-zero', 'All nonmaximum log probabilities are zero', 'Their logs are generally negative, not zero.'],
      ]),
      part('actual-intercept-gradient', 'What shape does the notebook’s `np.mean(residuals)` return?', 'scalar', [
        ['scalar', 'A scalar', 'With no axis argument, NumPy averages every entry.'],
        ['k-vector', 'A `(k,)` vector', 'That would require `np.mean(residuals, axis=0)`.'],
        ['batch-vector', 'A `(batch_size,)` vector', 'That would require reducing across class columns.'],
      ]),
    ],
    hints: ['One-hot rows contain exactly one 1.', 'Check the default behavior of `np.mean` without an axis.'],
    explanation:
      'One-hot multiplication selects each row’s assigned-class log probability. The supplied fit code computes a scalar intercept gradient, which broadcasts equally to every class; a class-specific gradient would average residuals along axis 0.',
  }),
]

function assignment(input: {
  id: string
  displayNumber: number
  title: string
  topic: string
  description: string
  questions: MultipleChoiceQuestionSpec[]
  lab: AssignmentSpec['questions'][number]
}): AssignmentSpec {
  return {
    id: input.id,
    displayNumber: input.displayNumber,
    version: 1,
    title: input.title,
    topic: input.topic,
    description: input.description,
    published: true,
    questions: [...input.questions, input.lab],
  }
}

export const linearRegressionAssignment = assignment({
  id: 'linear-regression',
  displayNumber: 5,
  title: 'Linear Regression',
  topic: 'Regression',
  description:
    'Read endpoint and least-squares implementations, track feature-matrix shapes, and interpret RSS and MSE.',
  questions: linearRegressionQuestions,
  lab: linearRegressionNotebookLab,
})

export const polynomialRegressionAssignment = assignment({
  id: 'polynomial-regression',
  displayNumber: 6,
  title: 'Polynomial Regression',
  topic: 'Feature Engineering',
  description:
    'Trace scaling and polynomial feature construction, then connect engineered powers to nonlinear predictions.',
  questions: polynomialRegressionQuestions,
  lab: polynomialRegressionNotebookLab,
})

export const overfittingAssignment = assignment({
  id: 'overfitting',
  displayNumber: 7,
  title: 'Overfitting',
  topic: 'Model Evaluation',
  description:
    'Audit train/test preprocessing, polynomial complexity, and training-versus-testing error directly from the notebook.',
  questions: overfittingQuestions,
  lab: overfittingNotebookLab,
})

export const gradientDescentAssignment = assignment({
  id: 'gradient-descent',
  displayNumber: 8,
  title: 'Gradient Descent',
  topic: 'Optimization',
  description:
    'Trace simultaneous gradient updates and the full-dataset GDRegressor implementation for mean squared error.',
  questions: gradientDescentQuestions,
  lab: gradientDescentNotebookLab,
})

export const batchGradientDescentAssignment = assignment({
  id: 'batch-gradient-descent',
  displayNumber: 9,
  title: 'Batch Gradient Descent',
  topic: 'Optimization',
  description:
    'Follow reproducible shuffling, batch slices, gradient scaling, and the relationship among full, batch, and stochastic updates.',
  questions: batchGradientDescentQuestions,
  lab: batchGradientDescentNotebookLab,
})

export const regularizationAssignment = assignment({
  id: 'regularization',
  displayNumber: 10,
  title: 'Regularization',
  topic: 'Generalization',
  description:
    'Compare L1 and L2 updates, evaluate the overfitting tradeoff, and trace L1-guided feature selection.',
  questions: regularizationQuestions,
  lab: regularizationNotebookLab,
})

export const logisticRegressionAssignment = assignment({
  id: 'logistic-regression',
  displayNumber: 11,
  title: 'Logistic Regression',
  topic: 'Binary Classification',
  description:
    'Connect sigmoid probabilities, cross-entropy gradients, thresholded predictions, and the Iris decision boundary.',
  questions: logisticRegressionQuestions,
  lab: logisticRegressionNotebookLab,
})

export const softmaxAssignment = assignment({
  id: 'softmax',
  displayNumber: 12,
  title: 'Softmax',
  topic: 'Multiclass Classification',
  description:
    'Trace one-hot encoding, multiclass parameter shapes, row-wise softmax probabilities, and categorical cross entropy.',
  questions: softmaxQuestions,
  lab: softmaxNotebookLab,
})

export const notebookAssignments = [
  linearRegressionAssignment,
  polynomialRegressionAssignment,
  overfittingAssignment,
  gradientDescentAssignment,
  batchGradientDescentAssignment,
  regularizationAssignment,
  logisticRegressionAssignment,
  softmaxAssignment,
]
