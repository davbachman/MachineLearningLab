import type {
  AssignmentSpec,
  MultipleChoicePart,
  MultipleChoiceQuestionSpec,
} from '../types'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import { polynomialRegressionAssignment } from './polynomialAssignment'
import { gradientDescentAssignment } from './gradientDescentAssignment'
export { gradientDescentAssignment } from './gradientDescentAssignment'
export { polynomialRegressionAssignment } from './polynomialAssignment'
import { linearFitQuestion } from './linearFitQuestion'
import { planeFitQuestion } from './planeFitQuestion'
import {
  linearRegressionNotebookLab,
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
  version?: number
  displayNumber: number
  title: string
  topic: string
  description: string
  questions: AssignmentSpec['questions']
  lab: AssignmentSpec['questions'][number]
}): AssignmentSpec {
  return {
    id: input.id,
    displayNumber: input.displayNumber,
    version: input.version ?? 1,
    title: input.title,
    topic: input.topic,
    description: input.description,
    published: true,
    questions: [...input.questions, input.lab],
  }
}

export const linearRegressionAssignment = assignment({
  id: 'linear-regression',
  version: 2,
  displayNumber: 4,
  title: 'Linear Regression',
  topic: 'Regression',
  description:
    'Fit a line and a plane by minimizing RSS, then trace the normal-equation implementation in the notebook lab.',
  questions: [linearFitQuestion, planeFitQuestion],
  lab: linearRegressionNotebookLab,
})


export const batchGradientDescentAssignment = assignment({
  id: 'batch-gradient-descent',
  displayNumber: 7,
  title: 'Batch Gradient Descent',
  topic: 'Optimization',
  description:
    'Follow reproducible shuffling, batch slices, gradient scaling, and the relationship among full, batch, and stochastic updates.',
  questions: batchGradientDescentQuestions,
  lab: batchGradientDescentNotebookLab,
})

export const regularizationAssignment = assignment({
  id: 'regularization',
  displayNumber: 8,
  title: 'Regularization',
  topic: 'Generalization',
  description:
    'Compare L1 and L2 updates, evaluate the overfitting tradeoff, and trace L1-guided feature selection.',
  questions: regularizationQuestions,
  lab: regularizationNotebookLab,
})

export const logisticRegressionAssignment = assignment({
  id: 'logistic-regression',
  displayNumber: 9,
  title: 'Logistic Regression',
  topic: 'Binary Classification',
  description:
    'Connect sigmoid probabilities, cross-entropy gradients, thresholded predictions, and the Iris decision boundary.',
  questions: logisticRegressionQuestions,
  lab: logisticRegressionNotebookLab,
})

export const softmaxAssignment = assignment({
  id: 'softmax',
  displayNumber: 10,
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
  gradientDescentAssignment,
  batchGradientDescentAssignment,
  regularizationAssignment,
  logisticRegressionAssignment,
  softmaxAssignment,
]
