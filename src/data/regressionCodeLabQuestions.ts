import { defineCodeLabQuestion, type CodeLabQuestionSpec } from '../lib/codeLab'

const linearRegressionCode = `import numpy as np

class LinearRegression():
    def __init__(self):
        pass

    def fit(self,X,y):
        '''stores the slope and intercept
        for the model defined by X and y'''
        Xnew=np.ones((X.shape[0],X.shape[1]+1))  # S1
        Xnew[:,1:]=X  # S2
        coeffs=np.linalg.inv(Xnew.T@Xnew)@(Xnew.T@y)  # S3
        self.intercept=coeffs[0]  # S4
        self.coef=coeffs[1:]  # S5

    def predict(self,x):
        '''x is expected to have shape
        (num_test_obs,num_feats)'''
        return x@self.coef+self.intercept  # S6`

const polynomialRegressionCode = `import numpy as np

class StandardScaler():
    def __init__(self):
        pass

    def fit(self,X):
        self.mean=X.mean(axis=0)  # S1
        self.std=X.std(axis=0)  # S2

    def transform(self,X):
        return (X-self.mean)/self.std  # S3

    def inverse_transform(self,X):
        return X*self.std+self.mean  # S4

class PolynomialFeatures():
    def __init__(self,degree,include_bias=False):
        self.degree=degree
        self.include_bias=include_bias

    def fit_transform(self,X):
        if self.include_bias:
            out=np.ones((len(X),self.degree+1))  # S5
            for i in range(self.degree+1):
              out[:,i]=X**i  # S6

        else:
            out=np.zeros((len(X),self.degree))  # S7
            for i in range(self.degree):
              out[:,i]=X**(i+1)  # S8

        return out  # S9`

const overfittingCode = `import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

deg8=PolynomialFeatures(8,include_bias=True)  # S1
deg8_train_feats=deg8.fit_transform(scaled_Xtrain)  # S2
deg8_test_feats=deg8.fit_transform(scaled_Xtest)  # S3

deg8_model=LinearRegression(fit_intercept=False)  # S4
deg8_model.fit(deg8_train_feats,ytrain)  # S5

deg8_train_predictions=deg8_model.predict(deg8_train_feats)  # S6
deg8_test_predictions=deg8_model.predict(deg8_test_feats)  # S7`

const gradientDescentCode = `import numpy as np

class GDRegressor():
    def __init__(self,learning_rate,max_iter):
        self.lr=learning_rate
        self.max_iter=max_iter

    def fit(self,X,y):
        self.coef=np.ones((X.shape[1],))  # S1
        self.intercept=1  # S2
        for i in range(self.max_iter):
            residuals=self.predict(X)-y  # S3
            coef_grad=(X.T)@residuals/len(X)  # S4
            intercept_grad=np.mean(residuals)  # S5
            self.coef-=self.lr*coef_grad  # S6
            self.intercept-=self.lr*intercept_grad  # S7

    def predict(self,X):
        return X@self.coef+self.intercept  # S8`

export const linearRegressionNotebookLab: CodeLabQuestionSpec = defineCodeLabQuestion({
  id: 'linear-regression-notebook-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Trace the Normal-Equation Model',
  prompt:
    'Use the completed `5LinearRegression.ipynb` notebook to trace its normal-equation implementation, then diagnose a prediction method that drops the fitted intercept.',
  instructions:
    'The displayed class is the notebook implementation with statement labels added. Run the fixture and invocation conceptually from top to bottom. Treat values shown as integers as their floating-point equivalents.',
  datasetId: 'linear-regression-notebook-v1',
  variantId: 'linear-regression-notebook-v1',
  language: 'python',
  code: linearRegressionCode,
  fixtureTitle: 'Deterministic line with a nonzero intercept',
  fixtureHeading: 'Trace data',
  fixture: `X_trace = np.array([
    [0.0],
    [1.0],
    [2.0],
])
y_trace = np.array([1.0, 3.0, 5.0])`,
  invocationTitle: 'Notebook commands to trace',
  invocationLead: 'After defining the class and trace arrays, Python evaluates:',
  invocation: `model = LinearRegression()
model.fit(X_trace, y_trace)

coef = model.coef.tolist()
intercept = model.intercept
predictions = model.predict(
    np.array([[3.0], [4.0]])
).tolist()`,
  hintSchedule: [2, 4, 6],
  hints: [
    'S1 and S2 create the design matrix `[[1, 0], [1, 1], [1, 2]]`; the first coefficient therefore belongs to the intercept column.',
    'The three target values lie exactly on `y = 2x + 1`.',
    'A test at `x = 0` isolates the intercept, while a difference between two predictions isolates the slope.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Trace the normal-equation fit',
      prompt:
        'Mentally execute the notebook class on the trace data. What parameter values and predictions are produced?',
      successCopy:
        'Correct: the normal equation recovers the exact line `y = 2x + 1`, so the two new predictions are 7 and 9.',
      fields: [
        {
          id: 'COEF',
          label: 'coef',
          correctOptionId: 'COEF_2',
          options: [
            { id: 'COEF_2', label: '[2.0]' },
            { id: 'COEF_1', label: '[1.0]' },
            { id: 'COEF_2_1', label: '[2.0, 1.0]' },
          ],
        },
        {
          id: 'INTERCEPT',
          label: 'intercept',
          correctOptionId: 'INTERCEPT_1',
          options: [
            { id: 'INTERCEPT_1', label: '1.0' },
            { id: 'INTERCEPT_0', label: '0.0' },
            { id: 'INTERCEPT_2', label: '2.0' },
          ],
        },
        {
          id: 'PREDICTIONS',
          label: 'predictions',
          correctOptionId: 'PREDICTIONS_7_9',
          options: [
            { id: 'PREDICTIONS_7_9', label: '[7.0, 9.0]' },
            { id: 'PREDICTIONS_6_8', label: '[6.0, 8.0]' },
            { id: 'PREDICTIONS_NESTED', label: '[[7.0], [9.0]]' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Read a changed prediction method',
      prompt:
        'Suppose S6 were changed to `return x@self.coef`, leaving the fitted parameters unchanged. What would `predictions` become?',
      successCopy:
        'Correct: the mutant retains the fitted slope of 2 but omits the fitted intercept of 1, lowering every prediction by one.',
      fields: [
        {
          id: 'MUTATED_PREDICTIONS',
          label: 'predictions after the change',
          correctOptionId: 'MUTATED_6_8',
          options: [
            { id: 'MUTATED_6_8', label: '[6.0, 8.0]' },
            { id: 'MUTATED_7_9', label: '[7.0, 9.0]' },
            { id: 'MUTATED_3_4', label: '[3.0, 4.0]' },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Choose a distinguishing test',
      prompt:
        'Which assertion passes for the notebook implementation but fails when S6 omits the intercept?',
      successCopy:
        'Correct: at `x = 0`, the notebook returns the fitted intercept 1, whereas the mutant always returns zero.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'TEST_INTERCEPT',
          options: [
            {
              id: 'TEST_INTERCEPT',
              label: 'Check the prediction at zero',
              description:
                '`assert np.allclose(model.predict(np.array([[0.0]])), [1.0])`',
            },
            {
              id: 'TEST_COEF_SHAPE',
              label: 'Check only the coefficient shape',
              description: '`assert model.coef.shape == (1,)`',
            },
            {
              id: 'TEST_OUTPUT_SHAPE',
              label: 'Check only the prediction shape',
              description:
                '`assert model.predict(np.array([[3.0], [4.0]])).shape == (2,)`',
            },
            {
              id: 'TEST_SLOPE_DIFFERENCE',
              label: 'Check only the change between predictions',
              description:
                '`assert np.allclose(np.diff(model.predict(np.array([[3.0], [4.0]]))), [2.0])`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'The leading ones in `Xnew` create an intercept coefficient. S3 solves for `[1, 2]`, S4 stores 1 as the intercept, and S5 stores `[2]` as the slope array. The notebook prediction method uses both quantities. Removing the intercept preserves output shapes and differences between predictions, so a prediction at zero distinguishes the two versions.',
  },
  successCopy:
    'Notebook Lab complete: the normal-equation fit, intercept mutation, and distinguishing test are all correct.',
})

export const polynomialRegressionNotebookLab: CodeLabQuestionSpec = defineCodeLabQuestion({
  id: 'polynomial-regression-notebook-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Trace Scaling and Polynomial Features',
  prompt:
    'Use the completed `6PolynomialRegression.ipynb` notebook to trace its `StandardScaler` and `PolynomialFeatures` implementations.',
  instructions:
    'The displayed classes reproduce the notebook code with statement labels added. The two trace values were chosen so their population mean and standard deviation are exact integers.',
  datasetId: 'polynomial-regression-notebook-v1',
  variantId: 'polynomial-regression-notebook-v1',
  language: 'python',
  code: polynomialRegressionCode,
  fixtureTitle: 'Two values with mean 2 and standard deviation 1',
  fixtureHeading: 'Trace data',
  fixture: `X_trace = np.array([1.0, 3.0])`,
  invocationTitle: 'Notebook commands to trace',
  invocationLead: 'After defining the classes and trace array, Python evaluates:',
  invocation: `scaler = StandardScaler()
scaler.fit(X_trace)
scaled = scaler.transform(X_trace)

poly = PolynomialFeatures(3)
features = poly.fit_transform(scaled)
restored = scaler.inverse_transform(scaled)`,
  hintSchedule: [2, 4, 6],
  hints: [
    'For `[1, 3]`, S1 stores 2 and S2 stores the population standard deviation 1.',
    'Because `include_bias` uses its default value, S7 creates three columns and S8 fills them with powers 1, 2, and 3.',
    'The mutation begins with power zero, so both entries in its first column are 1.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Trace scaling and feature engineering',
      prompt:
        'Mentally execute the notebook classes. What are the scaled values, cubic feature matrix, and restored values?',
      successCopy:
        'Correct: scaling maps 1 and 3 to -1 and 1, powers 1 through 3 fill the three feature columns, and inverse transformation recovers the original values.',
      fields: [
        {
          id: 'SCALED',
          label: 'scaled.tolist()',
          correctOptionId: 'SCALED_NEG1_1',
          options: [
            { id: 'SCALED_NEG1_1', label: '[-1.0, 1.0]' },
            { id: 'SCALED_0_1', label: '[0.0, 1.0]' },
            { id: 'SCALED_NEG2_2', label: '[-2.0, 2.0]' },
          ],
        },
        {
          id: 'FEATURES',
          label: 'features.tolist()',
          correctOptionId: 'FEATURES_POWERS_1_TO_3',
          options: [
            {
              id: 'FEATURES_POWERS_1_TO_3',
              label: '[[-1.0, 1.0, -1.0], [1.0, 1.0, 1.0]]',
            },
            {
              id: 'FEATURES_POWERS_0_TO_2',
              label: '[[1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]',
            },
            {
              id: 'FEATURES_ONLY_TWO_COLUMNS',
              label: '[[-1.0, 1.0], [1.0, 1.0]]',
            },
          ],
        },
        {
          id: 'RESTORED',
          label: 'restored.tolist()',
          correctOptionId: 'RESTORED_1_3',
          options: [
            { id: 'RESTORED_1_3', label: '[1.0, 3.0]' },
            { id: 'RESTORED_NEG1_1', label: '[-1.0, 1.0]' },
            { id: 'RESTORED_0_4', label: '[0.0, 4.0]' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Read a changed exponent',
      prompt:
        'Suppose S8 were changed from `out[:,i]=X**(i+1)` to `out[:,i]=X**i`. What would `features.tolist()` become?',
      successCopy:
        'Correct: the changed loop creates powers zero, one, and two instead of powers one, two, and three.',
      fields: [
        {
          id: 'MUTATED_FEATURES',
          label: 'features.tolist() after the change',
          correctOptionId: 'MUTATED_POWERS_0_TO_2',
          options: [
            {
              id: 'MUTATED_POWERS_0_TO_2',
              label: '[[1.0, -1.0, 1.0], [1.0, 1.0, 1.0]]',
            },
            {
              id: 'MUTATED_UNCHANGED',
              label: '[[-1.0, 1.0, -1.0], [1.0, 1.0, 1.0]]',
            },
            {
              id: 'MUTATED_TWO_COLUMNS',
              label: '[[1.0, -1.0], [1.0, 1.0]]',
            },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Choose a distinguishing test',
      prompt:
        'Which assertion passes for the notebook implementation but fails when S8 starts at power zero?',
      successCopy:
        'Correct: without a bias column, the reference first feature must equal the original input. The mutant instead fills that column with ones.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'TEST_FIRST_COLUMN',
          options: [
            {
              id: 'TEST_FIRST_COLUMN',
              label: 'Check that the first feature is the input',
              description: '`assert np.allclose(features[:, 0], scaled)`',
            },
            {
              id: 'TEST_SHAPE',
              label: 'Check only the matrix shape',
              description: '`assert features.shape == (2, 3)`',
            },
            {
              id: 'TEST_POSITIVE_ROW',
              label: 'Check the row produced by positive one',
              description: '`assert np.allclose(features[1], [1.0, 1.0, 1.0])`',
            },
            {
              id: 'TEST_FINITE',
              label: 'Check only that all entries are finite',
              description: '`assert np.all(np.isfinite(features))`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'The scaler stores mean 2 and standard deviation 1, so its output is `[-1, 1]`. With `include_bias=False`, the notebook allocates exactly `degree` columns and S8 starts at exponent one. Replacing `i+1` with `i` silently inserts a bias column and drops the cubic column without changing the matrix shape.',
  },
  successCopy:
    'Notebook Lab complete: scaling, polynomial feature construction, mutation diagnosis, and testing are all correct.',
})

export const overfittingNotebookLab: CodeLabQuestionSpec = defineCodeLabQuestion({
  id: 'overfitting-notebook-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Trace Degree-Eight Features',
  prompt:
    'Use the completed `7Overfitting.ipynb` notebook to trace the degree-eight feature matrix used by its sklearn linear model.',
  instructions:
    'The displayed statements reproduce the notebook degree-eight block with statement labels added. This deterministic trace isolates that runnable block: the notebook later calls an undefined `PolynomialRegression` in its metrics loop, and its `test_size=0.8` call places 80% of the data in the test set.',
  datasetId: 'overfitting-notebook-v1',
  variantId: 'overfitting-notebook-v1',
  language: 'python',
  code: overfittingCode,
  fixtureTitle: 'Deterministic scaled train and test arrays',
  fixtureHeading: 'Trace data',
  fixture: `scaled_Xtrain = np.array([
    [-1.0],
    [ 0.0],
    [ 2.0],
])
ytrain = np.array([1.0, 2.0, 5.0])
scaled_Xtest = np.array([[1.0]])`,
  invocationTitle: 'Feature expressions to inspect',
  invocationLead: 'After running S1 through S7 on the fixture, Python evaluates:',
  invocation: `feature_shape = deg8_train_feats.shape
negative_row = deg8_train_feats[0, :].tolist()
positive_row = deg8_train_feats[2, :].tolist()`,
  hintSchedule: [2, 4, 6],
  hints: [
    'A degree-eight transformer with `include_bias=True` makes columns for exponents zero through eight, for nine columns total.',
    'Powers of -1 alternate between -1 and 1, while the zero-degree column is always 1.',
    'Turning off `include_bias` removes only the exponent-zero column; the final `x**8` column remains.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Trace the degree-eight feature matrix',
      prompt:
        'Mentally execute the displayed sklearn transformation. What shape and two feature rows are produced?',
      successCopy:
        'Correct: `include_bias=True` produces nine columns ordered as `1, x, x**2, ..., x**8`.',
      fields: [
        {
          id: 'FEATURE_SHAPE',
          label: 'feature_shape',
          correctOptionId: 'SHAPE_3_9',
          options: [
            { id: 'SHAPE_3_9', label: '(3, 9)' },
            { id: 'SHAPE_3_8', label: '(3, 8)' },
            { id: 'SHAPE_9_3', label: '(9, 3)' },
          ],
        },
        {
          id: 'NEGATIVE_ROW',
          label: 'negative_row',
          correctOptionId: 'NEGATIVE_WITH_BIAS',
          options: [
            {
              id: 'NEGATIVE_WITH_BIAS',
              label: '[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]',
            },
            {
              id: 'NEGATIVE_WITHOUT_BIAS',
              label: '[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]',
            },
            {
              id: 'NEGATIVE_ALL_ONES',
              label: '[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]',
            },
          ],
        },
        {
          id: 'POSITIVE_ROW',
          label: 'positive_row',
          correctOptionId: 'POSITIVE_WITH_BIAS',
          options: [
            {
              id: 'POSITIVE_WITH_BIAS',
              label: '[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]',
            },
            {
              id: 'POSITIVE_WITHOUT_BIAS',
              label: '[2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]',
            },
            {
              id: 'POSITIVE_EVEN_SEQUENCE',
              label: '[1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]',
            },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Remove the explicit bias feature',
      prompt:
        'Suppose S1 were changed to `PolynomialFeatures(8,include_bias=False)`. What would the feature shape and `negative_row` become?',
      successCopy:
        'Correct: disabling the bias removes the leading exponent-zero column, leaving powers one through eight.',
      fields: [
        {
          id: 'MUTATED_SUMMARY',
          label: '(feature_shape, negative_row) after the change',
          correctOptionId: 'MUTATED_NO_BIAS',
          options: [
            {
              id: 'MUTATED_NO_BIAS',
              label:
                '((3, 8), [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])',
            },
            {
              id: 'MUTATED_UNCHANGED',
              label:
                '((3, 9), [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])',
            },
            {
              id: 'MUTATED_DROP_HIGHEST',
              label:
                '((3, 8), [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])',
            },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Choose a distinguishing test',
      prompt:
        'Which assertion passes for the notebook feature construction but fails when `include_bias=False`?',
      successCopy:
        'Correct: only the reference matrix has nine columns. Both versions retain three rows, finite entries, and the final value `2**8 = 256`.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'TEST_NINE_COLUMNS',
          options: [
            {
              id: 'TEST_NINE_COLUMNS',
              label: 'Check for the bias column in the shape',
              description: '`assert deg8_train_feats.shape == (3, 9)`',
            },
            {
              id: 'TEST_THREE_ROWS',
              label: 'Check only the number of observations',
              description: '`assert deg8_train_feats.shape[0] == 3`',
            },
            {
              id: 'TEST_HIGHEST_POWER',
              label: 'Check only the highest power',
              description: '`assert deg8_train_feats[-1, -1] == 256`',
            },
            {
              id: 'TEST_FINITE_FEATURES',
              label: 'Check only that all entries are finite',
              description: '`assert np.all(np.isfinite(deg8_train_feats))`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'The notebook asks sklearn for all polynomial powers through degree eight and explicitly includes the constant column, so the feature matrix has nine columns. Its linear model correspondingly uses `fit_intercept=False`. Disabling the bias removes the first column but does not remove `x**8`. Separately, the later metrics loop cannot execute until `PolynomialRegression` is defined or replaced with the imported sklearn classes.',
  },
  successCopy:
    'Notebook Lab complete: degree-eight features, explicit bias handling, mutation diagnosis, and testing are all correct.',
})

export const gradientDescentNotebookLab: CodeLabQuestionSpec = defineCodeLabQuestion({
  id: 'gradient-descent-notebook-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Trace One Gradient Step',
  prompt:
    'Use the completed `8GradientDescent.ipynb` notebook to trace one iteration of its `GDRegressor` and diagnose a reversed residual.',
  instructions:
    'The displayed class is the notebook implementation with statement labels added. With `max_iter=1`, begin from the coefficient and intercept values assigned at S1 and S2 and execute exactly one update.',
  datasetId: 'gradient-descent-notebook-v1',
  variantId: 'gradient-descent-notebook-v1',
  language: 'python',
  code: gradientDescentCode,
  fixtureTitle: 'Two observations for one gradient step',
  fixtureHeading: 'Trace data',
  fixture: `X_trace = np.array([
    [0.0],
    [1.0],
])
y_trace = np.array([1.0, 3.0])`,
  invocationTitle: 'Notebook commands to trace',
  invocationLead: 'After defining the class and trace arrays, Python evaluates:',
  invocation: `model = GDRegressor(learning_rate=0.1, max_iter=1)
model.fit(X_trace, y_trace)

coef = model.coef.tolist()
intercept = model.intercept
predictions = model.predict(X_trace).tolist()`,
  hintSchedule: [2, 4, 6],
  hints: [
    'Before the update, S1 and S2 make the predictions `[1, 2]`, so S3 gives residuals `[0, -1]`.',
    'S4 and S5 both equal `-0.5`; subtracting `0.1 * -0.5` adds 0.05 to each parameter.',
    'Reversing the residual reverses both gradients while S6 and S7 still subtract them, so that mutation moves uphill on this fixture.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Trace one notebook iteration',
      prompt:
        'Mentally execute one iteration. What are the initial residuals, updated parameters, and predictions after the update?',
      successCopy:
        'Correct: both gradients are -0.5, so one step raises the coefficient and intercept from 1 to 1.05 and produces predictions 1.05 and 2.10.',
      fields: [
        {
          id: 'INITIAL_RESIDUALS',
          label: 'residuals at S3',
          correctOptionId: 'RESIDUALS_0_NEG1',
          options: [
            { id: 'RESIDUALS_0_NEG1', label: '[0.0, -1.0]' },
            { id: 'RESIDUALS_0_1', label: '[0.0, 1.0]' },
            { id: 'RESIDUALS_NEG1_NEG2', label: '[-1.0, -2.0]' },
          ],
        },
        {
          id: 'UPDATED_PARAMETERS',
          label: '(coef, intercept)',
          correctOptionId: 'PARAMETERS_1_05',
          options: [
            { id: 'PARAMETERS_1_05', label: '([1.05], 1.05)' },
            { id: 'PARAMETERS_0_95', label: '([0.95], 0.95)' },
            { id: 'PARAMETERS_1_10', label: '([1.10], 1.10)' },
          ],
        },
        {
          id: 'PREDICTIONS',
          label: 'predictions',
          correctOptionId: 'PREDICTIONS_1_05_2_10',
          options: [
            { id: 'PREDICTIONS_1_05_2_10', label: '[1.05, 2.10]' },
            { id: 'PREDICTIONS_0_95_1_90', label: '[0.95, 1.90]' },
            { id: 'PREDICTIONS_1_05_3_10', label: '[1.05, 3.10]' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Reverse the residual',
      prompt:
        'Suppose S3 were changed to `residuals=y-self.predict(X)`, while S6 and S7 still subtract the gradients. What parameters and predictions would result?',
      successCopy:
        'Correct: reversing the residual makes both gradients positive 0.5, so the unchanged subtraction updates both parameters to 0.95.',
      fields: [
        {
          id: 'MUTATED_RESULT',
          label: '(coef, intercept, predictions) after the change',
          correctOptionId: 'MUTATED_0_95',
          options: [
            {
              id: 'MUTATED_0_95',
              label: '([0.95], 0.95, [0.95, 1.90])',
            },
            {
              id: 'MUTATED_1_05',
              label: '([1.05], 1.05, [1.05, 2.10])',
            },
            {
              id: 'MUTATED_UNCHANGED',
              label: '([1.0], 1.0, [1.0, 2.0])',
            },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Choose a distinguishing test',
      prompt:
        'Which test passes for the notebook update but fails for the reversed-residual mutation?',
      successCopy:
        'Correct: the notebook step lowers MSE from 0.5 to 0.40625, while the reversed-residual mutation raises it to 0.60625.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'TEST_MSE_DECREASES',
          options: [
            {
              id: 'TEST_MSE_DECREASES',
              label: 'Check that one step lowers MSE',
              description:
                '`assert ((model.predict(X_trace) - y_trace) ** 2).mean() < 0.5`',
            },
            {
              id: 'TEST_COEF_SHAPE',
              label: 'Check only the coefficient shape',
              description: '`assert model.coef.shape == (1,)`',
            },
            {
              id: 'TEST_OUTPUT_SHAPE',
              label: 'Check only the prediction shape',
              description: '`assert model.predict(X_trace).shape == (2,)`',
            },
            {
              id: 'TEST_FINITE',
              label: 'Check only that predictions are finite',
              description: '`assert np.all(np.isfinite(model.predict(X_trace)))`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'The notebook defines residual as prediction minus target and subtracts its resulting gradients. On the fixture, the initial predictions are `[1, 2]`, the residuals are `[0, -1]`, and both gradients are -0.5. The reference therefore moves both parameters upward to 1.05 and reduces MSE. Reversing only the residual reverses the update direction and increases MSE.',
  },
  successCopy:
    'Notebook Lab complete: the gradient step, reversed-residual mutation, and distinguishing test are all correct.',
})
