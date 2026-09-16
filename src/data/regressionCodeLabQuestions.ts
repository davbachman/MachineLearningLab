import { defineCodeLabQuestion, type CodeLabQuestionSpec, type CodeLabStageSpec } from '../lib/codeLab'

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
    'Use the completed `4LinearRegression.ipynb` notebook to trace its normal-equation implementation, then diagnose a prediction method that drops the fitted intercept.',
  instructions:
    'The displayed class is the notebook implementation with statement labels added. Work through one calculation at a time and check each answer before continuing. Array answers are displayed as lists; fractions represent exact values. Ignore floating-point roundoff.',
  datasetId: 'linear-regression-notebook-v3',
  variantId: 'linear-regression-notebook-v3',
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
      title: '1. Compute Xnew.T @ Xnew',
      prompt:
        'During model.fit(X_trace, y_trace), X and y refer to X_trace and y_trace. Trace S1 and S2 to construct the local variable Xnew, then evaluate Xnew.T @ Xnew, the first matrix expression in S3.',
      successCopy:
        'Correct: Xnew has rows [1, 0], [1, 1], and [1, 2]. Taking dot products of its columns gives [[3, 3], [3, 5]].',
      fields: [
        {
          id: 'GRAM',
          label: '(Xnew.T @ Xnew).tolist()',
          correctOptionId: 'GRAM_3_3_3_5',
          options: [
            { id: 'GRAM_3_3_3_5', label: '[[3.0, 3.0], [3.0, 5.0]]' },
            { id: 'GRAM_REVERSED', label: '[[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1.0, 3.0, 5.0]]' },
            { id: 'GRAM_NO_CROSS', label: '[[3.0, 0.0], [0.0, 5.0]]' },
          ],
        },
      ],
    },
    {
      id: 'INVERSE',
      kind: 'executionTrace',
      title: '2. Invert Xnew.T @ Xnew',
      prompt: 'Now evaluate the matrix inverse used in S3. This is a matrix inverse, not the elementwise reciprocal.',
      successCopy: 'Correct: the determinant is 3·5 − 3·3 = 6. The inverse is (1/6) times [[5, -3], [-3, 3]].',
      fields: [
        {
          id: 'GRAM_INVERSE',
          label: 'np.linalg.inv(Xnew.T @ Xnew).tolist()',
          correctOptionId: 'INVERSE_5_6',
          options: [
            { id: 'INVERSE_5_6', label: '[[5/6, -1/2], [-1/2, 1/2]]' },
            { id: 'INVERSE_RECIPROCALS', label: '[[1/3, 1/3], [1/3, 1/5]]' },
            { id: 'INVERSE_NO_DETERMINANT', label: '[[5.0, -3.0], [-3.0, 3.0]]' },
          ],
        },
      ],
    },
    {
      id: 'TARGET_PRODUCT',
      kind: 'executionTrace',
      title: '3. Compute Xnew.T @ y',
      prompt: 'Evaluate the other matrix product in S3 using y = y_trace. Pay attention to the shape of y.',
      successCopy: 'Correct: the entries are 1 + 3 + 5 = 9 and 0·1 + 1·3 + 2·5 = 13. Since y is one-dimensional, the result is also one-dimensional.',
      fields: [
        {
          id: 'XT_Y',
          label: '(Xnew.T @ y).tolist()',
          correctOptionId: 'XT_Y_9_13',
          options: [
            { id: 'XT_Y_9_13', label: '[9.0, 13.0]' },
            { id: 'XT_Y_REVERSED', label: '[13.0, 9.0]' },
            { id: 'XT_Y_COLUMN', label: '[[9.0], [13.0]]' },
          ],
        },
      ],
    },
    {
      id: 'COEFFICIENTS',
      kind: 'executionTrace',
      title: '4. Compute coeffs',
      prompt: 'Combine the inverse and the target product to evaluate the full expression in S3. What is the local variable coeffs immediately after S3?',
      successCopy: 'Correct: multiplying [[5/6, -1/2], [-1/2, 1/2]] by [9, 13] gives [1, 2]. The first entry multiplies the column of ones.',
      fields: [
        {
          id: 'COEFFS',
          label: 'coeffs.tolist() after S3',
          correctOptionId: 'COEFFS_1_2',
          options: [
            { id: 'COEFFS_1_2', label: '[1.0, 2.0]' },
            { id: 'COEFFS_REVERSED', label: '[2.0, 1.0]' },
            { id: 'COEFFS_NO_INVERSE', label: '[9.0, 13.0]' },
          ],
        },
      ],
    },
    {
      id: 'SLOPE',
      kind: 'executionTrace',
      title: '5. Read the stored slope',
      prompt: 'Trace S5, then the invocation coef = model.coef.tolist(). What is coef? Notice the slice coeffs[1:].',
      successCopy: 'Correct: coeffs[1:] is the one-element array containing the slope 2, so coef is [2.0], not a scalar.',
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
      ],
    },
    {
      id: 'INTERCEPT',
      kind: 'executionTrace',
      title: '6. Read the stored intercept',
      prompt: 'Trace S4, then the invocation intercept = model.intercept. What is intercept?',
      successCopy: 'Correct: coeffs[0] is the scalar 1.0, the coefficient of the column of ones in Xnew.',
      fields: [
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
      ],
    },
    {
      id: 'PREDICT',
      kind: 'executionTrace',
      title: '7. Predict at two new inputs',
      prompt: 'Use the fitted parameters to trace S6 for the invocation model.predict(np.array([[3.0], [4.0]])). What is predictions after .tolist()?',
      successCopy: 'Correct: 3·2 + 1 = 7 and 4·2 + 1 = 9. The returned array is one-dimensional, so predictions is [7.0, 9.0].',
      fields: [
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
      title: '8. Read a changed prediction method',
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
      title: '9. Choose a distinguishing test',
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
    'Use the completed `5PolynomialRegression.ipynb` notebook to trace its `StandardScaler` and `PolynomialFeatures` implementations.',
  instructions:
    'The displayed classes reproduce the notebook code with statement labels added. The two trace values were chosen so their population mean and standard deviation are exact integers.',
  datasetId: 'polynomial-regression-notebook-v2',
  variantId: 'polynomial-regression-notebook-v2',
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
      title: '1. Scale the inputs',
      prompt:
        'Run scaler.fit(X_trace), then scaler.transform(X_trace). What is scaled.tolist()?',
      successCopy:
        'Correct: the mean is 2 and the population standard deviation is 1, so scaling produces [-1.0, 1.0].',
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
      ],
    },
    {
      id: 'FEATURES', kind: 'executionTrace', title: '2. Build the powers',
      prompt: 'Continue with poly.fit_transform(scaled). Use the default include_bias=False.',
      successCopy: 'Correct: the columns are scaled, scaled**2, and scaled**3, with one row per observation.',
      fields: [
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
      ],
    },
    {
      id: 'RESTORE', kind: 'executionTrace', title: '3. Undo scaling',
      prompt: 'Now evaluate scaler.inverse_transform(scaled).',
      successCopy: 'Correct: multiply by the stored standard deviation 1, then add the stored mean 2, recovering [1.0, 3.0].',
      fields: [
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
      title: '4. Read a changed exponent',
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
      title: '5. Choose a distinguishing test',
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

function evaluationCheckpoint(id: string, title: string, prompt: string, label: string, options: string[], successCopy: string, kind: CodeLabStageSpec['kind'] = 'executionTrace'): CodeLabStageSpec {
  return { id, title, prompt, kind, successCopy,
    fields: [{ id: 'ANSWER', label, correctOptionId: `${id}_0`, options: options.map((label,i) => ({ id: `${id}_${i}`, label })) }] }
}

export const polynomialEvaluationNotebookLab: CodeLabQuestionSpec = defineCodeLabQuestion({
  id: 'polynomial-evaluation-notebook-lab', kind: 'codeLab',
  title: 'Notebook Lab: Fit, Predict, and Evaluate',
  prompt: 'Use the from-scratch classes in 5PolynomialRegression.ipynb to trace a quadratic fit and its held-out errors.',
  instructions: 'The displayed code isolates the degree-2 iteration of the notebook metrics loop, then uses its plotting-order operation. First define PolynomialFeatures and LinearRegression from the notebook—not sklearn. The fixture arrays are already scaled in one common coordinate system. Ignore floating-point roundoff. Array answers are shown as lists.',
  datasetId: 'polynomial-evaluation-v1', variantId: 'polynomial-evaluation-v1', language: 'python',
  code: `# Use the custom classes defined in the notebook.
poly=PolynomialFeatures(2)  # S1
train_feats=poly.fit_transform(scaled_Xtrain)  # S2
val_feats=poly.fit_transform(scaled_Xval)  # S3
model=LinearRegression()
model.fit(train_feats,ytrain)  # S4
train_predictions=model.predict(train_feats)  # S5
val_predictions=model.predict(val_feats)  # S6
MSEtrain=((train_predictions-ytrain)**2).mean()  # S7
MSEval=((val_predictions-yval)**2).mean()  # S8

order=np.argsort(scaled_Xtrain)  # S9`,
  fixtureTitle: 'Already-scaled inputs and targets', fixtureHeading: 'Trace data',
  fixture: `scaled_Xtrain=np.array([2.0, -1.0, 1.0, 0.0])
ytrain=np.array([5.0, 2.0, 2.0, 1.0])
scaled_Xval=np.array([0.5, 1.5])
yval=np.array([1.0, 3.0])`,
  invocationTitle: 'Values to inspect after S1–S9', invocationLead: 'After defining the two custom classes, run the fixture, then S1–S9. Inspect:',
  invocation: `train_feats.shape
val_feats.tolist()
model.coef.tolist()
model.intercept
val_predictions.tolist()
MSEtrain
MSEval
order.tolist()`,
  hintSchedule: [2,4,6], hints: [
    'The two engineered columns are x and x². LinearRegression adds the column of ones itself.',
    'Every training target equals 1 + x². The two validation residuals both have magnitude 0.25.',
    'Square residuals before averaging. Sorting must apply the same index order to inputs and predictions.',
  ],
  stages: [
    evaluationCheckpoint('TRACE', '1. Count the feature columns', 'Trace S1 and S2 with the default bias setting.', 'train_feats.shape', ['(4, 2)', '(4, 3)', '(2, 4)'], 'Correct: four observations and two columns, x and x².'),
    evaluationCheckpoint('VAL_FEATURES', '2. Transform the validation inputs', 'Trace S3 without fitting any new scaling parameters.', 'val_feats.tolist()', ['[[0.5, 0.25], [1.5, 2.25]]', '[[1.0, 0.5, 0.25], [1.0, 1.5, 2.25]]', '[[0.5, 1.0], [1.5, 3.0]]'], 'Correct: each row contains one scaled input and its square.'),
    evaluationCheckpoint('COEF', '3. Read the fitted coefficients', 'Trace S4 using the custom normal-equation model.', 'model.coef.tolist()', ['[0.0, 1.0]', '[1.0, 0.0]', '[1.0, 0.0, 1.0]'], 'Correct: y = 1 + x² has coefficient 0 for x and coefficient 1 for x²; the intercept is stored separately.'),
    evaluationCheckpoint('INTERCEPT', '4. Read the intercept', 'What scalar does S4 store separately from model.coef?', 'model.intercept', ['1.0', '0.0', '2.0'], 'Correct: the constant term of 1 + x² is 1.'),
    evaluationCheckpoint('PREDICT', '5. Predict the validation targets', 'Trace S6 using the same fitted parameters.', 'val_predictions.tolist()', ['[1.25, 3.25]', '[0.25, 2.25]', '[1.0, 3.0]'], 'Correct: 1 + 0.5² = 1.25 and 1 + 1.5² = 3.25. Predictions need not equal the held-out targets.'),
    evaluationCheckpoint('TRAIN_MSE', '6. Measure training error', 'Trace S5 and S7. Ignore roundoff.', 'MSEtrain', ['0.0', '0.0625', '1.0'], 'Correct: all four training targets lie exactly on the fitted quadratic.'),
    evaluationCheckpoint('VAL_MSE', '7. Measure validation error', 'Trace S8; the validation targets are [1.0, 3.0].', 'MSEval', ['0.0625', '0.125', '0.25'], 'Correct: the two squared errors are 0.0625 and 0.0625, whose mean is 0.0625. Zero training error does not imply zero validation error.'),
    evaluationCheckpoint('ORDER', '8. Put the curve in plotting order', 'Trace S9. Use this same order for both scaled_Xtrain and train_predictions.', 'order.tolist()', ['[1, 3, 2, 0]', '[-1.0, 0.0, 1.0, 2.0]', '[0, 1, 2, 3]'], 'Correct: argsort returns indices, not sorted values. Applying [1, 3, 2, 0] to both arrays preserves every input/prediction pair.'),
    evaluationCheckpoint('MUTATION', '9. Replace mean with sum', 'Suppose S8 uses .sum() instead of .mean(), with everything else unchanged.', 'MSEval after this change', ['0.125', '0.0625', '0.25'], 'Correct: the sum is 0.125. Despite the variable name, this changed code computes RSS rather than MSE.', 'diagnoseMutation'),
    evaluationCheckpoint('TEST', '10. Detect the changed reduction', 'Using the original fixture, which assertion passes for the original S8 but fails for the sum mutation? Use np.isclose to allow roundoff.', 'Assertion', ['assert np.isclose(MSEval, 0.0625)', 'assert MSEval >= 0', 'assert np.isfinite(MSEval)'], 'Correct: the expected numerical mean distinguishes the two reductions; positivity and finiteness do not.', 'distinguishingTest'),
  ],
  reveal: { explanation: 'The from-scratch model fits 1 + x². Validation error averages squared residuals on held-out targets. Sorting for plotting changes only display order, not the fitted parameters or errors.' },
  successCopy: 'Notebook Lab complete: feature construction, fitting, predictions, MSE, plotting order, and the changed reduction are all checked.',
})

export const gradientDescentNotebookLab: CodeLabQuestionSpec = defineCodeLabQuestion({
  id: 'gradient-descent-notebook-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Trace One Gradient Step',
  prompt:
    'Use the completed `6GradientDescent.ipynb` notebook to trace one iteration of its `GDRegressor` and diagnose a reversed residual.',
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
