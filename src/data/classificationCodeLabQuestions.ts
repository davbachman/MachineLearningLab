import { defineCodeLabQuestion, type CodeLabQuestionSpec } from '../lib/codeLab'

const batchGradientDescentCode = `import numpy as np

class SGDRegressor():
    def __init__(self,learning_rate, max_iter, batch_size):
        self.lr=learning_rate
        self.max_iter=max_iter #number of epochs
        self.batch_size=batch_size

    def fit(self,X,y):
        self.coef=np.ones((X.shape[1],)) #initial values
        self.intercept=1 #initial value
        indices=np.arange(len(X)) # S1
        for i in range(self.max_iter):
            np.random.seed(i) #Just so everyone gets the same answer! # S2
            np.random.shuffle(indices) # S3
            X_shuffle=X[indices]
            y_shuffle=y[indices]
            for j in range(0,len(X),self.batch_size):
                X_batch=X_shuffle[j:j+self.batch_size]
                y_batch=y_shuffle[j:j+self.batch_size]
                residuals=self.predict(X_batch)-y_batch
                #coef_grad=(X_batch.T)@residuals/len(X_batch)
                coef_grad=(X_batch.T)@residuals/len(X) # S4
                #intercept_grad=np.mean(residuals)
                intercept_grad=np.sum(residuals)/len(X) # S5
                self.coef-=self.lr*coef_grad
                self.intercept-=self.lr*intercept_grad

    def predict(self,X):
        return X@self.coef+self.intercept`

const regularizationCode = `import numpy as np

class SGDRegressor():
    def __init__(self,learning_rate, max_iter, batch_size, penalty='l2', alpha=0.0001):
        self.lr=learning_rate
        self.max_iter=max_iter #number of epochs
        self.batch_size=batch_size
        self.penalty=penalty #either 'l1' or 'l2'
        self.alpha=alpha #amount of regularization to apply

    def fit(self,X,y):
        self.coef=np.ones((X.shape[1],)) #Initial values
        self.intercept=1 #Initial value
        if self.penalty=='l1':
            #penalty_grad=lambda x:2*(x>0)-1
            penalty_grad=lambda x:np.sign(x) # S1
        elif self.penalty=='l2':
            penalty_grad=lambda x:x # S2
        indices=np.arange(len(X))
        for i in range(self.max_iter):
            np.random.seed(i) #Just so everyone gets the same answer!
            np.random.shuffle(indices)
            X_shuffle=X[indices]
            y_shuffle=y[indices]
            for j in range(0,len(X),self.batch_size):
                X_batch=X_shuffle[j:j+self.batch_size]
                y_batch=y_shuffle[j:j+self.batch_size]
                residuals=self.predict(X_batch)-y_batch
                coef_grad=(X_batch.T)@residuals/len(X_batch)
                intercept_grad=np.mean(residuals)
                self.coef-=self.lr*coef_grad+self.alpha*penalty_grad(self.coef) # S3
                self.intercept-=self.lr*intercept_grad+self.alpha*penalty_grad(self.intercept) # S4

    def predict(self,X):
        return X@self.coef+self.intercept

    def mse(self,X,y): #Not a sklearn method, but added here for convenience
        return ((self.predict(X)-y)**2).mean()`

const logisticRegressionCode = `import numpy as np

class LogisticRegression():
    def __init__(self,learning_rate, max_iter, batch_size, penalty='l2', alpha=0.0001):
        self.lr=learning_rate
        self.max_iter=max_iter #number of epochs
        self.batch_size=batch_size
        self.penalty=penalty #either 'l1' or 'l2'
        self.alpha=alpha #amount of regularization to apply

    def fit(self,X,y):
        self.coef=np.ones((X.shape[1],)) #Initial values
        self.intercept=1 #Initial value
        if self.penalty=='l1':
            penalty_grad=lambda x:2*(x>0)-1
        elif self.penalty=='l2':
            penalty_grad=lambda x:x
        indices=np.arange(len(X))
        for i in range(self.max_iter):
            np.random.seed(i) #Just so everyone gets the same answer!
            np.random.shuffle(indices)
            X_shuffle=X[indices]
            y_shuffle=y[indices]
            for j in range(0,len(X),self.batch_size):
                X_batch=X_shuffle[j:j+self.batch_size]
                y_batch=y_shuffle[j:j+self.batch_size]
                residuals=self.predict_proba(X_batch)-y_batch # S1
                coef_grad=(X_batch.T)@residuals/len(X_batch) # S2
                intercept_grad=np.mean(residuals) # S3
                self.coef-=self.lr*coef_grad+self.alpha*penalty_grad(self.coef)
                self.intercept-=self.lr*intercept_grad+self.alpha*penalty_grad(self.intercept)

    def predict_proba(self,X):
        '''returns the predicted probabilites that each x in X
        is in the "True" class'''
        t=X@self.coef+self.intercept # S4
        return 1/(1+np.exp(-t)) # S5

    def predict(self,X):
        '''returns a prediction, for each x in X, that the class
        is "True".'''
        return self.predict_proba(X)>0.5 # S6

    def score(self,X,y):
        '''returns accuracy of the model'''
        return (self.predict(X)==y).mean()

    def NegLogLoss(self,X,y): #Not a sklearn method!
        '''returns the negative of the mean Log Loss'''
        probs=self.predict_proba(X)
        return -np.mean(y*np.log(probs)+(1-y)*np.log(1-probs))`

const softmaxCode = `import numpy as np

class OneHotEncoder():
    def __init__(self):
        pass

    def fit(self,y):
        self.categories=np.unique(y) #Array of unique categories that appear in y # S1
        self.n_features_in=len(self.categories) #Number of categories

    def transform(self,y):
        Y=np.zeros((len(y),self.n_features_in))
        for i in range(self.n_features_in):
            Y[:,i]=(y==self.categories[i])
        return Y

    def fit_transform(self,y):
        '''Convenience method that applies fit and then
        immediately transforms'''
        self.fit(y)
        return self.transform(y)

class SoftmaxRegression():
    def __init__(self,learning_rate, max_iter, batch_size, penalty='l2', alpha=0.0001):
        self.lr=learning_rate
        self.max_iter=max_iter
        self.batch_size=batch_size
        self.penalty=penalty
        self.alpha=alpha
        self.encoder=OneHotEncoder()

    def fit(self,X,y):
        Y=self.encoder.fit_transform(y)
        self.coef=np.ones((X.shape[1],self.encoder.n_features_in))
        self.intercept=np.ones((self.encoder.n_features_in,))
        if self.penalty=='l1':
            penalty_grad=lambda x:2*(x>0)-1
        elif self.penalty=='l2':
            penalty_grad=lambda x:x
        indices=np.arange(len(X))
        for i in range(self.max_iter):
            np.random.seed(i)
            np.random.shuffle(indices)
            X_shuffle=X[indices]
            Y_shuffle=Y[indices]
            for j in range(0,len(X),self.batch_size):
                X_batch=X_shuffle[j:j+self.batch_size]
                Y_batch=Y_shuffle[j:j+self.batch_size]
                residuals=self.predict_proba(X_batch)-Y_batch # S2
                coef_grad=(X_batch.T)@residuals/len(X_batch) # S3
                intercept_grad=np.mean(residuals) # S4: scalar in the notebook
                self.coef-=self.lr*coef_grad+self.alpha*penalty_grad(self.coef)
                self.intercept-=self.lr*intercept_grad+self.alpha*penalty_grad(self.intercept)

    def predict_proba(self,X):
        '''returns the matrix of predicted probabilites'''
        t=X@self.coef+self.intercept # S5
        e=np.exp(t) # S6
        return e/np.sum(e,axis=1)[:,np.newaxis] # S7

    def predict(self,X):
        '''returns a prediction, for each observation in X,
        of one category.'''
        probs=self.predict_proba(X)
        indices=np.argmax(probs,axis=1) # S8
        return self.encoder.categories[indices] # S9

    def score(self,X,y):
        '''returns accuracy of the model'''
        return (self.predict(X)==y).mean()

    def CEloss(self,X,y): #Not a sklearn method!
        '''returns the Categorical Cross Entropy loss'''
        probs=self.predict_proba(X)
        Y=self.encoder.transform(y)
        return -np.sum(Y*np.log(probs))/len(y)`

export const batchGradientDescentNotebookLab: CodeLabQuestionSpec = defineCodeLabQuestion({
  id: 'batch-gradient-descent-notebook-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Trace Mini-batch Gradient Descent',
  prompt:
    'Use the completed `9BatchGradientDescent.ipynb` notebook to trace its mini-batch regression implementation, including its full-data gradient denominator.',
  instructions:
    'The displayed class is the notebook implementation. In particular, S4 and S5 divide each batch contribution by `len(X)`, not by the current batch size.',
  datasetId: 'batch-gradient-descent-notebook-v1',
  variantId: 'batch-gradient-descent-notebook-v1',
  language: 'python',
  code: batchGradientDescentCode,
  fixtureTitle: 'Notebook class with a deterministic four-row trace',
  fixture: `X_trace = np.array([
    [0.0],
    [0.0],
    [1.0],
    [1.0],
])
y_trace = np.zeros(4)

mod = SGDRegressor(
    learning_rate=0.5,
    max_iter=1,
    batch_size=2,
)`,
  invocationTitle: 'Statements to trace',
  invocationLead: 'After the displayed setup runs, Python evaluates:',
  invocation: `mod.fit(X_trace, y_trace)
query = np.array([[0.0], [1.0]])
pred = mod.predict(query)`,
  hintSchedule: [2, 4, 6],
  hints: [
    'With seed 0, S3 changes `[0, 1, 2, 3]` to `[2, 3, 1, 0]`, so the two rows with feature value 1 form the first batch.',
    'For the first batch, the residuals are `[2, 2]`; S4 and S5 both divide their sums by all four rows.',
    'For the mutation, dividing by the two-row batch doubles each first-batch gradient.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Trace the notebook updates',
      prompt:
        'Mentally execute the displayed statements through both batches of the single epoch.',
      successCopy:
        'Correct: epoch 0 shuffles the row indices to `[2, 3, 1, 0]`. The two updates leave `coef = [0.5]`, `intercept = 0.375`, and predictions `[0.375, 0.875]`.',
      fields: [
        {
          id: 'SHUFFLED_INDICES',
          label: 'indices after S3 in epoch 0',
          correctOptionId: 'INDICES_2_3_1_0',
          options: [
            { id: 'INDICES_2_3_1_0', label: '[2, 3, 1, 0]' },
            { id: 'INDICES_0_1_2_3', label: '[0, 1, 2, 3]' },
            { id: 'INDICES_3_2_1_0', label: '[3, 2, 1, 0]' },
          ],
        },
        {
          id: 'FINAL_COEF',
          label: 'mod.coef.tolist()',
          correctOptionId: 'COEF_POINT_5',
          options: [
            { id: 'COEF_POINT_5', label: '[0.5]' },
            { id: 'COEF_ZERO', label: '[0.0]' },
            { id: 'COEF_POINT_75', label: '[0.75]' },
          ],
        },
        {
          id: 'FINAL_INTERCEPT',
          label: 'round(float(mod.intercept), 3)',
          correctOptionId: 'INTERCEPT_POINT_375',
          options: [
            { id: 'INTERCEPT_POINT_375', label: '0.375' },
            { id: 'INTERCEPT_POINT_5', label: '0.500' },
            { id: 'INTERCEPT_ZERO', label: '0.000' },
          ],
        },
        {
          id: 'PREDICTIONS',
          label: 'pred.tolist()',
          correctOptionId: 'PRED_POINT_375_POINT_875',
          options: [
            { id: 'PRED_POINT_375_POINT_875', label: '[0.375, 0.875]' },
            { id: 'PRED_POINT_5_1', label: '[0.5, 1.0]' },
            { id: 'PRED_ZERO_ZERO', label: '[0.0, 0.0]' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Change the gradient denominator',
      prompt:
        'Suppose S4 and S5 both divide by `len(X_batch)` instead of `len(X)`. After rerunning the same fixture, what values result?',
      successCopy:
        'Correct: the two-row denominator makes the first update twice as large, taking both parameters to zero. The second batch then has zero residuals.',
      fields: [
        {
          id: 'MUTATED_PARAMETERS',
          label: 'mod.coef.tolist(), float(mod.intercept)',
          correctOptionId: 'MUTATED_ZERO_ZERO',
          options: [
            { id: 'MUTATED_ZERO_ZERO', label: '[0.0], 0.0' },
            { id: 'MUTATED_REFERENCE', label: '[0.5], 0.375' },
            { id: 'MUTATED_HALF_HALF', label: '[0.5], 0.5' },
          ],
        },
        {
          id: 'MUTATED_PREDICTIONS',
          label: 'pred.tolist() after rerunning the invocation',
          correctOptionId: 'MUTATED_PRED_ZERO_ZERO',
          options: [
            { id: 'MUTATED_PRED_ZERO_ZERO', label: '[0.0, 0.0]' },
            { id: 'MUTATED_PRED_REFERENCE', label: '[0.375, 0.875]' },
            { id: 'MUTATED_PRED_HALF_HALF', label: '[0.5, 0.5]' },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Choose a distinguishing test',
      prompt:
        'Which assertion passes for the notebook implementation but fails after both denominators are changed to `len(X_batch)`?',
      successCopy:
        'Correct: both versions preserve the output shape and metadata, but only the notebook denominators produce `[0.375, 0.875]`.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'TEST_REFERENCE_PREDICTIONS',
          options: [
            {
              id: 'TEST_REFERENCE_PREDICTIONS',
              label: 'Check the two predicted values',
              description:
                '`assert np.allclose(mod.predict(query), [0.375, 0.875])`',
            },
            {
              id: 'TEST_OUTPUT_SHAPE',
              label: 'Check only the prediction shape',
              description: '`assert mod.predict(query).shape == (2,)`',
            },
            {
              id: 'TEST_BATCH_SIZE',
              label: 'Check only the stored batch size',
              description: '`assert mod.batch_size == 2`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'The notebook scales each batch contribution by the four-row dataset size. The first batch changes `(coef, intercept)` from `(1, 1)` to `(0.5, 0.5)`, and the second changes only the intercept to `0.375`. A two-row denominator instead sends both parameters to zero in the first update.',
  },
  successCopy:
    'Notebook Lab complete: shuffling, batch updates, and the notebook’s gradient denominator are all correct.',
})

export const regularizationNotebookLab: CodeLabQuestionSpec = defineCodeLabQuestion({
  id: 'regularization-notebook-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Trace the Regularized Updates',
  prompt:
    'Use the completed `10Regularization.ipynb` notebook to trace its L1 update and distinguish it from an implementation that puts the penalty inside the learning-rate factor.',
  instructions:
    'The displayed class exactly follows the notebook: S3 and S4 apply `lr` to the data gradient and `alpha` separately to the penalty gradient, including the intercept.',
  datasetId: 'regularization-notebook-v1',
  variantId: 'regularization-notebook-v1',
  language: 'python',
  code: regularizationCode,
  fixtureTitle: 'Notebook class with a deterministic two-epoch trace',
  fixture: `X_trace = np.zeros((2, 2))
y_trace = np.ones(2)

mod = SGDRegressor(
    learning_rate=0.1,
    max_iter=2,
    batch_size=2,
    penalty='l1',
    alpha=0.2,
)`,
  invocationTitle: 'Statements to trace',
  invocation: `mod.fit(X_trace, y_trace)
pred = mod.predict(X_trace)
trace_mse = mod.mse(X_trace, y_trace)`,
  hintSchedule: [2, 4, 6],
  hints: [
    'At the start, every prediction equals the target 1, so the first data gradients are zero.',
    'S1 gives a penalty gradient of 1 for every positive parameter. S3 and S4 subtract `alpha * 1 = 0.2` on the first epoch.',
    'Under the mutation, the first penalty step is only `learning_rate * alpha = 0.02`.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Trace two L1-regularized epochs',
      prompt:
        'Mentally execute both epochs through the completed notebook implementation.',
      successCopy:
        'Correct: the coefficients receive only the two L1 steps and finish at `0.6`. The changing intercept also receives a data-gradient correction, so it finishes at `0.62`.',
      fields: [
        {
          id: 'FINAL_COEFFICIENTS',
          label: 'np.round(mod.coef, 3).tolist()',
          correctOptionId: 'COEF_POINT_6_POINT_6',
          options: [
            { id: 'COEF_POINT_6_POINT_6', label: '[0.6, 0.6]' },
            { id: 'COEF_POINT_8_POINT_8', label: '[0.8, 0.8]' },
            { id: 'COEF_POINT_96_POINT_96', label: '[0.96, 0.96]' },
          ],
        },
        {
          id: 'FINAL_INTERCEPT',
          label: 'round(float(mod.intercept), 3)',
          correctOptionId: 'INTERCEPT_POINT_62',
          options: [
            { id: 'INTERCEPT_POINT_62', label: '0.620' },
            { id: 'INTERCEPT_POINT_6', label: '0.600' },
            { id: 'INTERCEPT_POINT_962', label: '0.962' },
          ],
        },
        {
          id: 'PREDICTIONS',
          label: 'np.round(pred, 3).tolist()',
          correctOptionId: 'PRED_POINT_62_POINT_62',
          options: [
            { id: 'PRED_POINT_62_POINT_62', label: '[0.62, 0.62]' },
            { id: 'PRED_POINT_6_POINT_6', label: '[0.6, 0.6]' },
            { id: 'PRED_POINT_962_POINT_962', label: '[0.962, 0.962]' },
          ],
        },
        {
          id: 'MSE',
          label: 'round(float(trace_mse), 4)',
          correctOptionId: 'MSE_POINT_1444',
          options: [
            { id: 'MSE_POINT_1444', label: '0.1444' },
            { id: 'MSE_POINT_0014', label: '0.0014' },
            { id: 'MSE_POINT_38', label: '0.3800' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Move the penalty inside the learning-rate factor',
      prompt:
        'Suppose S3 and S4 are changed to subtract `self.lr * (gradient + self.alpha * penalty_grad(parameter))`. What results after rerunning the same fixture?',
      successCopy:
        'Correct: the effective penalty step falls from `0.2` to `0.02`. The coefficients finish at `0.96`, the intercept at `0.962`, and the MSE rounds to `0.0014`.',
      fields: [
        {
          id: 'MUTATED_COEFFICIENTS',
          label: 'np.round(mod.coef, 3).tolist()',
          correctOptionId: 'MUTATED_COEF_POINT_96',
          options: [
            { id: 'MUTATED_COEF_POINT_96', label: '[0.96, 0.96]' },
            { id: 'MUTATED_COEF_POINT_6', label: '[0.6, 0.6]' },
            { id: 'MUTATED_COEF_POINT_8', label: '[0.8, 0.8]' },
          ],
        },
        {
          id: 'MUTATED_INTERCEPT',
          label: 'round(float(mod.intercept), 3)',
          correctOptionId: 'MUTATED_INTERCEPT_POINT_962',
          options: [
            { id: 'MUTATED_INTERCEPT_POINT_962', label: '0.962' },
            { id: 'MUTATED_INTERCEPT_POINT_62', label: '0.620' },
            { id: 'MUTATED_INTERCEPT_POINT_96', label: '0.960' },
          ],
        },
        {
          id: 'MUTATED_MSE',
          label: 'round(float(trace_mse), 4)',
          correctOptionId: 'MUTATED_MSE_POINT_0014',
          options: [
            { id: 'MUTATED_MSE_POINT_0014', label: '0.0014' },
            { id: 'MUTATED_MSE_POINT_1444', label: '0.1444' },
            { id: 'MUTATED_MSE_POINT_038', label: '0.0380' },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Choose a distinguishing test',
      prompt:
        'Which assertion passes for the notebook update but fails when the penalty is moved inside the learning-rate factor?',
      successCopy:
        'Correct: both implementations retain the same shapes and settings, but only the notebook’s separately scaled penalty produces coefficients `[0.6, 0.6]`.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'TEST_NOTEBOOK_COEFFICIENTS',
          options: [
            {
              id: 'TEST_NOTEBOOK_COEFFICIENTS',
              label: 'Check the final coefficient values',
              description: '`assert np.allclose(mod.coef, [0.6, 0.6])`',
            },
            {
              id: 'TEST_COEFFICIENT_SHAPE',
              label: 'Check only the coefficient shape',
              description: '`assert mod.coef.shape == (2,)`',
            },
            {
              id: 'TEST_PENALTY_NAME',
              label: 'Check only the penalty setting',
              description: '`assert mod.penalty == "l1"`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'In the completed notebook, `lr` scales only the data gradient. The L1 term subtracts `alpha = 0.2` directly from each positive parameter on every epoch. Moving that term inside the `lr` factor changes the numerical algorithm and yields much smaller penalty steps.',
  },
  successCopy:
    'Notebook Lab complete: L1 penalty gradients and the notebook’s update scaling are all correct.',
})

export const logisticRegressionNotebookLab: CodeLabQuestionSpec = defineCodeLabQuestion({
  id: 'logistic-regression-notebook-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Trace Logistic Predictions',
  prompt:
    'Use the completed `11LogisticRegression.ipynb` notebook to trace sigmoid probabilities, the strict prediction threshold, accuracy, and negative mean log loss.',
  instructions:
    'The displayed class is the notebook implementation. The fixture supplies a deterministic fitted state so the questions exercise the notebook methods without depending on a long training run.',
  datasetId: 'logistic-regression-notebook-v1',
  variantId: 'logistic-regression-notebook-v1',
  language: 'python',
  code: logisticRegressionCode,
  fixtureTitle: 'Notebook class with a fixed fitted state',
  fixture: `mod = LogisticRegression(
    learning_rate=0.01,
    max_iter=1,
    batch_size=3,
    alpha=0,
)
mod.coef = np.array([1.0, -1.0])
mod.intercept = 0.0

X_trace = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [0.0, 2.0],
])
y_trace = np.array([False, True, False])`,
  invocationTitle: 'Statements to trace',
  invocation: `probs = mod.predict_proba(X_trace)
predictions = mod.predict(X_trace)
accuracy = mod.score(X_trace, y_trace)
loss = mod.NegLogLoss(X_trace, y_trace)`,
  hintSchedule: [2, 4, 6],
  hints: [
    'S4 gives linear scores `[0, 2, -2]`. Apply the sigmoid at S5 to each score independently.',
    'The first probability is exactly `0.5`; S6 uses a strict greater-than comparison.',
    'Changing only S6 cannot change `predict_proba` or `NegLogLoss`.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Trace the notebook prediction methods',
      prompt:
        'Mentally execute the four displayed statements through the completed notebook methods.',
      successCopy:
        'Correct: the scores `[0, 2, -2]` become probabilities `[0.5, 0.881, 0.119]`. The strict threshold classifies all three rows correctly, and the negative mean log loss rounds to `0.316`.',
      fields: [
        {
          id: 'PROBABILITIES',
          label: 'np.round(probs, 3).tolist()',
          correctOptionId: 'PROBS_POINT_5_POINT_881_POINT_119',
          options: [
            {
              id: 'PROBS_POINT_5_POINT_881_POINT_119',
              label: '[0.5, 0.881, 0.119]',
            },
            {
              id: 'PROBS_POINT_5_POINT_667_POINT_333',
              label: '[0.5, 0.667, 0.333]',
            },
            { id: 'PROBS_ZERO_ONE_ZERO', label: '[0.0, 1.0, 0.0]' },
          ],
        },
        {
          id: 'PREDICTIONS',
          label: 'predictions.tolist()',
          correctOptionId: 'PRED_FALSE_TRUE_FALSE',
          options: [
            { id: 'PRED_FALSE_TRUE_FALSE', label: '[False, True, False]' },
            { id: 'PRED_TRUE_TRUE_FALSE', label: '[True, True, False]' },
            { id: 'PRED_FALSE_FALSE_TRUE', label: '[False, False, True]' },
          ],
        },
        {
          id: 'ACCURACY',
          label: 'round(float(accuracy), 3)',
          correctOptionId: 'ACCURACY_ONE',
          options: [
            { id: 'ACCURACY_ONE', label: '1.000' },
            { id: 'ACCURACY_TWO_THIRDS', label: '0.667' },
            { id: 'ACCURACY_ONE_THIRD', label: '0.333' },
          ],
        },
        {
          id: 'NEG_LOG_LOSS',
          label: 'round(float(loss), 3)',
          correctOptionId: 'LOSS_POINT_316',
          options: [
            { id: 'LOSS_POINT_316', label: '0.316' },
            { id: 'LOSS_POINT_5', label: '0.500' },
            { id: 'LOSS_POINT_684', label: '0.684' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Change the classification threshold',
      prompt:
        'Suppose S6 is changed from `>0.5` to `>=0.5`. What changes when the same statements run?',
      successCopy:
        'Correct: only the zero-score row changes class. Accuracy falls to two-thirds, while the probability-based negative log loss is unchanged.',
      fields: [
        {
          id: 'MUTATED_PREDICTIONS',
          label: 'predictions.tolist()',
          correctOptionId: 'MUTATED_TRUE_TRUE_FALSE',
          options: [
            { id: 'MUTATED_TRUE_TRUE_FALSE', label: '[True, True, False]' },
            { id: 'MUTATED_FALSE_TRUE_FALSE', label: '[False, True, False]' },
            { id: 'MUTATED_TRUE_FALSE_TRUE', label: '[True, False, True]' },
          ],
        },
        {
          id: 'MUTATED_ACCURACY',
          label: 'round(float(accuracy), 3)',
          correctOptionId: 'MUTATED_ACCURACY_TWO_THIRDS',
          options: [
            { id: 'MUTATED_ACCURACY_TWO_THIRDS', label: '0.667' },
            { id: 'MUTATED_ACCURACY_ONE', label: '1.000' },
            { id: 'MUTATED_ACCURACY_ONE_THIRD', label: '0.333' },
          ],
        },
        {
          id: 'MUTATED_LOSS',
          label: 'round(float(loss), 3)',
          correctOptionId: 'MUTATED_LOSS_UNCHANGED',
          options: [
            { id: 'MUTATED_LOSS_UNCHANGED', label: '0.316' },
            { id: 'MUTATED_LOSS_POINT_5', label: '0.500' },
            { id: 'MUTATED_LOSS_UNDEFINED', label: 'undefined' },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Choose a distinguishing test',
      prompt:
        'Which assertion passes for the notebook’s strict threshold but fails after S6 is changed to `>=0.5`?',
      successCopy:
        'Correct: a zero score is the boundary case. Both implementations assign the same probability `0.5`, but only the notebook predicts `False`.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'TEST_ZERO_SCORE_CLASS',
          options: [
            {
              id: 'TEST_ZERO_SCORE_CLASS',
              label: 'Check the class at a zero score',
              description:
                '`assert mod.predict(np.array([[0.0, 0.0]])).tolist() == [False]`',
            },
            {
              id: 'TEST_ZERO_SCORE_PROBABILITY',
              label: 'Check only the probability at a zero score',
              description:
                '`assert np.allclose(mod.predict_proba(np.array([[0.0, 0.0]])), [0.5])`',
            },
            {
              id: 'TEST_POSITIVE_SCORE_CLASS',
              label: 'Check a class away from the boundary',
              description:
                '`assert mod.predict(np.array([[2.0, 0.0]])).tolist() == [True]`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'The notebook’s sigmoid maps zero to exactly `0.5`, and `predict` uses `>0.5`, so the first row is `False`. Changing only that comparison alters the first class and accuracy, but it cannot alter probabilities or negative log loss.',
  },
  successCopy:
    'Notebook Lab complete: sigmoid probabilities, thresholding, accuracy, and log loss are all correct.',
})

export const softmaxNotebookLab: CodeLabQuestionSpec = defineCodeLabQuestion({
  id: 'softmax-notebook-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Trace Multiclass Probabilities',
  prompt:
    'Use the completed `12Softmax.ipynb` notebook to trace one-hot category order, row-wise softmax probabilities, category prediction, and categorical cross-entropy.',
  instructions:
    'The displayed classes reproduce the notebook implementation. S4 deliberately remains the notebook’s scalar `np.mean(residuals)`; the deterministic fixture supplies a fitted state and exercises the prediction and evaluation methods.',
  datasetId: 'softmax-notebook-v1',
  variantId: 'softmax-notebook-v1',
  language: 'python',
  code: softmaxCode,
  fixtureTitle: 'Notebook classes with a fixed fitted state',
  fixture: `mod = SoftmaxRegression(
    learning_rate=0.01,
    max_iter=1,
    batch_size=3,
    alpha=0,
)
mod.encoder.fit(np.array(['cat', 'dog', 'owl']))
mod.coef = np.array([
    [1.0, 0.0, -1.0],
    [0.0, 1.0,  0.0],
])
mod.intercept = np.zeros(3)

X_trace = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
])
y_trace = np.array(['cat', 'dog', 'owl'])`,
  invocationTitle: 'Statements to trace',
  invocation: `probs = mod.predict_proba(X_trace)
predictions = mod.predict(X_trace)
accuracy = mod.score(X_trace, y_trace)
loss = mod.CEloss(X_trace, y_trace)`,
  hintSchedule: [2, 4, 6],
  hints: [
    'S1 sorts the category names. The columns therefore represent `cat`, `dog`, and `owl`, in that order.',
    'At S7, each row is divided by its own sum. Equal scores in the last row therefore become three probabilities of one-third.',
    '`np.argmax` and `np.argmin` both choose the first index when a row has a tie.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Trace the notebook prediction methods',
      prompt:
        'Mentally execute the four displayed statements through the completed notebook methods.',
      successCopy:
        'Correct: the first probability row is `[0.665, 0.245, 0.090]`. The equal third row ties and `argmax` selects the first category, giving predictions `cat, dog, cat`, accuracy two-thirds, and loss `0.686`.',
      fields: [
        {
          id: 'FIRST_PROBABILITY_ROW',
          label: 'np.round(probs[0], 3).tolist()',
          correctOptionId: 'PROBS_POINT_665_POINT_245_POINT_09',
          options: [
            {
              id: 'PROBS_POINT_665_POINT_245_POINT_09',
              label: '[0.665, 0.245, 0.090]',
            },
            {
              id: 'PROBS_POINT_5_POINT_333_POINT_167',
              label: '[0.500, 0.333, 0.167]',
            },
            { id: 'PROBS_ONE_ZERO_NEG_ONE', label: '[1.000, 0.000, -1.000]' },
          ],
        },
        {
          id: 'PREDICTIONS',
          label: 'predictions.tolist()',
          correctOptionId: 'PRED_CAT_DOG_CAT',
          options: [
            { id: 'PRED_CAT_DOG_CAT', label: "['cat', 'dog', 'cat']" },
            { id: 'PRED_CAT_DOG_OWL', label: "['cat', 'dog', 'owl']" },
            { id: 'PRED_OWL_CAT_CAT', label: "['owl', 'cat', 'cat']" },
          ],
        },
        {
          id: 'ACCURACY',
          label: 'round(float(accuracy), 3)',
          correctOptionId: 'ACCURACY_TWO_THIRDS',
          options: [
            { id: 'ACCURACY_TWO_THIRDS', label: '0.667' },
            { id: 'ACCURACY_ONE', label: '1.000' },
            { id: 'ACCURACY_ONE_THIRD', label: '0.333' },
          ],
        },
        {
          id: 'CROSS_ENTROPY',
          label: 'round(float(loss), 3)',
          correctOptionId: 'LOSS_POINT_686',
          options: [
            { id: 'LOSS_POINT_686', label: '0.686' },
            { id: 'LOSS_POINT_333', label: '0.333' },
            { id: 'LOSS_POINT_1099', label: '1.099' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Reverse the category-selection rule',
      prompt:
        'Suppose S8 is changed from `np.argmax(probs,axis=1)` to `np.argmin(probs,axis=1)`. What changes when the same statements run?',
      successCopy:
        'Correct: the changed method selects `owl`, `cat`, and then the first tied category `cat`. None matches the supplied target, while the probability matrix and cross-entropy remain unchanged.',
      fields: [
        {
          id: 'MUTATED_PREDICTIONS',
          label: 'predictions.tolist()',
          correctOptionId: 'MUTATED_OWL_CAT_CAT',
          options: [
            { id: 'MUTATED_OWL_CAT_CAT', label: "['owl', 'cat', 'cat']" },
            { id: 'MUTATED_CAT_DOG_CAT', label: "['cat', 'dog', 'cat']" },
            { id: 'MUTATED_DOG_CAT_OWL', label: "['dog', 'cat', 'owl']" },
          ],
        },
        {
          id: 'MUTATED_ACCURACY',
          label: 'round(float(accuracy), 3)',
          correctOptionId: 'MUTATED_ACCURACY_ZERO',
          options: [
            { id: 'MUTATED_ACCURACY_ZERO', label: '0.000' },
            { id: 'MUTATED_ACCURACY_ONE_THIRD', label: '0.333' },
            { id: 'MUTATED_ACCURACY_TWO_THIRDS', label: '0.667' },
          ],
        },
        {
          id: 'MUTATED_PROBABILITIES',
          label: 'Effect on probs',
          correctOptionId: 'MUTATED_PROBS_UNCHANGED',
          options: [
            { id: 'MUTATED_PROBS_UNCHANGED', label: 'The probability matrix is unchanged.' },
            { id: 'MUTATED_PROBS_NEGATED', label: 'Every probability is negated.' },
            { id: 'MUTATED_PROBS_COLUMNS', label: 'The columns, rather than rows, sum to 1.' },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Choose a distinguishing test',
      prompt:
        'Which assertion passes for the notebook’s S8 `argmax` but fails after it is changed to `argmin`?',
      successCopy:
        'Correct: the mutation changes only category selection. Probability shape, row sums, and encoder category order remain the same.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'TEST_REFERENCE_CATEGORIES',
          options: [
            {
              id: 'TEST_REFERENCE_CATEGORIES',
              label: 'Check all predicted categories',
              description:
                "`assert mod.predict(X_trace).tolist() == ['cat', 'dog', 'cat']`",
            },
            {
              id: 'TEST_ROW_SUMS',
              label: 'Check only the probability row sums',
              description:
                '`assert np.allclose(mod.predict_proba(X_trace).sum(axis=1), 1)`',
            },
            {
              id: 'TEST_ENCODER_ORDER',
              label: 'Check only the encoder category order',
              description:
                "`assert mod.encoder.categories.tolist() == ['cat', 'dog', 'owl']`",
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'The encoder sorts the categories, and S7 normalizes each score row. S8 then chooses the first largest probability. Replacing `argmax` with `argmin` changes only the selected category indices; it does not recompute probabilities or categorical cross-entropy. The fit method’s S4 remains a scalar mean exactly as written in the notebook.',
  },
  successCopy:
    'Notebook Lab complete: one-hot category order, softmax probabilities, predictions, and cross-entropy are all correct.',
})
