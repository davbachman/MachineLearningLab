import type { DistanceMetric, Vec2 } from '../types'
import { assignPointsToCentroids } from '../lib/kmeansMath'
import { batchGradientVisualDatasets, type BatchGradientVisualDataset } from './batchGradientAssignment'
import { regularizationVisualDatasets, type RegularizationVisualDataset } from './regularizationAssignment'
import { logisticVisualDatasets, type LogisticVisualDataset } from './logisticAssignment'
import { softmaxVisualDatasets, type SoftmaxVisualDataset } from './softmaxAssignment'
import { neuralFoundationsVisualDatasets, type NeuralFoundationsVisualDataset } from './neuralFoundationsAssignments'
import { sequenceFoundationsVisualDatasets, type SequenceFoundationsVisualDataset } from './sequenceFoundationsAssignments'
import { transformerInterpretabilityVisualDatasets, type TransformerInterpretabilityVisualDataset } from './transformerInterpretabilityAssignments'

export interface MultipleChoiceTableDataset {
  id: string
  kind: 'table'
  title?: string
  headers: string[]
  rows: Array<Array<string | number>>
  caption: string
}

export interface MultipleChoicePointCloudScenariosDataset {
  id: string
  kind: 'pointCloudScenarios'
  caption: string
  points: Vec2[]
  scenarios: {
    id: string
    title: string
    centroids: Vec2[]
  }[]
}

export interface MultipleChoiceMetricComparisonDataset {
  id: string
  kind: 'metricComparison'
  caption: string
  points: Vec2[]
  centroids: Vec2[]
  euclideanAssignments: number[]
  manhattanAssignments: number[]
}

export interface MultipleChoiceElbowDataset {
  id: string
  kind: 'elbowChart'
  caption: string
  values: {
    k: number
    objective: number
  }[]
}

export interface RegressionVisualDataset {
  id: string
  kind: 'regressionVisual'
  mode: 'curves' | 'errors'
  caption: string
}

export type MultipleChoiceDataset =
  | NeuralFoundationsVisualDataset
  | SequenceFoundationsVisualDataset
  | TransformerInterpretabilityVisualDataset
  | BatchGradientVisualDataset
  | RegularizationVisualDataset
  | LogisticVisualDataset
  | SoftmaxVisualDataset
  | GradientVisualDataset
  | MultipleChoiceTableDataset
  | MultipleChoicePointCloudScenariosDataset
  | MultipleChoiceMetricComparisonDataset
  | MultipleChoiceElbowDataset
  | RegressionVisualDataset

export interface GradientVisualDataset {
  id: string
  kind: 'gradientVisual'
  mode: 'slopes' | 'contours' | 'runs' | 'minima'
  caption: string
}

const houseFeatureRows = [
  [210000, 2],
  [285000, 3],
  [360000, 4],
  [435000, 5],
]

const trainTestAccuracyRows: Array<Array<string | number>> = [
  [1, 'train', 'teal', 'teal'],
  [2, 'train', 'teal', 'teal'],
  [3, 'train', 'teal', 'orange'],
  [4, 'train', 'orange', 'orange'],
  [5, 'train', 'orange', 'orange'],
  [6, 'train', 'orange', 'orange'],
  [7, 'train', 'teal', 'teal'],
  [8, 'train', 'orange', 'orange'],
  [9, 'test', 'teal', 'orange'],
  [10, 'test', 'orange', 'orange'],
  [11, 'test', 'teal', 'teal'],
  [12, 'test', 'orange', 'teal'],
]

const initializationPoints: Vec2[] = [
  [-5, 0],
  [-4, 1],
  [-4, -1],
  [4, 4],
  [5, 4],
  [4, 5],
  [4, -4],
  [5, -4],
  [4, -5],
]

const metricCentroids: Vec2[] = [
  [-5, -4],
  [1, 4],
]
const metricPoints: Vec2[] = [
  [-6, -4],
  [-5, -5],
  [-4, -4],
  [-1, 4],
  [1, 3],
  [2, 4],
  [4, -4],
  [5, -4],
]

export const multipleChoiceDatasets: Record<string, MultipleChoiceDataset> = {
  ...neuralFoundationsVisualDatasets,
  ...sequenceFoundationsVisualDatasets,
  ...transformerInterpretabilityVisualDatasets,
  ...batchGradientVisualDatasets,
  ...regularizationVisualDatasets,
  ...logisticVisualDatasets,
  ...softmaxVisualDatasets,
  gradientSlopes: { id: 'gradientSlopes', kind: 'gradientVisual', mode: 'slopes', caption: 'Loss L(w) = (w − 1)² + 1. Orange segments show tangent slopes. A is at w = −2, B at w = 1, and C at w = 3.' },
  gradientContours: { id: 'gradientContours', kind: 'gradientVisual', mode: 'contours', caption: 'L(a, b) = 1 + (a − 1)² + 4(b + 1)². S marks the current parameters (−2, 1); the × marks the minimum. Numbers on contours are loss values.' },
  gradientRuns: { id: 'gradientRuns', kind: 'gradientVisual', mode: 'runs', caption: 'The same quadratic loss and start as Questions 2–3. All panels use identical axes. Step 0 is before any update; the minimum possible loss is 1.' },
  gradientMinima: { id: 'gradientMinima', kind: 'gradientVisual', mode: 'minima', caption: 'A nonconvex loss: L(w) = w⁴/4 − w³/3 − w² + 3. S is the initial parameter; A, B, and C have zero slope.' },
  polynomialCurves: {
    id: 'polynomialCurves', kind: 'regressionVisual', mode: 'curves',
    caption: 'Three least-squares models fitted to the same twelve training observations. These small example data are separate from the car dataset. All panels use identical axes; no validation observations are shown.',
  },
  polynomialErrors: {
    id: 'polynomialErrors', kind: 'regressionVisual', mode: 'errors',
    caption: 'A separate car displacement/MPG experiment. Teal circles show training MSE; orange squares show validation MSE. Smaller is better.',
  },
  polynomialFeatureTrace: {
    id: 'polynomialFeatureTrace', kind: 'table', title: 'Scaled inputs',
    headers: ['Observation', 'x'], rows: [[1, -2], [2, -1], [3, 0], [4, 2]],
    caption: 'X = np.array([-2.0, -1.0, 0.0, 2.0]).',
  },
  polynomialModelReference: {
    id: 'polynomialModelReference', kind: 'table', title: 'Custom model pipeline',
    headers: ['Step', 'Code'], rows: [
      ['Features', 'PolynomialFeatures(2, include_bias=False)'],
      ['Fit', 'LinearRegression().fit(features, ytrain)'],
      ['Inside fit', 'Xnew = ones((n, m+1)); Xnew[:,1:] = features'],
      ['Solve', 'coeffs = inv(Xnew.T @ Xnew) @ (Xnew.T @ ytrain)'],
    ], caption: 'The notebook defines all three classes itself. The regression model supplies its own intercept column.',
  },
  polynomialEvaluationReference: {
    id: 'polynomialEvaluationReference', kind: 'table', title: 'Training and validation',
    headers: ['Quantity', 'Meaning'], rows: [
      ['Xtrain, ytrain', '80% used for learning'], ['Xval, yval', '20% held out for degree selection'],
      ['metrics[d, 0]', 'Training MSE for degree d+1'], ['metrics[d, 1]', 'Validation MSE for degree d+1'],
    ], caption: 'The same split and preprocessing are reused across degrees. MSE averages squared errors so different set sizes can be compared.',
  },
  numpyMiniArrays: {
    id: 'numpyMiniArrays',
    kind: 'table',
    title: 'Arrays used in these questions',
    headers: ['Name', 'NumPy definition'],
    rows: [
      ['y', 'np.arange(1, 13)'],
      ['Y', 'y.reshape((3, 4))'],
      ['X', 'np.ones((3, 4))'],
    ],
    caption:
      'Assume these statements run from top to bottom. NumPy uses zero-based indexing, and reshape preserves the order of the values.',
  },
  housePrices: {
    id: 'housePrices',
    kind: 'table',
    headers: ['House price ($)', 'Bedrooms'],
    rows: houseFeatureRows,
    caption:
      'Suppose you are trying to predict one house value from nearby house prices and bedroom counts. These are the two centered features you feed into PCA.',
  },
  trainTestAccuracy: {
    id: 'trainTestAccuracy',
    kind: 'table',
    headers: ['ID', 'Split', 'True label', 'Model prediction'],
    rows: trainTestAccuracyRows,
    caption:
      'A classifier has already made the predictions shown here. Use the split column to compute train and test accuracy separately.',
  },
  initSensitivity: {
    id: 'initSensitivity',
    kind: 'pointCloudScenarios',
    caption:
      'Both panels show the same dataset with different starting centroids for k = 3. Compare how well each initialization covers the visible regions.',
    points: initializationPoints,
    scenarios: [
      {
        id: 'A',
        title: 'Initialization A',
        centroids: [
          [-5.9, 1.4],
          [2.9, 3.1],
          [5.7, -2.6],
        ],
      },
      {
        id: 'B',
        title: 'Initialization B',
        centroids: [
          [-5.6, 2.5],
          [2.3, 4.1],
          [4.1, 0.8],
        ],
      },
    ],
  },
  metricComparison: {
    id: 'metricComparison',
    kind: 'metricComparison',
    caption:
      'The panel starts with Euclidean assignments. After the answer is checked or revealed, the Manhattan result appears for comparison.',
    points: metricPoints,
    centroids: metricCentroids,
    euclideanAssignments: assignPointsToCentroids(metricPoints, metricCentroids, 'euclidean'),
    manhattanAssignments: assignPointsToCentroids(metricPoints, metricCentroids, 'manhattan' as DistanceMetric),
  },
  elbowCurve: {
    id: 'elbowCurve',
    kind: 'elbowChart',
    caption:
      'Within-cluster sum of squares drops as k increases. Look for the point where the large gains start to level off.',
    values: [
      { k: 1, objective: 175.4 },
      { k: 2, objective: 36.7 },
      { k: 3, objective: 15.0 },
      { k: 4, objective: 10.0 },
    ],
  },
  linearRegressionNotebook: {
    id: 'linearRegressionNotebook',
    kind: 'table',
    title: '4LinearRegression.ipynb reference',
    headers: ['Notebook element', 'Exact implementation'],
    rows: [
      ['Endpoint model', 'coef = (ymax - ymin) / (Xmax - Xmin)'],
      ['Design matrix', 'Xnew[:, 1:] = X; first column remains ones'],
      ['Normal equation', 'inv(Xnew.T @ Xnew) @ (Xnew.T @ y)'],
      ['Prediction', 'x @ self.coef + self.intercept'],
      ['Loss summaries', 'RSS uses sum; MSE uses mean'],
    ],
    caption:
      'These are the operations used by the two regression implementations and loss calculations in the completed notebook.',
  },
  polynomialRegressionNotebook: {
    id: 'polynomialRegressionNotebook',
    kind: 'table',
    title: '5PolynomialRegression.ipynb reference',
    headers: ['Notebook element', 'Exact implementation'],
    rows: [
      ['Scale', '(X - mean) / std'],
      ['Undo scaling', 'X * std + mean'],
      ['No bias', 'columns X**1 through X**degree'],
      ['Include bias', 'columns X**0 through X**degree'],
      ['Polynomial model', 'linear regression on engineered columns'],
    ],
    caption:
      'The notebook scales displacement before generating polynomial powers, then fits an ordinary linear model to the engineered feature matrix.',
  },
  gradientDescentNotebook: {
    id: 'gradientDescentNotebook',
    kind: 'table',
    title: '6GradientDescent.ipynb reference',
    headers: ['Notebook element', 'Exact implementation'],
    rows: [
      ['Residuals', 'predict(X) - y'],
      ['Coefficient gradient', 'X.T @ residuals / len(X)'],
      ['Intercept gradient', 'mean(residuals)'],
      ['Update', 'parameter -= learning_rate * gradient'],
      ['Initialization', 'all coefficients and intercept start at 1'],
    ],
    caption:
      'Both the two-variable function and GDRegressor update parameters from gradients; GDRegressor uses full-dataset gradients each iteration.',
  },
  batchGradientDescentNotebook: {
    id: 'batchGradientDescentNotebook',
    kind: 'table',
    title: '7BatchGradientDescent.ipynb reference',
    headers: ['Notebook element', 'Exact implementation'],
    rows: [
      ['Epoch seed', 'np.random.seed(i)'],
      ['Shuffle', 'X[indices] and y[indices]'],
      ['Batch slice', 'j : j + batch_size'],
      ['Coefficient scale', 'divide by len(X), not batch length'],
      ['Updates', 'one coefficient/intercept update per batch'],
    ],
    caption:
      'The notebook’s SGDRegressor uses reproducible reshuffling and several parameter updates within every epoch.',
  },
  regularizationNotebook: {
    id: 'regularizationNotebook',
    kind: 'table',
    title: '8Regularization.ipynb reference',
    headers: ['Notebook element', 'Exact implementation'],
    rows: [
      ['L1 penalty gradient', 'np.sign(parameter)'],
      ['L2 penalty gradient', 'parameter'],
      ['Regularized update', 'lr * data_gradient + alpha * penalty_gradient'],
      ['Scaling', 'fit scaler on training features only'],
      ['Feature reduction', 'retain columns [0, 1]'],
    ],
    caption:
      'The notebook applies regularization to both coefficients and the intercept, then uses L1 coefficients to motivate a smaller feature matrix.',
  },
  logisticRegressionNotebook: {
    id: 'logisticRegressionNotebook',
    kind: 'table',
    title: '9LogisticRegression.ipynb reference',
    headers: ['Notebook element', 'Exact implementation'],
    rows: [
      ['Linear score', 't = X @ coef + intercept'],
      ['Probability', '1 / (1 + exp(-t))'],
      ['Residuals', 'predict_proba(X_batch) - y_batch'],
      ['Class rule', 'probability > 0.5'],
      ['Loss', 'negative mean binary log loss'],
    ],
    caption:
      'The implementation changes the residual definition from linear regression while retaining the same gradient structure.',
  },
  softmaxNotebook: {
    id: 'softmaxNotebook',
    kind: 'table',
    title: '10Softmax.ipynb reference',
    headers: ['Notebook element', 'Shape or operation'],
    rows: [
      ['One-hot target Y', '(observations, classes)'],
      ['Coefficient matrix', '(features, classes)'],
      ['Intercept', '(classes,)'],
      ['Softmax normalization', 'sum exp(t) across axis=1'],
      ['Prediction', 'categories[argmax(probabilities, axis=1)]'],
    ],
    caption:
      'Softmax extends the notebook’s binary classifier to a probability row over every learned category.',
  },
}
