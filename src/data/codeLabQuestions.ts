import { defineCodeLabQuestion, type CodeLabQuestionSpec } from '../lib/codeLab'

const numpyNotebookCode = `import numpy as np

X = np.ones((100, 10))  # S1
y = np.arange(1, 1001)  # S2
Y = y.reshape((100, 10))  # S3
z = y[:, np.newaxis]  # S4
A = Y[0:5, 2:6]  # S5
b = Y[(Y % 3 == 0) & (Y > 20) & (Y < 70)]  # S6
Z = np.sqrt(Y) + X  # S7
m = Y.max(axis=0)  # S8
s = Y.sum(axis=1)  # S9
P = np.matmul(X, Y.T)  # S10`

const pcaCode = `import numpy as np

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        centeredX = X - X.mean(axis=0)  # S1
        CVmatrix = centeredX.T @ centeredX / centeredX.shape[0]  # S2
        eigvals, eigvecs = np.linalg.eig(CVmatrix)  # S3
        indices = np.argsort(eigvals)[::-1]  # S4
        self.basis = eigvecs[:, indices[0:self.n_components]]  # S5

    def transform(self, X):
        centeredX = X - X.mean(axis=0)  # S6
        return centeredX @ self.basis  # S7

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)  # S8`

const kmeansCode = `import numpy as np

def lloyd_step(X, centroids):
    differences = X[:, None] - centroids[None, :]
    squared = (differences ** 2).sum(axis=2)  # S1
    labels = squared.argmin(axis=1)  # S2
    updated = centroids.copy()  # S3
    for cluster_id in range(len(centroids)):
        members = X[labels == cluster_id]  # S4
        if len(members) > 0:  # S5
            updated[cluster_id] = members.mean(axis=0)  # S6
    return labels, updated  # S7`

const knnCode = `import numpy as np
from sklearn.datasets import load_iris

class KNeighborsClassifier():
    def __init__(self, k):
        self.n_neighbors = k  # S1

    def fit(self, X, y):
        self.X = X  # S2
        self.y = y  # S3

    def sq_distances(self, length, width):
        return (
            (self.X[:, 0] - length) ** 2
            + (self.X[:, 1] - width) ** 2
        )  # S4

    def SpeciesOfNeighbors(self, length, width):
        distances = self.sq_distances(length, width)
        indices = distances.argsort()[:self.n_neighbors]  # S5
        specieslist = self.y[indices]  # S6
        return specieslist

    def majority(self, labels):
        return np.bincount(labels).argmax()  # S7

    def predict(self, length, width):
        return self.majority(
            self.SpeciesOfNeighbors(length, width)
        )  # S8`

const knnEvaluationCode = `from sklearn.neighbors import KNeighborsClassifier

def predict_labels(train_data, train_target, test_data, k):
    knn = KNeighborsClassifier(k)  # S1
    knn.fit(train_data, train_target)  # S2
    return knn.predict(test_data)  # S3

def accuracy(train_data, train_target, test_data, test_target, k):
    predictions = predict_labels(
        train_data, train_target, test_data, k
    )  # S4
    correct = (predictions == test_target).sum()  # S5
    return correct / len(predictions)  # S6`

const decisionTreeCode = `import numpy as np

def Gini(labels):
    return 1 - ((np.bincount(labels) / len(labels)) ** 2).sum()  # S1

def GiniSplit(X, y, split_feature, split_threshold):
    num = len(y)
    mask = X[:, split_feature] >= split_threshold  # S2
    num_right = len(y[mask])
    num_left = num - num_right
    return (
        num_left * Gini(y[~mask]) + num_right * Gini(y[mask])
    ) / num  # S3

def BestSplit(X, y):
    num_obs, num_features = X.shape
    split_feature = 0
    split_threshold = 0
    minGini = 1
    for feat in range(num_features):
        vals = np.unique(X[:, feat])
        thresholds = (vals[1:] + vals[:-1]) / 2  # S4
        for thresh in thresholds:
            splitval = GiniSplit(X, y, feat, thresh)
            if splitval < minGini:  # S5
                minGini = splitval
                split_feature = feat
                split_threshold = thresh
    return split_feature, split_threshold  # S6

class Node():
    def __init__(self, X, y, depth, max_depth):
        self.leaf = depth == max_depth or len(np.unique(y)) == 1
        if self.leaf:
            self.label = np.bincount(y).argmax()  # S7
        else:
            self.split_feature, self.split_threshold = BestSplit(X, y)
            mask = X[:, self.split_feature] >= self.split_threshold
            self.right = Node(X[mask], y[mask], depth + 1, max_depth)
            self.left = Node(X[~mask], y[~mask], depth + 1, max_depth)

    def predict(self, x):
        if self.leaf:
            return self.label
        if x[self.split_feature] >= self.split_threshold:  # S8
            return self.right.predict(x)
        return self.left.predict(x)`

const randomForestCode = `import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifier():
    def __init__(self, max_depth, n_estimators):
        self.max_depth = max_depth
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_estimators):
            rows, cols = X.shape
            np.random.seed(i)  # S1
            samples = np.random.choice(range(rows), rows, replace=True)  # S2
            features = np.random.choice(
                range(cols), int(np.sqrt(cols)), replace=False
            )  # S3
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X[samples][:, features], y[samples])  # S4
            self.trees.append((tree, features))  # S5

    def predict(self, x):
        preds = []
        for i in range(self.n_estimators):
            tree, feats = self.trees[i]
            preds.append(tree.predict(x[np.newaxis, feats])[0])  # S6
        return np.bincount(np.array(preds)).argmax()  # S7`

export const numpyNotebookLab = defineCodeLabQuestion({
  id: 'numpy-notebook-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Trace the NumPy Arrays',
  prompt:
    'Use the completed `0Numpy.ipynb` notebook to answer questions about its array construction, slicing, Boolean masks, reductions, and matrix multiplication.',
  instructions:
    'The code below reproduces the completed notebook cells in their original order, with statement labels added for reference. Read each expression using NumPy row and column conventions.',
  datasetId: 'numpy-notebook-v1',
  variantId: 'numpy-notebook-v1',
  language: 'python',
  code: numpyNotebookCode,
  fixtureTitle: 'Notebook execution context',
  fixtureHeading: 'Execution rule',
  fixture: `# Run S1 through S10 once, from top to bottom.
# No variables are reassigned after S10.`,
  invocationTitle: 'Expressions to trace',
  invocationLead: 'After the notebook cells S1 through S10 run, Python evaluates:',
  invocation: `array_shape = Y.shape
column_shape = z.shape
slice_value = A[3, 3]
filtered_values = b.tolist()
summary = (Z[5, 5], m[5], s[5])
matrix_summary = (P.shape, P[5, 5])`,
  hintSchedule: [2, 4, 6],
  hints: [
    '`reshape` fills rows from the one-dimensional array in order, while `np.newaxis` adds a length-one dimension without changing the values.',
    'The two coordinates in `A[3, 3]` are relative to the sliced array `A`, not to the original array `Y`.',
    'For S10, write the two matrix shapes after applying `.T`, then match the inner dimensions before calculating an entry.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Trace the completed notebook',
      prompt: 'Mentally execute the displayed expressions after running S1 through S10.',
      successCopy:
        'Correct: reshaping preserves the value order, the mask retains the multiples of three from 21 through 69, and each entry in row 5 of P is a dot product with one row of Y.',
      fields: [
        {
          id: 'ARRAY_SHAPE',
          label: 'array_shape',
          correctOptionId: 'SHAPE_100_10',
          options: [
            { id: 'SHAPE_100_10', label: '(100, 10)' },
            { id: 'SHAPE_10_100', label: '(10, 100)' },
            { id: 'SHAPE_1000', label: '(1000,)' },
          ],
        },
        {
          id: 'COLUMN_SHAPE',
          label: 'column_shape',
          correctOptionId: 'SHAPE_1000_1',
          options: [
            { id: 'SHAPE_1000_1', label: '(1000, 1)' },
            { id: 'SHAPE_1_1000', label: '(1, 1000)' },
            { id: 'SHAPE_1000', label: '(1000,)' },
          ],
        },
        {
          id: 'SLICE_VALUE',
          label: 'slice_value',
          correctOptionId: 'VALUE_36',
          options: [
            { id: 'VALUE_36', label: '36' },
            { id: 'VALUE_34', label: '34' },
            { id: 'VALUE_46', label: '46' },
          ],
        },
        {
          id: 'FILTERED_VALUES',
          label: 'filtered_values',
          correctOptionId: 'MULTIPLES_21_TO_69',
          options: [
            {
              id: 'MULTIPLES_21_TO_69',
              label: '[21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69]',
            },
            {
              id: 'MULTIPLES_24_TO_69',
              label: '[24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69]',
            },
            {
              id: 'MULTIPLES_21_TO_66',
              label: '[21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66]',
            },
          ],
        },
        {
          id: 'SUMMARY',
          label: 'summary',
          correctOptionId: 'SUMMARY_8_483_996_555',
          options: [
            { id: 'SUMMARY_8_483_996_555', label: '(8.483314773547882, 996, 555)' },
            { id: 'SUMMARY_7_483_60_50100', label: '(7.483314773547883, 60, 50100)' },
            { id: 'SUMMARY_8_416_995_545', label: '(8.416198487095663, 995, 545)' },
          ],
        },
        {
          id: 'MATRIX_SUMMARY',
          label: 'matrix_summary',
          correctOptionId: 'MATRIX_100_100_555',
          options: [
            { id: 'MATRIX_100_100_555', label: '((100, 100), 555.0)' },
            { id: 'MATRIX_10_10_50100', label: '((10, 10), 50100.0)' },
            { id: 'MATRIX_100_10_56', label: '((100, 10), 56.0)' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Read a changed matrix product',
      prompt:
        'Suppose S10 were changed to `P = np.matmul(X.T, Y)`, leaving every other statement unchanged. What would `matrix_summary` become?',
      successCopy:
        'Correct: the changed product has shape `(10, 100) @ (100, 10) = (10, 10)`. Its entry at `[5, 5]` is the sum of column 5 of Y, which is 50100.',
      fields: [
        {
          id: 'CHANGED_MATRIX_SUMMARY',
          label: 'matrix_summary after the change',
          correctOptionId: 'CHANGED_10_10_50100',
          options: [
            { id: 'CHANGED_10_10_50100', label: '((10, 10), 50100.0)' },
            { id: 'CHANGED_100_100_555', label: '((100, 100), 555.0)' },
            { id: 'CHANGED_VALUE_ERROR', label: 'a matrix-dimension ValueError' },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Choose a comprehension check',
      prompt:
        'Which assertion passes for the notebook implementation but fails after S10 is changed to `np.matmul(X.T, Y)`?',
      successCopy:
        'Correct: the original product is 100 by 100, and its `[5, 5]` entry equals the sum of row 5 of Y. The changed product satisfies neither property.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'TEST_NOTEBOOK_PRODUCT',
          options: [
            {
              id: 'TEST_NOTEBOOK_PRODUCT',
              label: 'Check the product shape and a row-sum entry',
              description: '`assert P.shape == (100, 100) and P[5, 5] == s[5]`',
            },
            {
              id: 'TEST_TWO_DIMENSIONS',
              label: 'Check only the number of dimensions',
              description: '`assert P.ndim == 2`',
            },
            {
              id: 'TEST_FLOAT_DTYPE',
              label: 'Check only that the result is floating point',
              description: '`assert np.issubdtype(P.dtype, np.floating)`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'The notebook reshapes the integers 1 through 1000 into 100 rows, adds a column dimension with `np.newaxis`, slices by row and column positions, filters with a combined Boolean mask, and reduces different axes for column maxima and row sums. The original matrix product uses `Y.T`, so `(100, 10) @ (10, 100)` produces a 100 by 100 matrix; changing the transpose to `X.T` instead produces a different valid 10 by 10 product.',
  },
  successCopy:
    'Notebook Lab complete: construction, slicing, filtering, reductions, and matrix multiplication are all correct.',
})

export const pcaCodeLab = defineCodeLabQuestion({
  id: 'pca-code-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Read the PCA Implementation',
  prompt:
    'Use the completed `homework3solutions.ipynb` notebook to answer questions about its PCA implementation and the way `fit_transform` combines its methods.',
  instructions:
    'The class below is the notebook implementation, reformatted only for readability. The trace matrix has a diagonal covariance matrix, so the displayed basis and projections are exact.',
  datasetId: 'pca-notebook-v1',
  variantId: 'pca-notebook-v1',
  language: 'python',
  code: pcaCode,
  fixtureTitle: 'Notebook class with a small trace matrix',
  fixture: `X_trace = np.array([
    [-2.0, 0.0],
    [ 0.0, 0.0],
    [ 2.0, 0.0],
])

pca = PCA(n_components=1)`,
  invocationTitle: 'Statements to trace',
  invocation: `projected = pca.fit_transform(X_trace)

X_new = np.array([
    [10.0, 0.0],
    [12.0, 0.0],
])
new_projected = pca.transform(X_new)`,
  hintSchedule: [2, 4, 6],
  hints: [
    'S1 centers each column before S2 forms the covariance matrix. Here only the first column varies.',
    '`fit_transform` first creates `self.basis`, then calls `transform` on the same matrix.',
    'S6 computes the mean of the array passed to `transform`; the class does not store the mean used during `fit`.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Trace the notebook methods',
      prompt:
        'Mentally execute the displayed statements through the completed notebook class.',
      successCopy:
        'Correct: the single principal direction is the first coordinate. `fit_transform` produces three rows, while `transform(X_new)` centers `X_new` around its own mean and produces `[-1, 1]`.',
      fields: [
        {
          id: 'BASIS_SHAPE',
          label: 'pca.basis.shape',
          correctOptionId: 'SHAPE_2_1',
          options: [
            { id: 'SHAPE_2_1', label: '(2, 1)' },
            { id: 'SHAPE_1_2', label: '(1, 2)' },
            { id: 'SHAPE_3_1', label: '(3, 1)' },
          ],
        },
        {
          id: 'PROJECTED',
          label: 'projected.tolist()',
          correctOptionId: 'PROJECTED_NEG2_0_2',
          options: [
            { id: 'PROJECTED_NEG2_0_2', label: '[[-2.0], [0.0], [2.0]]' },
            { id: 'PROJECTED_4_0_4', label: '[[4.0], [0.0], [4.0]]' },
            { id: 'PROJECTED_NEG2_0', label: '[[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]]' },
          ],
        },
        {
          id: 'NEW_PROJECTED',
          label: 'new_projected.tolist()',
          correctOptionId: 'NEW_NEG1_1',
          options: [
            { id: 'NEW_NEG1_1', label: '[[-1.0], [1.0]]' },
            { id: 'NEW_10_12', label: '[[10.0], [12.0]]' },
            { id: 'NEW_8_10', label: '[[8.0], [10.0]]' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Read a changed notebook line',
      prompt:
        'Suppose S6 were changed to `centeredX = X`, leaving every other line unchanged. What would `new_projected.tolist()` become?',
      successCopy:
        'Correct: without S6 centering, the already-fitted basis projects the raw first coordinates 10 and 12.',
      fields: [
        {
          id: 'CHANGED_S6_EFFECT',
          label: 'new_projected.tolist() after the change',
          correctOptionId: 'CHANGED_10_12',
          options: [
            { id: 'CHANGED_10_12', label: '[[10.0], [12.0]]' },
            { id: 'CHANGED_NEG1_1', label: '[[-1.0], [1.0]]' },
            { id: 'CHANGED_8_10', label: '[[8.0], [10.0]]' },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Choose a comprehension check',
      prompt:
        'Which assertion passes for the notebook implementation but fails after S6 is changed to skip centering?',
      successCopy:
        'Correct: the notebook centers the two rows passed to `transform`, so their one-dimensional projections have mean zero.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'TEST_TRANSFORM_MEAN_ZERO',
          options: [
            {
              id: 'TEST_TRANSFORM_MEAN_ZERO',
              label: 'Check the mean of the transformed rows',
              description:
                '`assert np.allclose(pca.transform(X_new).mean(axis=0), [0.0])`',
            },
            {
              id: 'TEST_OUTPUT_SHAPE',
              label: 'Check only the output shape',
              description: '`assert pca.transform(X_new).shape == (2, 1)`',
            },
            {
              id: 'TEST_BASIS_SHAPE',
              label: 'Check only the fitted basis shape',
              description: '`assert pca.basis.shape == (2, 1)`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'The notebook sorts the covariance eigenvectors by descending eigenvalue and retains `n_components` columns. Its `transform` method then centers whichever array it receives before projecting it. Thus `X_new` becomes `[[-1, 0], [1, 0]]`; skipping S6 centering instead projects the raw values 10 and 12.',
  },
  successCopy: 'Notebook Lab complete: PCA fitting, transformation, and code comprehension are all correct.',
})

export const kmeansCodeLab = defineCodeLabQuestion({
  id: 'kmeans-code-lab',
  kind: 'codeLab',
  title: 'Code Lab: Trace and Test One Lloyd Step',
  prompt:
    'Read `lloyd_step`, trace the values produced by both displayed statements, diagnose incorrect empty-cluster behavior, and choose an input that exposes it.',
  instructions:
    'Centroid indices 0 and 1 refer to rows 0 and 1 of `centroids`, even after those rows move.',
  datasetId: 'kmeans-code-v4',
  variantId: 'kmeans-code-v4',
  language: 'python',
  code: kmeansCode,
  fixtureTitle: 'Executable trace setup',
  fixture: `X = np.array([[0, 0], [2, 0], [8, 0], [10, 0]], dtype=float)
centroids = np.array([[0, 0], [7, 0]], dtype=float)`,
  invocationTitle: 'Statements to trace',
  invocation: `labels, updated_centroids = lloyd_step(X, centroids)
post_objective = ((X - updated_centroids[labels]) ** 2).sum()`,
  hintSchedule: [2, 4, 6],
  hints: [
    'S1 stores squared distances. Taking square roots would not change the nearest-centroid labels at S2.',
    'At S6, `members` contains exactly the rows whose S2 label equals the current `cluster_id`.',
    'Choose an input for which no row’s S2 label equals one centroid index.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Predict one Lloyd step',
      prompt:
        'Mentally execute both displayed statements. What are `labels.tolist()`, `updated_centroids.tolist()`, and `post_objective`?',
      successCopy:
        'Correct: S7 returns labels `[0, 0, 1, 1]` and centroids `[[1, 0], [9, 0]]`; the second statement then gives `post_objective = 4`.',
      fields: [
        {
          id: 'ASSIGNMENTS',
          label: 'labels.tolist()',
          correctOptionId: 'LABELS_0_0_1_1',
          options: [
            { id: 'LABELS_0_0_1_1', label: '[0, 0, 1, 1]' },
            { id: 'LABELS_0_1_1_1', label: '[0, 1, 1, 1]' },
            { id: 'LABELS_0_0_0_1', label: '[0, 0, 0, 1]' },
          ],
        },
        {
          id: 'UPDATED_CENTROIDS',
          label: 'updated_centroids.tolist()',
          correctOptionId: 'CENTROIDS_1_0_9_0',
          options: [
            { id: 'CENTROIDS_1_0_9_0', label: '[[1.0, 0.0], [9.0, 0.0]]' },
            { id: 'CENTROIDS_0_0_7_0', label: '[[0.0, 0.0], [7.0, 0.0]]' },
            { id: 'CENTROIDS_2_0_8_0', label: '[[2.0, 0.0], [8.0, 0.0]]' },
          ],
        },
        {
          id: 'POST_OBJECTIVE',
          label: 'post_objective',
          correctOptionId: 'OBJECTIVE_4',
          options: [
            { id: 'OBJECTIVE_2', label: '2.0' },
            { id: 'OBJECTIVE_4', label: '4.0' },
            { id: 'OBJECTIVE_8', label: '8.0' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Diagnose the mutation',
      prompt:
        'Now use three centroids, with no row assigned to centroid 2. A mutant writes `[nan, nan]` into `updated[2]` instead of retaining centroid 2’s previous position. Which code change causes that result?',
      successCopy:
        'Correct: without the S5 guard, S6 takes the mean of an empty array and writes `[nan, nan]` into row 2.',
      fields: [
        {
          id: 'MUTATION_ID',
          label: 'Code change',
          correctOptionId: 'M_REMOVE_EMPTY_GUARD',
          options: [
            {
              id: 'M_REMOVE_EMPTY_GUARD',
              label: 'Run S6 even when members is empty',
              description:
                'Delete the S5 guard and execute `updated[cluster_id] = members.mean(axis=0)` unconditionally.',
            },
            {
              id: 'M_USE_L1_DISTANCE',
              label: 'Use Manhattan distance at S1',
              description: 'Change S1 to `squared = np.abs(differences).sum(axis=2)`.',
            },
            {
              id: 'M_ZERO_INITIALIZATION',
              label: 'Initialize updated with zeros at S3',
              description: 'Change S3 to `updated = np.zeros_like(centroids)`.',
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
        'Which executable test passes for the reference implementation but fails for the guard-removal mutation?',
      successCopy:
        'Correct: centroid 2 receives no rows. The reference keeps it at `[100, 100]`; the mutant writes `[nan, nan]`.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'TEST_EMPTY_CLUSTER',
          options: [
            {
              id: 'TEST_EMPTY_CLUSTER',
              label: 'Leave a third centroid unused',
              description:
                '`X = np.array([[0, 0], [1, 0]], dtype=float); centroids = np.array([[0, 0], [1, 0], [100, 100]], dtype=float); _, updated = lloyd_step(X, centroids); assert np.allclose(updated[2], [100, 100])`',
            },
            {
              id: 'TEST_WELL_SEPARATED',
              label: 'Give both centroids members',
              description:
                '`X = np.array([[0, 0], [1, 0], [9, 0], [10, 0]], dtype=float); centroids = np.array([[0, 0], [10, 0]], dtype=float); _, updated = lloyd_step(X, centroids); assert np.allclose(updated, [[0.5, 0], [9.5, 0]])`',
            },
            {
              id: 'TEST_PERMUTED_ROWS',
              label: 'Reverse the trace rows',
              description:
                '`X = np.array([[10, 0], [8, 0], [2, 0], [0, 0]], dtype=float); centroids = np.array([[0, 0], [7, 0]], dtype=float); _, updated = lloyd_step(X, centroids); assert np.allclose(updated, [[1, 0], [9, 0]])`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'S3 begins with a copy of every input centroid. If a cluster is empty, S5 skips S6, so that copied row remains unchanged. In the selected test, centroid 2 stays at `[100, 100]`; removing the guard replaces it with `[nan, nan]`.',
  },
  successCopy: 'Code Lab complete: trace, diagnosis, and test are all correct.',
})

export const knnCodeLab = defineCodeLabQuestion({
  id: 'knn-code-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Read the k-NN Implementation',
  prompt:
    'Use the completed `homework1solutions.ipynb` notebook to answer questions about its Iris data preparation and `KNeighborsClassifier` implementation.',
  instructions:
    'The code below is the classifier from the notebook, reformatted only for readability. The small trace dataset uses the same column order as the notebook: petal length, then petal width.',
  datasetId: 'knn-notebook-v1',
  variantId: 'knn-notebook-v1',
  language: 'python',
  code: knnCode,
  fixtureTitle: 'Notebook context and trace input',
  fixture: `# Earlier notebook cells:
iris = load_iris()
X = iris.data
y = iris.target
X = X[:, 2:]

# Smaller data for tracing the same class:
X_trace = np.array([
    [1.0, 1.0],
    [2.0, 1.0],
    [4.0, 3.0],
    [5.0, 4.0],
])
y_trace = np.array([1, 0, 0, 2])

knn = KNeighborsClassifier(2)
knn.fit(X_trace, y_trace)`,
  invocationTitle: 'Statements to trace',
  invocation: `distances = knn.sq_distances(3.0, 1.5)
neighbor_species = knn.SpeciesOfNeighbors(3.0, 1.5)
prediction = knn.predict(3.0, 1.5)`,
  hintSchedule: [2, 4, 6],
  hints: [
    'The original Iris feature matrix has four columns. The slice `X[:, 2:]` retains columns 2 and 3.',
    'At S5, `argsort()` returns row indices ordered from the smallest squared distance to the largest.',
    '`np.bincount(labels).argmax()` returns the most frequent nonnegative integer label.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Connect the notebook cells',
      prompt:
        'Read the Iris preprocessing and then trace the three displayed statements through the completed class.',
      successCopy:
        'Correct: the notebook retains two petal measurements per flower. On the trace data, rows 1 and 2 are nearest, both have label 0, and `predict` returns 0.',
      fields: [
        {
          id: 'IRIS_X_SHAPE',
          label: 'X.shape after X = X[:, 2:] in the notebook',
          correctOptionId: 'SHAPE_150_2',
          options: [
            { id: 'SHAPE_150_2', label: '(150, 2)' },
            { id: 'SHAPE_2_150', label: '(2, 150)' },
            { id: 'SHAPE_150_4', label: '(150, 4)' },
          ],
        },
        {
          id: 'IRIS_X_COLUMNS',
          label: 'columns remaining in X after the slice',
          correctOptionId: 'PETAL_LENGTH_WIDTH',
          options: [
            { id: 'PETAL_LENGTH_WIDTH', label: 'petal length, petal width' },
            { id: 'SEPAL_LENGTH_WIDTH', label: 'sepal length, sepal width' },
            { id: 'SEPAL_WIDTH_PETAL_LENGTH', label: 'sepal width, petal length' },
          ],
        },
        {
          id: 'DISTANCES',
          label: 'distances.tolist()',
          correctOptionId: 'DISTANCES_4_25_1_25_3_25_10_25',
          options: [
            {
              id: 'DISTANCES_4_25_1_25_3_25_10_25',
              label: '[4.25, 1.25, 3.25, 10.25]',
            },
            {
              id: 'DISTANCES_2_0625_1_118_1_803_3_202',
              label: '[2.062, 1.118, 1.803, 3.202]',
            },
            {
              id: 'DISTANCES_10_25_3_25_1_25_4_25',
              label: '[10.25, 3.25, 1.25, 4.25]',
            },
          ],
        },
        {
          id: 'NEIGHBOR_SPECIES',
          label: 'neighbor_species.tolist()',
          correctOptionId: 'SPECIES_0_0',
          options: [
            { id: 'SPECIES_0_0', label: '[0, 0]' },
            { id: 'SPECIES_1_2', label: '[1, 2]' },
            { id: 'SPECIES_0_0_1_2', label: '[0, 0, 1, 2]' },
          ],
        },
        {
          id: 'PREDICTION',
          label: 'prediction',
          correctOptionId: 'PREDICTION_0',
          options: [
            { id: 'PREDICTION_0', label: '0' },
            { id: 'PREDICTION_1', label: '1' },
            { id: 'PREDICTION_2', label: '2' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Read a changed notebook line',
      prompt:
        'Suppose S5 used `distances.argsort()[-self.n_neighbors:]` instead of `distances.argsort()[:self.n_neighbors]`. What would the changed class do on the trace input?',
      successCopy:
        'Correct: the negative slice keeps the final two sorted indices, so the changed class selects the two farthest rows. Their labels are [1, 2], and `argmax()` breaks the 1–1 tie in favor of label 1.',
      fields: [
        {
          id: 'CHANGED_SLICE_EFFECT',
          label: 'effect of the changed S5 slice',
          correctOptionId: 'FARTHEST_LABELS_1_2_PREDICTION_1',
          options: [
            {
              id: 'FARTHEST_LABELS_1_2_PREDICTION_1',
              label: 'Select the two farthest rows',
              description: '`neighbor_species.tolist()` is `[1, 2]`; `prediction` is `1`.',
            },
            {
              id: 'REVERSE_NEAREST_LABELS_0_0_PREDICTION_0',
              label: 'Reverse the same two nearest rows',
              description: '`neighbor_species.tolist()` is `[0, 0]`; `prediction` is `0`.',
            },
            {
              id: 'LAST_TRAINING_LABELS_0_2_PREDICTION_0',
              label: 'Select the final two rows of X',
              description: '`neighbor_species.tolist()` is `[0, 2]`; `prediction` is `0`.',
            },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Choose a comprehension check',
      prompt:
        'Which assertion passes for the notebook implementation but fails after the S5 slice is changed to select the farthest rows?',
      successCopy:
        'Correct: checking the exact neighbor labels directly tests whether S5 keeps the beginning or the end of the sorted index array.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'TEST_EXACT_NEIGHBOR_LABELS',
          options: [
            {
              id: 'TEST_EXACT_NEIGHBOR_LABELS',
              label: 'Check the exact neighbor labels',
              description:
                '`assert knn.SpeciesOfNeighbors(3.0, 1.5).tolist() == [0, 0]`',
            },
            {
              id: 'TEST_NEIGHBOR_COUNT',
              label: 'Check only the number of neighbors',
              description:
                '`assert len(knn.SpeciesOfNeighbors(3.0, 1.5)) == 2`',
            },
            {
              id: 'TEST_VALID_SPECIES_LABELS',
              label: 'Check only that labels are valid',
              description:
                '`assert set(knn.SpeciesOfNeighbors(3.0, 1.5)) <= {0, 1, 2}`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'The notebook narrows Iris to petal length and petal width, stores the fitted arrays on `self`, sorts squared distances, keeps the first k indices, and takes the majority label. On the trace data, `[:2]` selects labels `[0, 0]`, while `[-2:]` selects labels `[1, 2]`.',
  },
  successCopy: 'Notebook Lab complete: data preparation, implementation trace, and test are all correct.',
})

export const knnEvaluationNotebookLab = defineCodeLabQuestion({
  id: 'knn-evaluation-notebook-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Evaluate and Select a k-NN Model',
  prompt:
    'Use the completed `homework2solutions.ipynb` notebook to answer questions about its train/test split, accuracy function, and selection of k.',
  instructions:
    'Read the NumPy operations exactly as written. In particular, `np.random.choice` samples with replacement unless `replace=False` is supplied.',
  datasetId: 'knn-evaluation-notebook-v1',
  variantId: 'knn-evaluation-notebook-v1',
  language: 'python',
  code: knnEvaluationCode,
  fixtureTitle: 'Notebook data and split cells',
  fixture: `import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

np.random.seed(6)
size = len(X)
test_frac = 0.2
test_size = int(size * test_frac)
test_indices = np.random.choice(np.arange(size), test_size)
test_mask = np.zeros(size, dtype=bool)
test_mask[test_indices] = True
train_mask = ~test_mask

train_data = X[train_mask]
train_target = y[train_mask]
test_data = X[test_mask]
test_target = y[test_mask]`,
  invocationTitle: 'Accuracy sweep and values to inspect',
  invocation: `k = np.arange(1, 20)
accuracies = np.array([
    accuracy(train_data, train_target, test_data, test_target, i)
    for i in k
])

requested_test_size = test_size
actual_test_size = int(test_mask.sum())
actual_train_size = int(train_mask.sum())
best_position = int(accuracies.argmax())
best_k = int(k[best_position])
best_accuracy = float(accuracies[best_position])`,
  hintSchedule: [2, 4, 6],
  hints: [
    'The notebook requests 30 random indices, but repeated indices set the same Boolean mask entry to True more than once.',
    '`train_mask = ~test_mask` flips every Boolean value, so every row belongs to exactly one of the two masks.',
    '`argmax()` returns the zero-based position of the first maximum, not the corresponding value stored in `k`.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Trace the notebook split',
      prompt:
        'For the seed and split code shown above, distinguish the requested test size from the number of distinct rows selected by the Boolean mask.',
      successCopy:
        'Correct: the code requests 30 indices, but three repeated draws leave 27 True entries in `test_mask`. The complementary training mask therefore contains 123 rows.',
      fields: [
        {
          id: 'ORIGINAL_X_SHAPE',
          label: 'X.shape in homework2solutions.ipynb',
          correctOptionId: 'SHAPE_150_4',
          options: [
            { id: 'SHAPE_150_4', label: '(150, 4)' },
            { id: 'SHAPE_150_2', label: '(150, 2)' },
            { id: 'SHAPE_4_150', label: '(4, 150)' },
          ],
        },
        {
          id: 'REQUESTED_TEST_SIZE',
          label: 'requested_test_size',
          correctOptionId: 'REQUESTED_30',
          options: [
            { id: 'REQUESTED_27', label: '27' },
            { id: 'REQUESTED_30', label: '30' },
            { id: 'REQUESTED_120', label: '120' },
          ],
        },
        {
          id: 'ACTUAL_TEST_SIZE',
          label: 'actual_test_size',
          correctOptionId: 'ACTUAL_TEST_27',
          options: [
            { id: 'ACTUAL_TEST_27', label: '27' },
            { id: 'ACTUAL_TEST_30', label: '30' },
            { id: 'ACTUAL_TEST_123', label: '123' },
          ],
        },
        {
          id: 'ACTUAL_TRAIN_SIZE',
          label: 'actual_train_size',
          correctOptionId: 'ACTUAL_TRAIN_123',
          options: [
            { id: 'ACTUAL_TRAIN_120', label: '120' },
            { id: 'ACTUAL_TRAIN_123', label: '123' },
            { id: 'ACTUAL_TRAIN_150', label: '150' },
          ],
        },
        {
          id: 'TEST_DATA_SHAPE',
          label: 'test_data.shape',
          correctOptionId: 'TEST_SHAPE_27_4',
          options: [
            { id: 'TEST_SHAPE_27_4', label: '(27, 4)' },
            { id: 'TEST_SHAPE_30_4', label: '(30, 4)' },
            { id: 'TEST_SHAPE_123_4', label: '(123, 4)' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Read a changed sampling call',
      prompt:
        'Suppose the notebook instead used `np.random.choice(np.arange(size), test_size, replace=False)`. What would change about the split?',
      successCopy:
        'Correct: sampling without replacement guarantees 30 distinct test indices, leaving 120 training indices.',
      fields: [
        {
          id: 'WITHOUT_REPLACEMENT_EFFECT',
          label: 'effect of replace=False',
          correctOptionId: 'EXACTLY_30_TEST_120_TRAIN',
          options: [
            {
              id: 'EXACTLY_30_TEST_120_TRAIN',
              label: '30 test rows and 120 training rows',
              description: 'Every sampled index is distinct before the mask is constructed.',
            },
            {
              id: 'STILL_27_TEST_123_TRAIN',
              label: '27 test rows and 123 training rows',
              description: 'The same repeated indices would still be present.',
            },
            {
              id: 'EXACTLY_120_TEST_30_TRAIN',
              label: '120 test rows and 30 training rows',
              description: 'The meaning of the train and test masks would be reversed.',
            },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Interpret the accuracy sweep',
      prompt:
        'The notebook output has the same maximum accuracy at k = 5 and k = 11. Which assertion correctly verifies the notebook’s rule of choosing the first k that reaches the maximum?',
      successCopy:
        'Correct: `accuracies.argmax()` is position 4, and `k[4]` is 5. The maximum accuracy is 26/27, approximately 0.962963.',
      fields: [
        {
          id: 'BEST_K_ASSERTION',
          label: 'Assertion',
          correctOptionId: 'ASSERT_FIRST_MAX_K_5',
          options: [
            {
              id: 'ASSERT_FIRST_MAX_K_5',
              label: 'Check the k value and maximum accuracy',
              description:
                '`assert k[accuracies.argmax()] == 5 and np.isclose(accuracies.max(), 26 / 27)`',
            },
            {
              id: 'ASSERT_ARGMAX_POSITION_5',
              label: 'Treat the array position as the k value',
              description: '`assert accuracies.argmax() == 5`',
            },
            {
              id: 'ASSERT_FIRST_MAX_K_11',
              label: 'Choose the later tied maximum',
              description: '`assert k[accuracies.argmax()] == 11`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'With seed 6, the 30 draws contain three repeats, so Boolean indexing produces 27 test rows and 123 training rows. The recorded accuracy array reaches 26/27 first at k = 5 and again at k = 11; NumPy `argmax()` returns the position of the first occurrence.',
  },
  successCopy: 'Notebook Lab complete: split construction, sampling behavior, and model selection are all correct.',
})

export const randomForestCodeLab = defineCodeLabQuestion({
  id: 'rf-code-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Read the Random Forest Implementation',
  prompt:
    'Use the completed `homework5solutions.ipynb` notebook to answer questions about its seeded bootstrap samples, random feature subsets, and majority vote.',
  instructions:
    'The class below is the notebook implementation, reformatted only for readability. NumPy seeds make the fit trace reproducible. `FixedTree` is used only to isolate and trace the notebook’s voting code.',
  datasetId: 'rf-notebook-v1',
  variantId: 'rf-notebook-v1',
  language: 'python',
  code: randomForestCode,
  fixtureTitle: 'Small fit trace and isolated voting trace',
  fixture: `# X_trace has 5 rows and 4 feature columns.
X_trace = np.arange(20).reshape(5, 4)
y_trace = np.array([0, 0, 1, 1, 2])
forest = RandomForestClassifier(max_depth=2, n_estimators=2)

class FixedTree:
    def __init__(self, label):
        self.label = label

    def predict(self, X):
        return np.array([self.label])

voting_forest = RandomForestClassifier(max_depth=2, n_estimators=4)
voting_forest.trees = [
    (FixedTree(2), np.array([0, 2])),
    (FixedTree(1), np.array([1, 3])),
    (FixedTree(2), np.array([0, 1])),
    (FixedTree(1), np.array([2, 3])),
]`,
  invocationTitle: 'Statements to trace',
  invocation: `forest.fit(X_trace, y_trace)

# The same S1-S3 statements, isolated for inspection:
np.random.seed(0)
samples_0 = np.random.choice(range(5), 5, replace=True)
features_0 = np.random.choice(range(4), 2, replace=False)

np.random.seed(1)
samples_1 = np.random.choice(range(5), 5, replace=True)
features_1 = np.random.choice(range(4), 2, replace=False)

prediction = voting_forest.predict(np.array([10, 20, 30, 40]))`,
  hintSchedule: [2, 4, 6],
  hints: [
    'For each tree, S1 resets NumPy’s random generator to that tree’s index before S2 and S3 draw values.',
    '`int(np.sqrt(4))` is 2, so every tree is fitted with two of the four columns.',
    'The four fixed trees vote `[2, 1, 2, 1]`. `np.bincount(...).argmax()` returns the smallest label when counts tie.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Trace sampling, features, and voting',
      prompt:
        'Read the arrays generated inside the first two fit iterations, then trace the separate `FixedTree` vote through S6 and S7.',
      successCopy:
        'Correct: tree 0 bootstraps `[4, 0, 3, 3, 3]` and uses columns `[0, 2]`; tree 1 bootstraps `[3, 4, 0, 1, 3]` and uses columns `[3, 2]`. The tied vote is resolved as class 1.',
      fields: [
        {
          id: 'SAMPLES_0',
          label: 'samples.tolist() when i == 0',
          correctOptionId: 'SAMPLES_4_0_3_3_3',
          options: [
            { id: 'SAMPLES_4_0_3_3_3', label: '[4, 0, 3, 3, 3]' },
            { id: 'SAMPLES_0_1_2_3_4', label: '[0, 1, 2, 3, 4]' },
            { id: 'SAMPLES_3_4_0_1_3', label: '[3, 4, 0, 1, 3]' },
          ],
        },
        {
          id: 'FEATURES_0',
          label: 'features.tolist() when i == 0',
          correctOptionId: 'FEATURES_0_2',
          options: [
            { id: 'FEATURES_0_2', label: '[0, 2]' },
            { id: 'FEATURES_0_1', label: '[0, 1]' },
            { id: 'FEATURES_3_2', label: '[3, 2]' },
          ],
        },
        {
          id: 'SAMPLES_1',
          label: 'samples.tolist() when i == 1',
          correctOptionId: 'SAMPLES_3_4_0_1_3',
          options: [
            { id: 'SAMPLES_3_4_0_1_3', label: '[3, 4, 0, 1, 3]' },
            { id: 'SAMPLES_4_0_3_3_3', label: '[4, 0, 3, 3, 3]' },
            { id: 'SAMPLES_1_3_4_0_2', label: '[1, 3, 4, 0, 2]' },
          ],
        },
        {
          id: 'FEATURES_1',
          label: 'features.tolist() when i == 1',
          correctOptionId: 'FEATURES_3_2',
          options: [
            { id: 'FEATURES_3_2', label: '[3, 2]' },
            { id: 'FEATURES_0_2', label: '[0, 2]' },
            { id: 'FEATURES_2_3', label: '[2, 3]' },
          ],
        },
        {
          id: 'PREDICTION',
          label: 'prediction',
          correctOptionId: 'CLASS_1',
          options: [
            { id: 'CLASS_1', label: '1' },
            { id: 'CLASS_2', label: '2' },
            { id: 'CLASS_NONE', label: 'None' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Read a changed notebook line',
      prompt:
        'Suppose S2 used `replace=False` instead of `replace=True`. Which statement best describes the training rows selected for each tree?',
      successCopy:
        'Correct: drawing five indices without replacement from five rows produces a permutation, so every tree sees every row exactly once rather than a bootstrap sample with omissions and repeats.',
      fields: [
        {
          id: 'WITHOUT_REPLACEMENT_EFFECT',
          label: 'effect of replace=False at S2',
          correctOptionId: 'PERMUTATION_EACH_ROW_ONCE',
          options: [
            {
              id: 'PERMUTATION_EACH_ROW_ONCE',
              label: 'Every row appears exactly once',
              description: 'Each sample array is a permutation of `[0, 1, 2, 3, 4]`.',
            },
            {
              id: 'DUPLICATES_WITH_NO_OMISSIONS',
              label: 'Some rows repeat but none are omitted',
              description: 'The sample still contains duplicate row indices.',
            },
            {
              id: 'TWO_ROWS_SQRT_FIVE',
              label: 'Only two rows are selected',
              description: 'The row count is reduced to `int(np.sqrt(5))`.',
            },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Choose a comprehension check',
      prompt:
        'Which assertion directly verifies how the notebook resolves the 2–2 tie in the fixed-tree voting trace?',
      successCopy:
        'Correct: the notebook counts votes by class index, and `argmax()` returns the first maximum, so the smaller tied label 1 wins.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'TEST_TIE_BREAK',
          options: [
            {
              id: 'TEST_TIE_BREAK',
              label: 'Check the tied vote’s exact prediction',
              description:
                '`assert voting_forest.predict(np.array([10, 20, 30, 40])) == 1`',
            },
            {
              id: 'TEST_VALID_CLASS',
              label: 'Check only that the result is a voter label',
              description: '`assert voting_forest.predict(np.array([10, 20, 30, 40])) in {1, 2}`',
            },
            {
              id: 'TEST_TREE_COUNT',
              label: 'Check only the number of trees',
              description: '`assert len(voting_forest.trees) == 4`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'The notebook deliberately seeds each estimator with its loop index. S2 samples rows with replacement, while S3 selects `sqrt(number of columns)` distinct features. At prediction time, the four fixed trees vote twice for class 1 and twice for class 2; `np.bincount(...).argmax()` chooses class 1 because it is the first maximum.',
  },
  successCopy: 'Notebook Lab complete: bootstrap sampling, feature subsampling, and voting are all correct.',
})

export const decisionTreeCodeLab = defineCodeLabQuestion({
  id: 'decision-tree-code-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Read the Decision Tree Implementation',
  prompt:
    'Use the completed `homework4solutions.ipynb` notebook to answer questions about Gini impurity, split search, node construction, and prediction routing.',
  instructions:
    'The code below is the notebook implementation, reformatted only for readability. The small trace input follows the same NumPy array conventions as the Wine data used in the notebook.',
  datasetId: 'decision-tree-notebook-v1',
  variantId: 'decision-tree-notebook-v1',
  language: 'python',
  code: decisionTreeCode,
  fixtureTitle: 'Small trace input',
  fixture: `X_trace = np.array([
    [0.0],
    [1.0],
    [3.0],
    [4.0],
])
y_trace = np.array([0, 0, 1, 1])`,
  invocationTitle: 'Statements to trace',
  invocation: `impurity = Gini(y_trace)
mask = X_trace[:, 0] >= 2.0
split_impurity = GiniSplit(X_trace, y_trace, 0, 2.0)
best_split = BestSplit(X_trace, y_trace)

root = Node(X_trace, y_trace, depth=0, max_depth=1)
left_prediction = root.predict(np.array([0.5]))
right_prediction = root.predict(np.array([2.5]))`,
  hintSchedule: [2, 4, 6],
  hints: [
    'The threshold 2.0 sends the final two rows to `mask` and leaves two class-0 rows on the other side.',
    'S4 tests only midpoints between successive distinct values: 0.5, 2.0, and 3.5.',
    'At S8, equality follows the `right` child because the comparison is `>=`.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Trace the notebook functions',
      prompt: 'Mentally execute the displayed statements through the completed notebook code.',
      successCopy:
        'Correct: the labels have impurity 0.5, threshold 2.0 makes two pure children, `BestSplit` returns `(0, 2.0)`, and the depth-one tree predicts 0 on the left and 1 on the right.',
      fields: [
        {
          id: 'IMPURITY',
          label: 'impurity',
          correctOptionId: 'IMPURITY_0_5',
          options: [
            { id: 'IMPURITY_0_5', label: '0.5' },
            { id: 'IMPURITY_0_25', label: '0.25' },
            { id: 'IMPURITY_0', label: '0.0' },
          ],
        },
        {
          id: 'MASK',
          label: 'mask.tolist()',
          correctOptionId: 'MASK_F_F_T_T',
          options: [
            { id: 'MASK_F_F_T_T', label: '[False, False, True, True]' },
            { id: 'MASK_T_T_F_F', label: '[True, True, False, False]' },
            { id: 'MASK_F_F_F_T', label: '[False, False, False, True]' },
          ],
        },
        {
          id: 'SPLIT_IMPURITY',
          label: 'split_impurity',
          correctOptionId: 'SPLIT_0',
          options: [
            { id: 'SPLIT_0', label: '0.0' },
            { id: 'SPLIT_0_25', label: '0.25' },
            { id: 'SPLIT_0_5', label: '0.5' },
          ],
        },
        {
          id: 'BEST_SPLIT',
          label: 'best_split',
          correctOptionId: 'BEST_0_2',
          options: [
            { id: 'BEST_0_2', label: '(0, 2.0)' },
            { id: 'BEST_0_0_5', label: '(0, 0.5)' },
            { id: 'BEST_0_3_5', label: '(0, 3.5)' },
          ],
        },
        {
          id: 'PREDICTIONS',
          label: 'left_prediction, right_prediction',
          correctOptionId: 'PREDICTIONS_0_1',
          options: [
            { id: 'PREDICTIONS_0_1', label: '0, 1' },
            { id: 'PREDICTIONS_1_0', label: '1, 0' },
            { id: 'PREDICTIONS_0_0', label: '0, 0' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Read a changed notebook line',
      prompt:
        'Suppose S8 used `>` instead of `>=`, without changing the split construction. What would `root.predict(np.array([2.0]))` return?',
      successCopy:
        'Correct: the notebook sends a value equal to the threshold right, but the changed comparison sends it left. The left leaf predicts class 0.',
      fields: [
        {
          id: 'CHANGED_BOUNDARY_PREDICTION',
          label: 'prediction at x == 2.0 after the change',
          correctOptionId: 'BOUNDARY_CLASS_0',
          options: [
            { id: 'BOUNDARY_CLASS_0', label: '0' },
            { id: 'BOUNDARY_CLASS_1', label: '1' },
            { id: 'BOUNDARY_ERROR', label: 'an IndexError' },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Choose a comprehension check',
      prompt:
        'Which assertion passes for the notebook implementation but fails after S8 is changed from `>=` to `>`?',
      successCopy:
        'Correct: only a query exactly on the learned threshold distinguishes the two routing comparisons in this tree.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'TEST_EXACT_THRESHOLD',
          options: [
            {
              id: 'TEST_EXACT_THRESHOLD',
              label: 'Test a query exactly on the threshold',
              description: '`assert root.predict(np.array([2.0])) == 1`',
            },
            {
              id: 'TEST_LEFT_OF_THRESHOLD',
              label: 'Test a query left of the threshold',
              description: '`assert root.predict(np.array([0.5])) == 0`',
            },
            {
              id: 'TEST_RIGHT_OF_THRESHOLD',
              label: 'Test a query right of the threshold',
              description: '`assert root.predict(np.array([2.5])) == 1`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      'The notebook tests midpoints between sorted distinct values and keeps the first split that strictly improves `minGini`. On the trace input, 2.0 is the unique perfect split. Node construction uses the same `>=` rule as prediction, so a query at 2.0 belongs to the right leaf and receives class 1.',
  },
  successCopy: 'Notebook Lab complete: impurity, split search, and prediction routing are all correct.',
})

export const codeLabQuestions = {
  numpy: numpyNotebookLab,
  pca: pcaCodeLab,
  kmeans: kmeansCodeLab,
  knn: knnCodeLab,
  decisionTrees: decisionTreeCodeLab,
  randomForests: randomForestCodeLab,
} satisfies Record<string, CodeLabQuestionSpec>
