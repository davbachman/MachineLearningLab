import { defineCodeLabQuestion } from '../lib/codeLab'

const kmeansNotebookCode = `import numpy as np

def centers_from_labels(data,labels):
    width=data.shape[1]
    height=len(np.bincount(labels))
    centers=np.zeros((height,width))
    for i in range(height):
        centers[i,:]=np.mean(data[labels==i],axis=0)
    return centers

def sq_distances(data,centers):
  k=len(centers)
  n=len(data)
  distances=np.zeros((n,k))
  for i in range(k):
    distances[:,i]=np.sum((data-centers[i,:])**2,axis=1)
  return distances

def labels_from_centers(data,centers):
  distances=sq_distances(data,centers)
  return np.argmin(distances,axis=1)  # S1

class KMeans():
    def __init__(self,k):
        self.k=k #Record the hyperparameter

    def fit(self,data):
        rng = np.random.default_rng(seed=1)#Define random number generator. Set seed only for gradescope!
        self.centers=rng.choice(data,size=self.k,replace=False,axis=0) #Choose 10 centers randomly
        labels=labels_from_centers(data,self.centers) #Decide which center each point is closest to
        diff=10
        while diff!=0: #Do this until labeling doesn't change
            old_labels=labels
            self.centers=centers_from_labels(data,labels) #Find center of each cluster
            labels=labels_from_centers(data,self.centers) #update labels based on closest center
            diff=np.sum((old_labels-labels)**2) #measure if labeling has changed
            print(diff) #Watch the algorithm converge!

    def predict(self,x):
        return labels_from_centers(x,self.centers) #Prediction is based on which center is closest

def confusion_matrix(clusters,target,k):
  matrix=np.zeros((k,k))
  for i in range(k):
    for j in range(k):
      matrix[i,j]=np.sum((clusters==i) & (target==j))
  return matrix`

export const kmeansNotebookLab = defineCodeLabQuestion({
  id: 'kmeans-code-lab',
  kind: 'codeLab',
  title: 'Notebook Lab: Trace the k-Means Implementation',
  prompt:
    'Use the completed `2Kmeans.ipynb` notebook to answer questions about its center calculation, squared-distance matrix, cluster assignments, prediction, and cluster-by-target confusion matrix.',
  instructions:
    'The displayed definitions reproduce the notebook implementations. The trace sets `model.centers` directly so that you can analyze `predict` without running the random initialization and convergence loop in `fit`.',
  datasetId: 'kmeans-notebook-v1',
  variantId: 'kmeans-notebook-v1',
  language: 'python',
  code: kmeansNotebookCode,
  fixtureTitle: 'Small deterministic trace input',
  fixtureHeading: 'Trace input',
  fixture: `data = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
    [8.0, 0.0],
    [10.0, 0.0],
])
labels = np.array([0, 0, 1, 1])
queries = np.array([
    [0.0, 0.0],
    [5.0, 0.0],
    [10.0, 0.0],
])
targets = np.array([0, 1, 1, 1])`,
  invocationTitle: 'Statements to trace',
  invocationLead: 'After the definitions and trace input run, Python evaluates:',
  invocation: `centers = centers_from_labels(data, labels)
distances = sq_distances(queries, centers)
assigned = labels_from_centers(data, centers)
matrix = confusion_matrix(labels, targets, 2)

model = KMeans(2)
model.centers = centers
prediction = model.predict(np.array([
    [5.0, 0.0],
    [10.0, 0.0],
]))`,
  hintSchedule: [2, 4, 6],
  hints: [
    '`sq_distances` returns one row per query and one column per center.',
    'The two computed centers are the coordinate-wise means of the points with labels 0 and 1.',
    '`np.argmin(..., axis=1)` selects one center-column index for every data row and resolves a tie by choosing the first minimum.',
  ],
  stages: [
    {
      id: 'TRACE',
      kind: 'executionTrace',
      title: '1. Trace the completed notebook',
      prompt:
        'Mentally execute the displayed statements through the completed `2Kmeans.ipynb` functions.',
      successCopy:
        'Correct: the label means are `[1, 0]` and `[9, 0]`; the middle query is equally far from them, so `argmin` assigns it to cluster 0.',
      fields: [
        {
          id: 'CENTERS',
          label: 'centers.tolist()',
          correctOptionId: 'CENTERS_MEANS',
          options: [
            { id: 'CENTERS_MEANS', label: '[[1.0, 0.0], [9.0, 0.0]]' },
            { id: 'CENTERS_ENDPOINTS', label: '[[0.0, 0.0], [10.0, 0.0]]' },
            { id: 'CENTERS_INNER', label: '[[2.0, 0.0], [8.0, 0.0]]' },
          ],
        },
        {
          id: 'DISTANCES',
          label: 'distances.tolist()',
          correctOptionId: 'DISTANCES_SQUARED',
          options: [
            {
              id: 'DISTANCES_SQUARED',
              label: '[[1.0, 81.0], [16.0, 16.0], [81.0, 1.0]]',
            },
            {
              id: 'DISTANCES_UNSQUARED',
              label: '[[1.0, 9.0], [4.0, 4.0], [9.0, 1.0]]',
            },
            {
              id: 'DISTANCES_TRANSPOSED',
              label: '[[1.0, 16.0, 81.0], [81.0, 16.0, 1.0]]',
            },
          ],
        },
        {
          id: 'ASSIGNED',
          label: 'assigned.tolist()',
          correctOptionId: 'ASSIGNED_BALANCED',
          options: [
            { id: 'ASSIGNED_BALANCED', label: '[0, 0, 1, 1]' },
            { id: 'ASSIGNED_LEFT_HEAVY', label: '[0, 0, 0, 1]' },
            { id: 'ASSIGNED_RIGHT_HEAVY', label: '[0, 1, 1, 1]' },
          ],
        },
        {
          id: 'MATRIX',
          label: 'matrix.tolist()',
          correctOptionId: 'MATRIX_CLUSTER_ROWS',
          options: [
            { id: 'MATRIX_CLUSTER_ROWS', label: '[[1.0, 1.0], [0.0, 2.0]]' },
            { id: 'MATRIX_TRANSPOSED', label: '[[1.0, 0.0], [1.0, 2.0]]' },
            { id: 'MATRIX_PURE', label: '[[2.0, 0.0], [0.0, 2.0]]' },
          ],
        },
        {
          id: 'PREDICTION',
          label: 'prediction.tolist()',
          correctOptionId: 'PREDICT_FIRST_TIE',
          options: [
            { id: 'PREDICT_FIRST_TIE', label: '[0, 1]' },
            { id: 'PREDICT_RIGHT_TIE', label: '[1, 1]' },
            { id: 'PREDICT_LEFT_ONLY', label: '[0, 0]' },
          ],
        },
      ],
    },
    {
      id: 'MUTATION',
      kind: 'diagnoseMutation',
      title: '2. Read a changed notebook line',
      prompt:
        'Suppose S1 in `labels_from_centers` is changed from `np.argmin(distances,axis=1)` to `np.argmin(distances,axis=0)`. Using the displayed `data` and computed `centers`, what does the changed function return?',
      successCopy:
        'Correct: reducing along axis 0 returns one data-row index for each center. Ties select the first matching rows, 0 and 2.',
      fields: [
        {
          id: 'MUTATED_RESULT',
          label: 'labels_from_centers(data, centers).tolist() after the change',
          correctOptionId: 'ROWS_PER_CENTER',
          options: [
            { id: 'ROWS_PER_CENTER', label: '[0, 2]' },
            { id: 'CENTER_PER_ROW', label: '[0, 0, 1, 1]' },
            { id: 'CENTER_INDICES', label: '[0, 1]' },
          ],
        },
      ],
    },
    {
      id: 'TEST',
      kind: 'distinguishingTest',
      title: '3. Choose a distinguishing test',
      prompt:
        'Which executable test passes for the notebook implementation but fails after the `axis=1` to `axis=0` mutation?',
      successCopy:
        'Correct: the reference returns one label for each of four rows, `[0, 0, 1, 1]`; the mutation instead returns the two row indices `[0, 3]`.',
      fields: [
        {
          id: 'TEST_ID',
          label: 'Test',
          correctOptionId: 'FOUR_ROWS_TWO_CENTERS',
          options: [
            {
              id: 'FOUR_ROWS_TWO_CENTERS',
              label: 'Require one label for each of four rows',
              description:
                '`X = np.array([[0., 0.], [1., 0.], [9., 0.], [10., 0.]]); C = np.array([[0., 0.], [10., 0.]]); assert labels_from_centers(X, C).tolist() == [0, 0, 1, 1]`',
            },
            {
              id: 'TWO_ROWS_TWO_CENTERS',
              label: 'Use the centers themselves as two rows',
              description:
                '`X = np.array([[0., 0.], [10., 0.]]); C = X.copy(); assert labels_from_centers(X, C).tolist() == [0, 1]`',
            },
            {
              id: 'DISTANCE_SHAPE',
              label: 'Check only the distance-matrix shape',
              description:
                '`X = np.array([[0., 0.], [1., 0.], [9., 0.], [10., 0.]]); C = np.array([[0., 0.], [10., 0.]]); assert sq_distances(X, C).shape == (4, 2)`',
            },
          ],
        },
      ],
    },
  ],
  reveal: {
    explanation:
      '`sq_distances` has one row per data point and one column per center. The notebook uses `axis=1` to select a center for every row. The mutation uses `axis=0` and instead selects a row for every center. A useful distinguishing test therefore uses different numbers of data rows and centers.',
  },
  successCopy:
    'Notebook Lab complete: center calculation, distance tracing, cluster assignment, and mutation testing are all correct.',
})
