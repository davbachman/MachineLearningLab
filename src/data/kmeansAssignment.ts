import type { AssignmentSpec, ValidationResult, Vec2 } from '../types'
import { kmeansInteractiveDatasets } from './kmeansDatasets'
import {
  assignPointsToCentroids,
  assignmentsMatchUpToPermutation,
  computeCentroidsFromAssignments,
  centroidsMatchUpToPermutation,
  runLloydToConvergence,
} from '../lib/kmeansMath'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import { kmeansNotebookLab } from './kmeansNotebookLab'

function validateLloydIteration(
  datasetId: keyof typeof kmeansInteractiveDatasets,
  tolerance: number,
) {
  const dataset = kmeansInteractiveDatasets[datasetId]
  const iterationAssignments = dataset.iterationAssignments ?? []
  const iterationCentroids = dataset.iterationCentroids ?? []

  return (submission: unknown): ValidationResult => {
    const payload = submission as {
      step?: 'assignments' | 'centroids'
      iteration?: number
      assignments?: number[]
      centroids?: Vec2[]
      referenceCentroids?: Vec2[]
      referenceAssignments?: number[]
    }
    const iterationIndex = payload.iteration ?? 0
    const hasReferenceCentroids = (payload.referenceCentroids?.length ?? 0) === dataset.k
    const hasReferenceAssignments = (payload.referenceAssignments?.length ?? 0) === dataset.points.length
    const targetAssignments =
      hasReferenceCentroids && payload.referenceCentroids
        ? assignPointsToCentroids(dataset.points, payload.referenceCentroids, dataset.metric)
        : iterationAssignments[iterationIndex] ?? []
    const targetCentroids =
      hasReferenceAssignments && payload.referenceAssignments
        ? computeCentroidsFromAssignments(
            dataset.points,
            payload.referenceAssignments,
            dataset.k,
            payload.referenceCentroids,
          )
        : iterationCentroids[iterationIndex] ?? []

    if (payload.step === 'assignments') {
      const assignmentCorrect =
        (payload.assignments?.length ?? 0) === targetAssignments.length &&
        (payload.assignments ?? []).every((assignment, index) => assignment === targetAssignments[index])

      return {
        correct: assignmentCorrect,
        message: assignmentCorrect
          ? 'Correct! Good Job!'
          : 'That assignment still violates the nearest-centroid rule for at least one point.',
      }
    }

    const centroidCorrect =
      (payload.centroids?.length ?? 0) === targetCentroids.length &&
      centroidsMatchUpToPermutation(payload.centroids ?? [], targetCentroids, tolerance)

    return {
      correct: centroidCorrect,
      message: centroidCorrect ? 'Correct! Good Job!' : 'Those centroids are not at the means of their assigned clusters yet.',
    }
  }
}

function validateBadInitialization(datasetId: keyof typeof kmeansInteractiveDatasets) {
  const dataset = kmeansInteractiveDatasets[datasetId]
  const obviousAssignments = dataset.obviousAssignments ?? []

  return (submission: unknown): ValidationResult => {
    const payload = submission as { centroids?: Vec2[] }
    const centroids = payload.centroids ?? []

    if (centroids.length !== dataset.k) {
      return {
        correct: false,
        message: `Place all ${dataset.k} initialization points before checking.`,
      }
    }

    const converged = runLloydToConvergence(dataset.points, centroids, dataset.metric)
    const convergesToObvious = assignmentsMatchUpToPermutation(
      converged.assignments,
      obviousAssignments,
      dataset.k,
    )

    return {
      correct: !convergesToObvious,
      message: !convergesToObvious
        ? 'Correct! Good Job!'
        : "Those starting points still converge to the obvious three-cluster solution. Try a more lopsided initialization.",
    }
  }
}

function validateMetricComparison(datasetId: keyof typeof kmeansInteractiveDatasets) {
  const dataset = kmeansInteractiveDatasets[datasetId]
  const euclideanAssignments = dataset.targetAssignments ?? []
  const manhattanAssignments = dataset.comparisonAssignments ?? []

  return (submission: unknown): ValidationResult => {
    const payload = submission as { step?: 'euclidean' | 'manhattan'; assignments?: number[] }
    const targetAssignments = payload.step === 'manhattan' ? manhattanAssignments : euclideanAssignments
    const assignmentCorrect =
      (payload.assignments?.length ?? 0) === targetAssignments.length &&
      (payload.assignments ?? []).every((assignment, index) => assignment === targetAssignments[index])

    return {
      correct: assignmentCorrect,
      message: assignmentCorrect
        ? 'Correct! Good Job!'
        : payload.step === 'manhattan'
          ? 'Those labels do not match the Manhattan-distance nearest-centroid assignments yet.'
          : 'Those labels do not match the Euclidean nearest-centroid assignments yet.',
    }
  }
}

export const kmeansAssignment: AssignmentSpec = {
  id: 'kmeans',
  displayNumber: 2,
  version: 7,
  title: 'K-means Clustering',
  topic: 'Clustering',
  description:
    'Practice full Lloyd iterations, bad initialization, metric changes, and elbow-method model selection.',
  published: true,
  questions: [
    {
      id: 'kmeans-two-iterations',
      kind: 'kmeans2d',
      title: "Run Lloyd's Algorithm to Convergence",
      prompt:
        'Keep alternating assignment and centroid-update steps until Lloyd’s algorithm converges.',
      instructions:
        'This question uses three centroids and three visible point clusters, with only a few bridge points between the clusters. The initial centroids do not start at the cluster centers.',
      datasetId: 'fullIteration',
      interactionMode: 'lloydIteration',
      centroidTolerance: 0.55,
      hintSchedule: [2, 4, 6],
      hints: [
        'Do not move the centroids yet. First make every point join its nearest current centroid.',
        'Once an assignment step is correct, each centroid should move to the arithmetic mean of the points with its color.',
        'The few bridge points between the clusters are the ones most likely to switch labels in the later iterations.',
      ],
      validator: validateLloydIteration('fullIteration', 0.55),
      reveal: {
        explanation:
          'Lloyd’s algorithm alternates between nearest-centroid assignments and centroid updates until the assignments and centroids stop changing. Because the centroids start away from the cluster centers here, the bridge points and centroid locations keep shifting for several rounds before the algorithm settles.',
      },
      successCopy: 'Correct! Good Job!',
    },
    {
      id: 'kmeans-bad-initialization',
      kind: 'kmeans2d',
      title: 'Pick a Bad Initialization',
      prompt:
        "Place three initialization points so Lloyd's algorithm does not converge to the obvious three-cluster solution.",
      instructions:
        'The clusters are visually clear. Your goal is to choose a bad starting configuration that pushes Lloyd’s algorithm into a different local optimum.',
      datasetId: 'badInitialization',
      interactionMode: 'placeCentroids',
      centroidTolerance: 0.5,
      hintSchedule: [2, 4],
      hints: [
        'Try placing two initialization points so they compete for the same natural cluster while another cluster starts under-covered.',
        "You are not looking for the visible cluster centers here. You want a starting state that makes Lloyd's algorithm settle into the wrong partition.",
      ],
      validator: validateBadInitialization('badInitialization'),
      reveal: {
        explanation:
          "K-means can converge to a poor local optimum when the initialization is bad. Even with three visually distinct clusters, Lloyd's algorithm is not guaranteed to recover the obvious grouping from every starting configuration.",
      },
      successCopy: 'Correct! Good Job!',
    },
    {
      id: 'kmeans-metric-assignments',
      kind: 'kmeans2d',
      title: 'Seeing the effect of different notions of distance.',
      prompt:
        'Using the same points and the same centroids, first color the points by the closest centroid under Euclidean distance, then repeat for Manhattan distance.',
      instructions:
        'The centroids stay fixed in both parts. Only the distance metric changes.',
      datasetId: 'metricComparison',
      interactionMode: 'metricComparison',
      centroidTolerance: 0.5,
      hintSchedule: [2, 4],
      hints: [
        'Under Euclidean distance, diagonal closeness matters directly.',
        'Under Manhattan distance, horizontal-plus-vertical travel matters more than diagonal shortcuts.',
      ],
      validator: validateMetricComparison('metricComparison'),
      reveal: {
        explanation:
          'Changing the metric changes the nearest-centroid assignments even when the points and centroids stay fixed. Manhattan distance can relabel points that look diagonally close under Euclidean distance.',
      },
      successCopy: 'Correct! Good Job!',
    },
    {
      id: 'kmeans-elbow',
      kind: 'multipleChoice',
      title: 'The Elbow Method',
      prompt: 'Choose the most reasonable value of k from the elbow curve.',
      instructions:
        'Look for the point where increasing k still helps, but the gains start flattening out instead of dropping sharply.',
      datasetId: 'elbowCurve',
      parts: [
        {
          id: 'best-k',
          prompt: 'Which k is the best tradeoff suggested by this elbow curve?',
          correctOptionId: 'k2',
          options: [
            {
              id: 'k1',
              title: 'k = 1',
              description: 'The objective is still far too large there, so the curve has not bent yet.',
            },
            {
              id: 'k2',
              title: 'k = 2',
              description: 'The biggest drop has already happened, and the curve starts flattening after this point.',
            },
            {
              id: 'k3',
              title: 'k = 3',
              description: 'The improvement from 2 to 3 is real, but it is much smaller than the jump from 1 to 2.',
            },
            {
              id: 'k4',
              title: 'k = 4',
              description: 'By that point the curve is already in the flatter regime.',
            },
          ],
        },
      ],
      hints: [
        'The elbow is where one more cluster stops buying you a comparably large improvement.',
        'Compare the size of the 1-to-2 drop with the later decreases.',
      ],
      validator: validateMultipleChoiceSelections(
        { 'best-k': 'k2' },
        'That choice misses the main elbow in the plotted objective values.',
      ),
      reveal: {
        explanation:
          'The elbow method looks for the point where adding more clusters gives diminishing returns. Here the big structural gain is at k = 2, and later improvements are much smaller.',
      },
      successCopy: 'Correct! Good Job!',
    },
    kmeansNotebookLab,
  ],
}
