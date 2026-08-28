import { assignmentsById, publishedAssignments } from '../../src/data/assignments'
import { kmeansInteractiveDatasets } from '../../src/data/kmeansDatasets'
import { knnInteractiveDatasets } from '../../src/data/knnDatasets'
import {
  handCalculationDatasets,
  pca2dDatasets,
  pca3dDatasets,
} from '../../src/data/pcaDatasets'
import { getCorrectCodeLabSubmission } from '../../src/lib/codeLab'
import { assignPointsToCentroids } from '../../src/lib/kmeansMath'
import {
  classifyGridCells,
  classifyKnnPoint,
  findBestK,
} from '../../src/lib/knnMath'
import { toJsonValue } from '../../src/lib/assignmentState'
import type {
  AssignmentSpec,
  QuestionSpec,
  RecordedQuestionAttempt,
  SubmissionExport,
} from '../../src/types'

const REFERENCE_TIMESTAMP = '2026-01-01T00:00:00.000Z'

interface ReferencePhase {
  answer: unknown
  outcome?: RecordedQuestionAttempt['outcome']
}

function requireValue<T>(value: T | null | undefined, message: string): T {
  if (value === null || value === undefined) {
    throw new Error(message)
  }

  return value
}

function validateReferencePhase(
  assignment: AssignmentSpec,
  question: QuestionSpec,
  phase: ReferencePhase,
  phaseIndex: number,
) {
  const result = question.validator(phase.answer)
  if (!result.correct) {
    throw new Error(
      `Reference answer for ${assignment.id}/${question.id}, phase ${phaseIndex + 1}, ` +
        `does not pass its app validator: ${result.message ?? 'no validator message'}`,
    )
  }
}

function multipleChoicePhases(question: Extract<QuestionSpec, { kind: 'multipleChoice' }>) {
  return [
    {
      answer: {
        selectedIds: Object.fromEntries(
          question.parts.map((part) => [part.id, part.correctOptionId]),
        ),
      },
    },
  ]
}

function pca2dPhases(question: Extract<QuestionSpec, { kind: 'pca2dLine' }>) {
  const dataset = requireValue(
    pca2dDatasets[question.datasetId],
    `Missing PCA dataset ${question.datasetId}`,
  )

  return [{ answer: { direction: dataset.answerDirection } }]
}

function pca3dPhases(question: Extract<QuestionSpec, { kind: 'pca3dPlane' }>) {
  const dataset = requireValue(
    pca3dDatasets[question.datasetId],
    `Missing 3D PCA dataset ${question.datasetId}`,
  )

  return [
    {
      answer: {
        first: dataset.answerFirst,
        second: dataset.answerSecond,
      },
    },
  ]
}

function tableEntryPhases(question: Extract<QuestionSpec, { kind: 'tableEntry' }>) {
  const dataset = requireValue(
    handCalculationDatasets[question.datasetId],
    `Missing table-entry dataset ${question.datasetId}`,
  )

  return [
    {
      outcome: 'progress' as const,
      answer: { step: 'centeredData', values: dataset.centeredRows },
    },
    {
      outcome: 'progress' as const,
      answer: { step: 'covariance', values: dataset.covariance },
    },
    {
      answer: { step: 'direction', values: dataset.firstDirection },
    },
  ]
}

function lloydIterationPhases(datasetId: string): ReferencePhase[] {
  const dataset = requireValue(
    kmeansInteractiveDatasets[datasetId],
    `Missing k-means dataset ${datasetId}`,
  )
  const initialCentroids = requireValue(
    dataset.initialCentroids,
    `K-means dataset ${datasetId} has no initial centroids`,
  )
  const iterationAssignments = requireValue(
    dataset.iterationAssignments,
    `K-means dataset ${datasetId} has no iteration assignments`,
  )
  const iterationCentroids = requireValue(
    dataset.iterationCentroids,
    `K-means dataset ${datasetId} has no iteration centroids`,
  )

  if (iterationAssignments.length !== iterationCentroids.length) {
    throw new Error(`K-means dataset ${datasetId} has mismatched iteration data`)
  }

  return iterationAssignments.flatMap((assignments, iteration): ReferencePhase[] => {
    const centroidsBeforeAssignment =
      iteration === 0 ? initialCentroids : iterationCentroids[iteration - 1]
    const updatedCentroids = iterationCentroids[iteration]
    const finalIteration = iteration === iterationAssignments.length - 1

    return [
      {
        outcome: 'progress',
        answer: {
          step: 'assignments',
          iteration,
          assignments,
          referenceCentroids: centroidsBeforeAssignment,
        },
      },
      {
        outcome: finalIteration ? 'correct' : 'progress',
        answer: {
          step: 'centroids',
          iteration,
          centroids: updatedCentroids,
          referenceCentroids: updatedCentroids,
          referenceAssignments: assignments,
        },
      },
    ]
  })
}

function kmeansPhases(question: Extract<QuestionSpec, { kind: 'kmeans2d' }>): ReferencePhase[] {
  const dataset = requireValue(
    kmeansInteractiveDatasets[question.datasetId],
    `Missing k-means dataset ${question.datasetId}`,
  )

  switch (question.interactionMode) {
    case 'lloydIteration':
      return lloydIterationPhases(question.datasetId)

    case 'metricComparison':
      return [
        {
          outcome: 'progress',
          answer: {
            step: 'euclidean',
            assignments: requireValue(
              dataset.targetAssignments,
              `K-means dataset ${question.datasetId} has no Euclidean assignments`,
            ),
          },
        },
        {
          answer: {
            step: 'manhattan',
            assignments: requireValue(
              dataset.comparisonAssignments,
              `K-means dataset ${question.datasetId} has no Manhattan assignments`,
            ),
          },
        },
      ]

    case 'placeCentroids':
      if (question.datasetId === 'badInitialization') {
        // Two initial centroids compete for the left cluster. Lloyd's algorithm
        // consequently merges the upper and right-hand natural clusters.
        return [{ answer: { centroids: [[-8, 0], [-7, 1], [-7, -1]] } }]
      }

      return [
        {
          answer: {
            centroids: requireValue(
              dataset.targetCentroids,
              `K-means dataset ${question.datasetId} has no target centroids`,
            ),
          },
        },
      ]

    case 'updateCentroids':
      return [
        {
          answer: {
            centroids: requireValue(
              dataset.targetCentroids,
              `K-means dataset ${question.datasetId} has no target centroids`,
            ),
          },
        },
      ]

    case 'assignPoints': {
      const assignments =
        dataset.targetAssignments ??
        assignPointsToCentroids(
          dataset.points,
          requireValue(
            dataset.fixedCentroids ?? dataset.initialCentroids,
            `K-means dataset ${question.datasetId} has no centroids`,
          ),
          dataset.metric,
        )
      return [{ answer: { assignments } }]
    }
  }
}

function knnPhases(question: Extract<QuestionSpec, { kind: 'knn2d' }>): ReferencePhase[] {
  const dataset = requireValue(
    knnInteractiveDatasets[question.datasetId],
    `Missing k-NN dataset ${question.datasetId}`,
  )

  switch (dataset.kind) {
    case 'predictSequence':
      return dataset.kSequence.map((k, step) => ({
        outcome: step === dataset.kSequence.length - 1 ? 'correct' : 'progress',
        answer: {
          step,
          predictedLabel: classifyKnnPoint(
            dataset.trainingPoints,
            dataset.queryPoint,
            k,
            dataset.metric,
          ).label,
        },
      }))

    case 'decisionBoundary':
      return dataset.kSequence.map((k, step) => ({
        outcome: step === dataset.kSequence.length - 1 ? 'correct' : 'progress',
        answer: {
          step,
          cells: classifyGridCells(
            dataset.trainingPoints,
            dataset.bounds,
            dataset.gridColumns,
            dataset.gridRows,
            k,
            dataset.metric,
          ),
        },
      }))

    case 'bestK':
      return [
        {
          answer: {
            k: findBestK(
              dataset.trainingPoints,
              dataset.testPoints,
              dataset.kValues,
              dataset.metric,
            ).k,
          },
        },
      ]

    case 'scalingTrap':
      return [
        {
          outcome: 'progress',
          answer: {
            step: 'raw',
            predictedLabel: classifyKnnPoint(
              dataset.trainingPoints,
              dataset.queryPoint,
              dataset.k,
              dataset.metric,
            ).label,
          },
        },
        {
          answer: {
            step: 'normalized',
            predictedLabel: classifyKnnPoint(
              dataset.normalizedTrainingPoints,
              dataset.normalizedQueryPoint,
              dataset.k,
              dataset.metric,
            ).label,
          },
        },
      ]

    case 'metricComparison':
      return [
        {
          outcome: 'progress',
          answer: {
            step: 'euclidean',
            predictedLabel: classifyKnnPoint(
              dataset.trainingPoints,
              dataset.queryPoint,
              dataset.k,
              dataset.metric,
            ).label,
          },
        },
        {
          answer: {
            step: 'manhattan',
            predictedLabel: classifyKnnPoint(
              dataset.trainingPoints,
              dataset.queryPoint,
              dataset.k,
              dataset.comparisonMetric,
            ).label,
          },
        },
      ]

    case 'adversarialPlacement':
      return [{ answer: { point: dataset.examplePoint.point } }]
  }
}

function phasesForQuestion(question: QuestionSpec): ReferencePhase[] {
  switch (question.kind) {
    case 'multipleChoice':
      return multipleChoicePhases(question)
    case 'codeLab':
      return [{ answer: getCorrectCodeLabSubmission(question) }]
    case 'pca2dLine':
      return pca2dPhases(question)
    case 'pca3dPlane':
      return pca3dPhases(question)
    case 'tableEntry':
      return tableEntryPhases(question)
    case 'kmeans2d':
      return kmeansPhases(question)
    case 'knn2d':
      return knnPhases(question)
    case 'decisionTree':
    case 'randomForest':
      throw new Error(
        `No reference answer is defined for unpublished ${question.kind} question ${question.id}`,
      )
  }
}

function makeAttemptHistory(phases: ReferencePhase[]): RecordedQuestionAttempt[] {
  return phases.map((phase, index) => ({
    attemptNumber: index + 1,
    outcome: phase.outcome ?? (index === phases.length - 1 ? 'correct' : 'progress'),
    submittedAt: REFERENCE_TIMESTAMP,
    answer: toJsonValue(phase.answer),
  }))
}

/**
 * Build a deterministic, fully correct export that follows the same raw-answer
 * shapes and multi-phase attempt history as the student app.
 */
export function createReferenceSubmission(assignment: AssignmentSpec): SubmissionExport {
  if (!assignment.published) {
    throw new Error(`Assignment ${assignment.id} is not published`)
  }

  const questions = assignment.questions.map((question) => {
    const phases = phasesForQuestion(question)
    if (phases.length === 0) {
      throw new Error(`No reference phases were generated for ${assignment.id}/${question.id}`)
    }

    phases.forEach((phase, phaseIndex) =>
      validateReferencePhase(assignment, question, phase, phaseIndex),
    )

    const attemptHistory = makeAttemptHistory(phases)
    return {
      id: question.id,
      status: 'correct' as const,
      attempts: attemptHistory.length,
      hintsShown: 0,
      latestAnswer: attemptHistory[attemptHistory.length - 1].answer,
      attemptHistory,
    }
  })

  return {
    formatVersion: 2,
    assignmentId: assignment.id,
    assignmentVersion: assignment.version,
    generatedAt: REFERENCE_TIMESTAMP,
    questions,
  }
}

/** Build the canonical export for a published assignment registry ID. */
export function createReferenceSubmissionForAssignmentId(id: string): SubmissionExport {
  const assignment = assignmentsById[id]
  if (!assignment || !publishedAssignments.some((candidate) => candidate.id === id)) {
    throw new Error(`Unknown or unpublished assignment ID: ${id}`)
  }

  return createReferenceSubmission(assignment)
}
