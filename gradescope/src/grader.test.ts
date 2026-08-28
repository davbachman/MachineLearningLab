import { describe, expect, it } from 'vitest'
import { publishedAssignments } from '../../src/data/assignments'
import type { JsonValue, SubmissionExport } from '../../src/types'
import { gradeSubmission } from './grader'
import { createReferenceSubmission } from './referenceSubmissions'

function cloneSubmission(submission: SubmissionExport): SubmissionExport {
  return JSON.parse(JSON.stringify(submission)) as SubmissionExport
}

function configFor(assignment: (typeof publishedAssignments)[number]) {
  return {
    assignmentId: assignment.id,
    assignmentVersion: assignment.version,
  }
}

describe('Gradescope grader', () => {
  it('awards exactly 100 points to a canonical export for every published assignment', () => {
    expect(publishedAssignments).toHaveLength(12)

    for (const assignment of publishedAssignments) {
      const results = gradeSubmission(
        configFor(assignment),
        createReferenceSubmission(assignment),
      )

      expect(results.score, assignment.id).toBe(100)
      expect(
        results.tests.reduce((sum, test) => sum + test.max_score, 0),
        assignment.id,
      ).toBeCloseTo(100, 10)
      expect(results.tests.every((test) => test.status === 'passed'), assignment.id).toBe(true)
    }
  })

  it('grades raw answers instead of trusting exported status and outcome fields', () => {
    const assignment = publishedAssignments[0]
    const submission = cloneSubmission(createReferenceSubmission(assignment))
    submission.questions.forEach((question) => {
      question.status = 'gave_up'
      question.attemptHistory.forEach((attempt) => {
        attempt.outcome = 'incorrect'
      })
    })

    expect(gradeSubmission(configFor(assignment), submission).score).toBe(100)
  })

  it('rejects wiped raw answers across every published question and interaction kind', () => {
    for (const assignment of publishedAssignments) {
      const submission = cloneSubmission(createReferenceSubmission(assignment))
      submission.questions.forEach((question) => {
        question.status = 'correct'
        question.latestAnswer = {}
        question.attemptHistory.forEach((attempt) => {
          attempt.outcome = 'correct'
          attempt.answer = {}
        })
      })

      const results = gradeSubmission(configFor(assignment), submission)
      expect(results.score, assignment.id).toBe(0)
      expect(results.tests.every((test) => test.status === 'failed'), assignment.id).toBe(true)
    }
  })

  it('denies a wrong raw answer even when its exported status says correct', () => {
    const assignment = publishedAssignments.find((candidate) =>
      candidate.questions.some((question) => question.kind === 'multipleChoice'),
    )
    expect(assignment).toBeDefined()
    if (!assignment) return

    const questionIndex = assignment.questions.findIndex(
      (question) => question.kind === 'multipleChoice',
    )
    const question = assignment.questions[questionIndex]
    if (question.kind !== 'multipleChoice') throw new Error('Expected a multiple-choice question')

    const submission = cloneSubmission(createReferenceSubmission(assignment))
    const rawAnswer = submission.questions[questionIndex].latestAnswer
    if (
      rawAnswer === null ||
      Array.isArray(rawAnswer) ||
      typeof rawAnswer !== 'object' ||
      rawAnswer.selectedIds === null ||
      Array.isArray(rawAnswer.selectedIds) ||
      typeof rawAnswer.selectedIds !== 'object'
    ) {
      throw new Error('Unexpected multiple-choice reference shape')
    }

    const part = question.parts[0]
    rawAnswer.selectedIds[part.id] = part.options.find(
      (option) => option.id !== part.correctOptionId,
    )?.id ?? 'deliberately-wrong'
    submission.questions[questionIndex].status = 'correct'

    const results = gradeSubmission(configFor(assignment), submission)
    expect(results.score).toBeGreaterThan(0)
    expect(results.score).toBeLessThan(100)
    expect(results.tests.some((test) => test.status === 'failed')).toBe(true)
  })

  it('accepts reordered question records but rejects missing and duplicate IDs', () => {
    const assignment = publishedAssignments.find((candidate) => candidate.questions.length > 1)
    expect(assignment).toBeDefined()
    if (!assignment) return

    const reference = createReferenceSubmission(assignment)
    const reordered = cloneSubmission(reference)
    reordered.questions.reverse()
    expect(gradeSubmission(configFor(assignment), reordered).score).toBe(100)

    const missing = cloneSubmission(reference)
    missing.questions.pop()
    expect(gradeSubmission(configFor(assignment), missing).score).toBe(0)

    const duplicate = cloneSubmission(reference)
    duplicate.questions[1].id = duplicate.questions[0].id
    expect(gradeSubmission(configFor(assignment), duplicate).score).toBe(0)
  })

  it('requires the matching assignment ID and version', () => {
    const assignment = publishedAssignments[0]
    const reference = createReferenceSubmission(assignment)

    expect(
      gradeSubmission(
        { assignmentId: assignment.id, assignmentVersion: assignment.version + 1 },
        reference,
      ).score,
    ).toBe(0)

    const wrongAssignment = cloneSubmission(reference)
    wrongAssignment.assignmentId = 'not-this-assignment'
    expect(gradeSubmission(configFor(assignment), wrongAssignment).score).toBe(0)
  })

  it('uses attempt history to award progressive-phase credit', () => {
    const assignment = publishedAssignments.find((candidate) => candidate.id === 'pca')
    expect(assignment).toBeDefined()
    if (!assignment) return

    const complete = createReferenceSubmission(assignment)
    const tableIndex = assignment.questions.findIndex((question) => question.kind === 'tableEntry')
    expect(tableIndex).toBeGreaterThanOrEqual(0)

    const incomplete = cloneSubmission(complete)
    const history = incomplete.questions[tableIndex].attemptHistory
    incomplete.questions[tableIndex].attemptHistory = [
      history[history.length - 1],
    ]
    const results = gradeSubmission(configFor(assignment), incomplete)
    expect(results.score).toBeLessThan(100)
    expect(
      results.tests.filter((test) => test.status === 'failed').length,
    ).toBeGreaterThanOrEqual(2)
  })

  it('does not let k-means reference fields redefine the answer key', () => {
    const assignment = publishedAssignments.find((candidate) => candidate.id === 'kmeans')
    expect(assignment).toBeDefined()
    if (!assignment) return

    const submission = cloneSubmission(createReferenceSubmission(assignment))
    const firstAttempt = submission.questions[0].attemptHistory[0].answer
    if (firstAttempt === null || Array.isArray(firstAttempt) || typeof firstAttempt !== 'object') {
      throw new Error('Unexpected Lloyd-iteration reference shape')
    }

    const assignmentCount = Array.isArray(firstAttempt.assignments)
      ? firstAttempt.assignments.length
      : 0
    firstAttempt.assignments = Array.from({ length: assignmentCount }, () => 0)
    firstAttempt.referenceCentroids = [
      [-100, -100],
      [-100, -100],
      [-100, -100],
    ] satisfies JsonValue

    const results = gradeSubmission(configFor(assignment), submission)
    expect(results.score).toBeLessThan(100)
    expect(results.tests.some((test) => test.status === 'failed')).toBe(true)
  })
})
