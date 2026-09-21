import { describe, expect, it } from 'vitest'
import { publishedAssignments } from '../../src/data/assignments'
import { gradeSubmission } from './grader'
import { createReferenceSubmission } from './referenceSubmissions'

describe('new notebook-module partial credit', () => {
  for (const assignment of publishedAssignments.filter(a => (a.displayNumber ?? 0) >= 11)) {
    it(`${assignment.id}: grades a single saved notebook checkpoint, not completion claims`, () => {
      const submission = createReferenceSubmission(assignment)
      submission.questions.forEach(question => {
        question.latestAnswer = null
        question.attemptHistory = []
        question.status = 'gave_up'
        question.attempts = 0
      })
      const labIndex = assignment.questions.findIndex(q => q.kind === 'codeLab')
      const lab = assignment.questions[labIndex]
      if (lab.kind !== 'codeLab') throw new Error('Missing notebook lab')
      const stage = lab.stages[0]
      const field = stage.fields[0]
      submission.questions[labIndex].latestAnswer = {
        formatVersion: 1, questionId: lab.id, variantId: lab.variantId,
        stages: { [stage.id]: { [field.id]: field.correctOptionId } },
      }
      const count = lab.stages.flatMap(s => s.fields).length
      const result = gradeSubmission(assignment.id, submission)
      expect(result.score).toBeCloseTo(100 / assignment.questions.length / count, 5)
      expect(result.tests.filter(t => t.status === 'passed')).toHaveLength(1)
      submission.questions[labIndex].latestAnswer = {
        formatVersion: 1, questionId: lab.id, variantId: lab.variantId, stages: {},
      }
      submission.questions[labIndex].status = 'correct'
      expect(gradeSubmission(assignment.id, submission).score).toBe(0)
    })
  }
})
