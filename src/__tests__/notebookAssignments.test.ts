import { describe, expect, it } from 'vitest'
import { getCorrectCodeLabSubmission } from '../lib/codeLab'
import {
  batchGradientDescentAssignment,
  gradientDescentAssignment,
  linearRegressionAssignment,
  logisticRegressionAssignment,
  notebookAssignments,
  overfittingAssignment,
  polynomialRegressionAssignment,
  regularizationAssignment,
  softmaxAssignment,
} from '../data/notebookAssignments'

const expectedAssignments = [
  [linearRegressionAssignment, 5, '5LinearRegression.ipynb'],
  [polynomialRegressionAssignment, 6, '6PolynomialRegression.ipynb'],
  [overfittingAssignment, 7, '7Overfitting.ipynb'],
  [gradientDescentAssignment, 8, '8GradientDescent.ipynb'],
  [batchGradientDescentAssignment, 9, '9BatchGradientDescent.ipynb'],
  [regularizationAssignment, 10, '10Regularization.ipynb'],
  [logisticRegressionAssignment, 11, '11LogisticRegression.ipynb'],
  [softmaxAssignment, 12, '12Softmax.ipynb'],
] as const

describe('notebook-backed assignments', () => {
  it('publishes one complete five-question assignment for every new notebook', () => {
    expect(notebookAssignments).toHaveLength(8)

    for (const [assignment, displayNumber, filename] of expectedAssignments) {
      expect(assignment.published).toBe(true)
      expect(assignment.displayNumber).toBe(displayNumber)
      expect(assignment.questions.map((question) => question.kind)).toEqual([
        'multipleChoice',
        'multipleChoice',
        'multipleChoice',
        'multipleChoice',
        'codeLab',
      ])
      const lab = assignment.questions[4]
      expect(lab.prompt).toContain(filename)
    }
  })

  it('accepts the fixed answer IDs and rejects a changed answer in every question', () => {
    for (const assignment of notebookAssignments) {
      for (const question of assignment.questions) {
        if (question.kind === 'multipleChoice') {
          const selectedIds = Object.fromEntries(
            question.parts.map((questionPart) => [questionPart.id, questionPart.correctOptionId]),
          )
          expect(question.validator({ selectedIds }).correct).toBe(true)

          const firstPart = question.parts[0]
          const wrongOption = firstPart.options.find(
            (option) => option.id !== firstPart.correctOptionId,
          )!
          expect(
            question.validator({
              selectedIds: { ...selectedIds, [firstPart.id]: wrongOption.id },
            }).correct,
          ).toBe(false)
          continue
        }

        if (question.kind !== 'codeLab') {
          throw new Error('Expected each new assignment to end with a Notebook Lab.')
        }
        const correct = getCorrectCodeLabSubmission(question)
        expect(question.validator(correct).correct).toBe(true)

        const changed = structuredClone(correct)
        const firstField = question.stages[0].fields[0]
        changed.stages.TRACE[firstField.id] = firstField.options.find(
          (option) => option.id !== firstField.correctOptionId,
        )!.id
        expect(question.validator(changed).correct).toBe(false)
      }
    }
  })

  it('tests literal notebook behavior where the supplied code and prose differ', () => {
    const splitQuestion = overfittingAssignment.questions[0]
    const auditQuestion = overfittingAssignment.questions[3]

    expect(
      splitQuestion.validator({
        selectedIds: {
          'split-fractions': 'twenty-eighty',
          repeatability: 'no-random-state',
        },
      }).correct,
    ).toBe(true)
    expect(
      auditQuestion.validator({
        selectedIds: {
          'first-failure': 'name-error',
          'best-degree-rule': 'test-argmin-plus-one',
        },
      }).correct,
    ).toBe(true)
  })

  it('uses unique assignment and question IDs', () => {
    expect(new Set(notebookAssignments.map((assignment) => assignment.id)).size).toBe(
      notebookAssignments.length,
    )

    for (const assignment of notebookAssignments) {
      const questionIds = assignment.questions.map((question) => question.id)
      expect(new Set(questionIds).size).toBe(questionIds.length)
    }
  })
})
