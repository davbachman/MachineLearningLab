import { describe, expect, it } from 'vitest'
import { getCorrectCodeLabSubmission } from '../lib/codeLab'
import { fitLeastSquaresLine, fitLeastSquaresPlane } from '../lib/linearRegression'
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
  it('traces each normal-equation expression before the fitted slope', () => {
    const lab = linearRegressionAssignment.questions.find((question) => question.kind === 'codeLab')!
    expect(linearRegressionAssignment.version).toBe(2)
    expect(lab.stages).toHaveLength(9)
    expect(lab.stages.every((stage) => stage.fields.length === 1)).toBe(true)
    const fields = lab.stages.slice(0, 4).flatMap((stage) => stage.fields)
    expect(fields.map((field) => [field.label, field.options.find((option) => option.id === field.correctOptionId)?.label])).toEqual([
      ['(Xnew.T @ Xnew).tolist()', '[[3.0, 3.0], [3.0, 5.0]]'],
      ['np.linalg.inv(Xnew.T @ Xnew).tolist()', '[[5/6, -1/2], [-1/2, 1/2]]'],
      ['(Xnew.T @ y).tolist()', '[9.0, 13.0]'],
      ['coeffs.tolist() after S3', '[1.0, 2.0]'],
    ])
    for (const field of fields) {
      const stage = lab.stages.find((candidate) => candidate.fields.includes(field))!
      const submission = getCorrectCodeLabSubmission(lab)
      delete submission.stages[stage.id][field.id]
      expect(lab.validator(submission).correct).toBe(false)
      for (const option of field.options) {
        submission.stages[stage.id][field.id] = option.id
        expect(lab.validator(submission).correct).toBe(option.id === field.correctOptionId)
      }
    }
  })

  it('publishes the rebuilt regression assignment and complete assignments for the other notebooks', () => {
    expect(notebookAssignments).toHaveLength(8)

    for (const [assignment, displayNumber, filename] of expectedAssignments) {
      expect(assignment.published).toBe(true)
      expect(assignment.displayNumber).toBe(displayNumber)
      expect(assignment.questions.map((question) => question.kind)).toEqual(assignment.id === 'linear-regression' ? ['linearFit', 'planeFit', 'codeLab'] : [
        'multipleChoice',
        'multipleChoice',
        'multipleChoice',
        'multipleChoice',
        'codeLab',
      ])
      const lab = assignment.questions[assignment.questions.length - 1]
      expect(lab.prompt).toContain(filename)
    }
  })

  it('accepts the fixed answer IDs and rejects a changed answer in every question', () => {
    for (const assignment of notebookAssignments) {
      for (const question of assignment.questions) {
        if (question.kind === 'planeFit') {
          expect(question.validator(fitLeastSquaresPlane(question.points)).correct).toBe(true)
          expect(question.validator(question.initialCoefficients).correct).toBe(false)
          continue
        }
        if (question.kind === 'linearFit') {
          expect(question.validator(fitLeastSquaresLine(question.points)).correct).toBe(true)
          expect(question.validator({ slope: question.initialSlope, intercept: question.initialIntercept }).correct).toBe(false)
          continue
        }
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
