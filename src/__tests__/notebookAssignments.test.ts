import { describe, expect, it } from 'vitest'
import { getCorrectCodeLabSubmission } from '../lib/codeLab'
import { fitLeastSquaresLine, fitLeastSquaresPlane } from '../lib/linearRegression'
import {
  batchGradientDescentAssignment,
  gradientDescentAssignment,
  linearRegressionAssignment,
  logisticRegressionAssignment,
  notebookAssignments,
  polynomialRegressionAssignment,
  regularizationAssignment,
  softmaxAssignment,
} from '../data/notebookAssignments'

const expectedAssignments = [
  [linearRegressionAssignment, 4, '4LinearRegression.ipynb'],
  [polynomialRegressionAssignment, 5, '5PolynomialRegression.ipynb'],
  [gradientDescentAssignment, 6, '6GradientDescent.ipynb'],
  [batchGradientDescentAssignment, 7, '7BatchGradientDescent.ipynb'],
  [regularizationAssignment, 8, '8Regularization.ipynb'],
  [logisticRegressionAssignment, 9, '9LogisticRegression.ipynb'],
  [softmaxAssignment, 10, '10Softmax.ipynb'],
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
    expect(notebookAssignments).toHaveLength(7)

    for (const [assignment, displayNumber, filename] of expectedAssignments) {
      expect(assignment.published).toBe(true)
      expect(assignment.displayNumber).toBe(displayNumber)
      expect(assignment.questions.map((question) => question.kind)).toEqual(assignment.id === 'linear-regression' ? ['linearFit', 'planeFit', 'codeLab'] : assignment.id === 'polynomial-regression' ? ['multipleChoice', 'polynomialDegree', 'polynomialDegree', 'multipleChoice', 'multipleChoice', 'codeLab', 'codeLab'] : [
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
        if (question.kind === 'polynomialDegree') {
          expect(question.validator({ degree: question.target === 'training' ? 8 : 2 }).correct).toBe(true)
          expect(question.validator({ degree: 1 }).correct).toBe(false)
          continue
        }
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

  it('combines polynomial regression and overfitting with visual warmups and short notebook checkpoints', () => {
    expect(polynomialRegressionAssignment.version).toBe(3)
    expect(polynomialRegressionAssignment.title).toBe('Polynomial Regression and Overfitting')
    expect(notebookAssignments.some(a => a.id === 'overfitting')).toBe(false)
    expect(polynomialRegressionAssignment.questions[0].datasetId).toBe('polynomialCurves')
    expect(polynomialRegressionAssignment.questions[3].datasetId).toBe('polynomialErrors')
    for (const q of polynomialRegressionAssignment.questions.filter(q => q.kind === 'codeLab')) {
      expect(q.stages.every(s => s.fields.length === 1)).toBe(true)
      expect(q.prompt).toContain('5PolynomialRegression.ipynb')
      expect(q.code).not.toContain('from sklearn')
    }
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
