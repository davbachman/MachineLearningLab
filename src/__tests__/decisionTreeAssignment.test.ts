import { describe, expect, it } from 'vitest'
import { decisionTreeAssignment } from '../data/decisionTreeAssignment'
import { publishedAssignments } from '../data/assignments'

describe('decisionTreeAssignment', () => {
  it('keeps seven structured decision-tree questions while hiding the assignment', () => {
    expect(decisionTreeAssignment.id).toBe('decision-trees')
    expect(decisionTreeAssignment.version).toBe(7)
    expect(decisionTreeAssignment.published).toBe(false)
    expect(
      decisionTreeAssignment.questions.filter((question) => question.kind === 'decisionTree'),
    ).toHaveLength(7)
    expect(decisionTreeAssignment.questions.map((question) => question.id)).toEqual(
      expect.arrayContaining(['dt-visual-axis-split', 'dt-visual-depth-two']),
    )
    expect(publishedAssignments.map((assignment) => assignment.id)).toEqual(['pca', 'kmeans', 'knn'])
  })

  it('accepts the warm-up Gini values and rejects swapped impurity answers', () => {
    const question = decisionTreeAssignment.questions[0]

    expect(
      question.validator({
        values: {
          a: 0,
          b: 0.375,
          c: 0.5,
          d: 0.375,
        },
      }).correct,
    ).toBe(true)

    expect(
      question.validator({
        values: {
          a: 0,
          b: 0.5,
          c: 0.375,
          d: 0.375,
        },
      }).correct,
    ).toBe(false)
  })

  it('requires row routing and counts before accepting the split-score arithmetic', () => {
    const question = decisionTreeAssignment.questions[1]
    const sharedTrace = {
      leftRowIds: ['A', 'B', 'C', 'D'],
      counts: {
        left: { negative: 3, positive: 1 },
        right: { negative: 1, positive: 3 },
      },
      values: { left: 0.375, right: 0.375 },
    }

    expect(
      question.validator({
        step: 'partition',
        leftRowIds: sharedTrace.leftRowIds,
      }).correct,
    ).toBe(true)

    expect(
      question.validator({
        step: 'partition',
        leftRowIds: ['A', 'B', 'C'],
      }).correct,
    ).toBe(false)

    expect(
      question.validator({
        step: 'childStats',
        ...sharedTrace,
      }).correct,
    ).toBe(true)

    expect(
      question.validator({
        step: 'weightedGini',
        ...sharedTrace,
        value: 0.375,
      }).correct,
    ).toBe(true)
  })

  it('accepts only the lowest weighted-Gini split for the root comparison', () => {
    const question = decisionTreeAssignment.questions[3]

    expect(question.validator({ splitId: 'attendance-81' }).correct).toBe(true)
    expect(question.validator({ splitId: 'hours-9-5' }).correct).toBe(false)
  })

  it('requires the genuinely lowest-Gini line in the single-split scatterplot', () => {
    const question = decisionTreeAssignment.questions[4]

    expect(question.validator({ featureId: 'x', threshold: 2.5 }).correct).toBe(true)
    expect(question.validator({ featureId: 'x', threshold: 4.5 }).correct).toBe(false)
    expect(question.validator({ featureId: 'y', threshold: 3.5 }).correct).toBe(false)
  })

  it('grades both choices in the geometric depth-two tree', () => {
    const question = decisionTreeAssignment.questions[5]

    expect(
      question.validator({ rootSplitId: 'x-4-5', rightChildSplitId: 'y-3-5' }).correct,
    ).toBe(true)
    expect(
      question.validator({ rootSplitId: 'x-3-5', rightChildSplitId: 'y-3-5' }).correct,
    ).toBe(false)
  })
})
