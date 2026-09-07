import { describe, expect, it } from 'vitest'
import { kmeansAssignment } from '../data/kmeansAssignment'
import { kmeansInteractiveDatasets } from '../data/kmeansDatasets'
import { getCorrectCodeLabSubmission } from '../lib/codeLab'

describe('kmeansAssignment validators', () => {
  it('ends with a lab grounded in the renamed k-means notebook', () => {
    expect(kmeansAssignment.version).toBe(7)
    const lab = kmeansAssignment.questions.at(-1)
    if (lab?.kind !== 'codeLab') {
      throw new Error('Expected the final k-means question to be a Notebook Lab.')
    }

    expect(lab.title).toContain('Notebook Lab')
    expect(lab.prompt).toContain('2Kmeans.ipynb')
    expect(lab.code).toContain('def centers_from_labels(data,labels):')
    expect(lab.code).toContain('def labels_from_centers(data,centers):')
    expect(lab.code).not.toContain('def lloyd_step')
    expect(lab.validator(getCorrectCodeLabSubmission(lab)).correct).toBe(true)
  })

  it('checks both iterations of the multipart Lloyd question', () => {
    const question = kmeansAssignment.questions[0]
    const dataset = kmeansInteractiveDatasets.fullIteration
    const lastIteration = (dataset.iterationAssignments?.length ?? 1) - 1

    expect(
      question.validator({
        step: 'assignments',
        iteration: 0,
        assignments: dataset.iterationAssignments?.[0],
      }).correct,
    ).toBe(true)

    expect(
      question.validator({
        step: 'assignments',
        iteration: 0,
        assignments: Array.from({ length: dataset.points.length }, () => 0),
      }).correct,
    ).toBe(false)

    expect(
      question.validator({
        step: 'centroids',
        iteration: 0,
        centroids: dataset.iterationCentroids?.[0],
      }).correct,
    ).toBe(true)

    expect(
      question.validator({
        step: 'assignments',
        iteration: 1,
        assignments: dataset.iterationAssignments?.[1],
      }).correct,
    ).toBe(true)

    expect(
      question.validator({
        step: 'centroids',
        iteration: 1,
        centroids: dataset.iterationCentroids?.[1],
      }).correct,
    ).toBe(true)

    expect(
      question.validator({
        step: 'centroids',
        iteration: lastIteration,
        centroids: dataset.iterationCentroids?.[lastIteration],
      }).correct,
    ).toBe(true)

    expect(
      question.validator({
        step: 'centroids',
        iteration: lastIteration,
        centroids: dataset.initialCentroids,
      }).correct,
    ).toBe(false)
  })

  it('starts with one clear label per visible cloud while retaining a later reassignment', () => {
    const rounds = kmeansInteractiveDatasets.fullIteration.iterationAssignments ?? []

    expect(rounds[0]?.slice(0, 12)).toEqual([
      0, 0, 0, 0,
      1, 1, 1, 1,
      2, 2, 2, 2,
    ])
    expect(rounds[0]?.[13]).toBe(2)
    expect(rounds[1]?.[13]).toBe(1)
  })

  it('accepts a bad initialization that avoids the obvious clustering', () => {
    const question = kmeansAssignment.questions[1]

    expect(question.validator({ centroids: [[-7, 0], [5, 0], [7, 0]] }).correct).toBe(true)

    expect(question.validator({ centroids: [[-7, 0], [-1, 5], [7, 0]] }).correct).toBe(false)
  })

  it('checks both Euclidean and Manhattan assignments in the metric-comparison question', () => {
    const metricQuestion = kmeansAssignment.questions[2]
    const dataset = kmeansInteractiveDatasets.metricComparison

    expect(
      metricQuestion.validator({
        step: 'euclidean',
        assignments: dataset.targetAssignments,
      }).correct,
    ).toBe(true)

    expect(
      metricQuestion.validator({
        step: 'manhattan',
        assignments: dataset.comparisonAssignments,
      }).correct,
    ).toBe(true)

    expect(
      metricQuestion.validator({
        step: 'manhattan',
        assignments: dataset.targetAssignments,
      }).correct,
    ).toBe(false)
  })

  it('accepts the elbow choice', () => {
    const elbowQuestion = kmeansAssignment.questions[3]

    expect(
      elbowQuestion.validator({
        selectedIds: {
          'best-k': 'k4',
        },
      }).correct,
    ).toBe(false)
  })
})
