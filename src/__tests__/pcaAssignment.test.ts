import { describe, expect, it } from 'vitest'
import { pcaAssignment } from '../data/pcaAssignment'
import { pca2dDatasets, pca3dDatasets } from '../data/pcaDatasets'

describe('pcaAssignment validators', () => {
  it('references the renamed PCA notebook', () => {
    expect(pcaAssignment.version).toBe(7)
    expect(pcaAssignment.questions.at(-1)?.kind).toBe('codeLab')
    expect(pcaAssignment.questions.at(-1)?.prompt).toContain('1PCA.ipynb')
  })

  it('accepts the optimal uncentered 2D direction and rejects an orthogonal guess', () => {
    const question = pcaAssignment.questions[0]
    const dataset = pca2dDatasets.uncentered

    expect(question.validator({ direction: dataset.answerDirection }).correct).toBe(true)
    expect(question.validator({ direction: [-dataset.answerDirection[1], dataset.answerDirection[0]] }).correct).toBe(false)
  })

  it('rejects the no-outlier reference direction for the outlier-sensitive question', () => {
    const question = pcaAssignment.questions[3]
    const dataset = pca2dDatasets.outlierSensitive

    expect(question.validator({ direction: dataset.answerDirection }).correct).toBe(true)
    expect(question.validator({ direction: dataset.comparisonDirection }).correct).toBe(false)
  })

  it('accepts the standardize choice and rejects center-only', () => {
    const question = pcaAssignment.questions[4]

    expect(
      question.validator({
        selectedIds: {
          'pca-behavior': 'price-dominates',
          'pca-fix': 'standardize',
        },
      }).correct,
    ).toBe(true)
    expect(
      question.validator({
        selectedIds: {
          'pca-behavior': 'equal-weight',
          'pca-fix': 'recenter',
        },
      }).correct,
    ).toBe(false)
  })

  it('accepts the optimal PCA plane and rejects a poor one', () => {
    const question = pcaAssignment.questions[5]
    const dataset = pca3dDatasets.centered3d

    expect(question.validator({ first: dataset.answerFirst, second: dataset.answerSecond }).correct).toBe(true)
    expect(question.validator({ first: [1, 0, 0], second: [0, 1, 0] }).correct).toBe(false)
  })

  it('validates each hand-calculation substep', () => {
    const question = pcaAssignment.questions[6]

    expect(
      question.validator({
        step: 'centeredData',
        values: [
          [-3, -6],
          [-1, -2],
          [1, 2],
          [3, 6],
        ],
      }).correct,
    ).toBe(true)

    expect(
      question.validator({
        step: 'covariance',
        values: [
          [5, 10],
          [10, 20],
        ],
      }).correct,
    ).toBe(true)

    expect(question.validator({ step: 'direction', values: [1, 2] }).correct).toBe(true)
    expect(question.validator({ step: 'direction', values: [1, -1] }).correct).toBe(false)
  })
})
