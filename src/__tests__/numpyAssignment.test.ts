import { describe, expect, it } from 'vitest'
import { numpyAssignment } from '../data/numpyAssignment'
import { getCorrectCodeLabSubmission } from '../lib/codeLab'

describe('numpyAssignment', () => {
  it('publishes four structured questions followed by the notebook lab', () => {
    expect(numpyAssignment.id).toBe('numpy')
    expect(numpyAssignment.version).toBe(1)
    expect(numpyAssignment.published).toBe(true)
    expect(numpyAssignment.questions).toHaveLength(5)
    expect(numpyAssignment.questions.map((question) => question.kind)).toEqual([
      'multipleChoice',
      'multipleChoice',
      'multipleChoice',
      'multipleChoice',
      'codeLab',
    ])
  })

  it('grades the construction and indexing trace from raw selections', () => {
    const question = numpyAssignment.questions[0]
    expect(
      question.validator({
        selectedIds: {
          'y-shape': 'shape-12',
          'Y-shape': 'shape-3-4',
          'indexed-value': 'value-7',
          'newaxis-shape': 'shape-12-1',
        },
      }).correct,
    ).toBe(true)
    expect(
      question.validator({
        selectedIds: {
          'y-shape': 'shape-1-12',
          'Y-shape': 'shape-3-4',
          'indexed-value': 'value-7',
          'newaxis-shape': 'shape-12-1',
        },
      }).correct,
    ).toBe(false)
  })

  it('autogrades every notebook-lab stage as structured IDs', () => {
    const question = numpyAssignment.questions[4]
    if (question.kind !== 'codeLab') {
      throw new Error('Expected the final NumPy question to be a Notebook Lab.')
    }

    const correct = getCorrectCodeLabSubmission(question)
    expect(question.validator(correct).correct).toBe(true)

    const changed = structuredClone(correct)
    changed.stages.MUTATION.CHANGED_MATRIX_SUMMARY = 'CHANGED_100_100_555'
    expect(question.validator(changed).correct).toBe(false)
  })
})
