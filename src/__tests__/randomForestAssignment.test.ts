import { describe, expect, it } from 'vitest'
import { randomForestAssignment } from '../data/randomForestAssignment'
import { getCorrectCodeLabSubmission } from '../lib/codeLab'

describe('randomForestAssignment', () => {
  it('keeps six structured questions while hiding the assignment', () => {
    expect(randomForestAssignment.id).toBe('random-forests')
    expect(randomForestAssignment.version).toBe(6)
    expect(randomForestAssignment.published).toBe(false)
    expect(
      randomForestAssignment.questions.map((question) =>
        question.kind === 'randomForest' ? question.interactionMode : question.kind,
      ),
    ).toEqual([
      'bootstrapAudit',
      'featureSubsamplingGeometry',
      'forestVoteGeometry',
      'oobEstimate',
      'varianceReduction',
      'codeLab',
    ])
  })

  it('validates the bootstrap multiplicities and OOB IDs as raw structured data', () => {
    const question = randomForestAssignment.questions[0]
    expect(
      question.validator({
        multiplicities: { A: 2, B: 0, C: 3, D: 1, E: 0, F: 1, G: 0, H: 1 },
        oobIds: ['B', 'E', 'G'],
      }).correct,
    ).toBe(true)
    expect(
      question.validator({
        multiplicities: { A: 1, B: 0, C: 3, D: 1, E: 0, F: 1, G: 0, H: 1 },
        oobIds: ['B', 'E', 'G'],
      }).correct,
    ).toBe(false)
  })

  it('requires both nontrivial feature-limited splits and the probe traces', () => {
    const question = randomForestAssignment.questions[1]
    expect(
      question.validator({
        selectedSplits: {
          'tree-a': 'tree-a-x-3.5',
          'tree-b': 'tree-b-y-4.5',
        },
        predictions: {
          P: { 'tree-a': 0, 'tree-b': 1 },
          Q: { 'tree-a': 1, 'tree-b': 0 },
        },
      }).correct,
    ).toBe(true)
    expect(
      question.validator({
        selectedSplits: {
          'tree-a': 'tree-a-x-3.5',
          'tree-b': 'tree-b-y-4.5',
        },
        predictions: {
          P: { 'tree-a': 0, 'tree-b': 0 },
          Q: { 'tree-a': 1, 'tree-b': 0 },
        },
      }).correct,
    ).toBe(false)
  })

  it('validates forest votes, OOB estimates, and correlation-aware variance separately', () => {
    expect(
      randomForestAssignment.questions[2].validator({
        predictions: { P: 0, Q: 0, R: 1, S: 1, T: 1 },
      }).correct,
    ).toBe(true)

    expect(
      randomForestAssignment.questions[3].validator({
        predictions: { A: 0, B: 0, C: 0, D: 1, E: 1, F: 1, G: 1, H: 0 },
        accuracy: 0.75,
      }).correct,
    ).toBe(true)

    expect(
      randomForestAssignment.questions[4].validator({
        variances: { A: 0.028, B: 0.109, C: 0.02 },
        bestScenarioId: 'C',
        interventionId: 'feature-subsampling',
      }).correct,
    ).toBe(true)
  })

  it('autogrades the Code Lab trace without free-response text', () => {
    const question = randomForestAssignment.questions[5]
    if (question.kind !== 'codeLab') {
      throw new Error('Expected the final Random Forests question to be a Code Lab.')
    }

    const correct = getCorrectCodeLabSubmission(question)
    expect(question.validator(correct).correct).toBe(true)

    const changed = structuredClone(correct)
    changed.stages.MUTATION.WITHOUT_REPLACEMENT_EFFECT = 'DUPLICATES_WITH_NO_OMISSIONS'
    expect(question.validator(changed).correct).toBe(false)
  })
})
