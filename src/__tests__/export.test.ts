import { describe, expect, it } from 'vitest'
import { pcaAssignment } from '../data/pcaAssignment'
import { applyAttemptOutcome, createInitialAssignmentState } from '../lib/assignmentState'
import { buildSubmissionExport } from '../lib/export'

describe('submission export', () => {
  it('exports raw answers and the complete structured attempt history', () => {
    let state = createInitialAssignmentState(pcaAssignment)
    state = applyAttemptOutcome(state, pcaAssignment, 0, 'incorrect', {
      direction: [1, 0],
      optionalValue: undefined,
    })
    state = applyAttemptOutcome(state, pcaAssignment, 0, 'correct', {
      direction: [0.6, 0.8],
    })

    const exported = buildSubmissionExport(pcaAssignment, state)

    expect(exported.formatVersion).toBe(2)
    expect(exported.assignmentVersion).toBe(pcaAssignment.version)
    expect(exported.questions[0].latestAnswer).toEqual({ direction: [0.6, 0.8] })
    expect(exported.questions[0].attemptHistory).toHaveLength(2)
    expect(exported.questions[0].attemptHistory[0].answer).toEqual({
      direction: [1, 0],
      optionalValue: null,
    })
  })
})
