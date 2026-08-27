import { describe, expect, it } from 'vitest'
import { pcaAssignment } from '../data/pcaAssignment'
import {
  applyAttemptOutcome,
  applyGiveUp,
  createInitialAssignmentState,
  getVisibleHints,
  hydrateAssignmentState,
  isAssignmentComplete,
} from '../lib/assignmentState'

describe('assignmentState', () => {
  it('starts with only the first question active', () => {
    const state = createInitialAssignmentState(pcaAssignment)
    expect(state.questionStates[0].status).toBe('active')
    expect(state.questionStates[1].status).toBe('locked')
  })

  it('unlocks hints based on incorrect attempts and unlocks the next question on success', () => {
    let state = createInitialAssignmentState(pcaAssignment)
    const question = pcaAssignment.questions[0]

    state = applyAttemptOutcome(state, pcaAssignment, 0, 'incorrect', { direction: [1, 0] })
    state = applyAttemptOutcome(state, pcaAssignment, 0, 'incorrect', { direction: [0.8, 0.2] })
    expect(state.questionStates[0].incorrectAttempts).toBe(2)
    expect(getVisibleHints(question, state.questionStates[0])).toHaveLength(1)

    state = applyAttemptOutcome(state, pcaAssignment, 0, 'correct', { direction: [0.6, 0.8] })
    expect(state.questionStates[0].status).toBe('correct')
    expect(state.questionStates[0].attempts).toBe(3)
    expect(state.questionStates[0].latestAnswer).toEqual({ direction: [0.6, 0.8] })
    expect(state.questionStates[0].attemptHistory).toHaveLength(3)
    expect(state.questionStates[0].attemptHistory[0]).toMatchObject({
      attemptNumber: 1,
      outcome: 'incorrect',
      answer: { direction: [1, 0] },
    })
    expect(state.questionStates[1].status).toBe('active')
  })

  it('records give up without adding an attempt and unlocks the next question', () => {
    let state = createInitialAssignmentState(pcaAssignment)
    state = applyGiveUp(state, pcaAssignment, 0, { direction: [0.25, 0.97] })

    expect(state.questionStates[0].status).toBe('gave_up')
    expect(state.questionStates[0].attempts).toBe(0)
    expect(state.questionStates[0].latestAnswer).toEqual({ direction: [0.25, 0.97] })
    expect(state.questionStates[0].attemptHistory).toEqual([])
    expect(state.questionStates[1].status).toBe('active')
  })

  it('detects when an assignment is fully resolved', () => {
    let state = createInitialAssignmentState(pcaAssignment)

    for (let index = 0; index < pcaAssignment.questions.length; index += 1) {
      state = applyGiveUp(state, pcaAssignment, index)
    }

    expect(isAssignmentComplete(state)).toBe(true)
  })

  it('resets saved progress when assignment content changes version', () => {
    const saved = createInitialAssignmentState(pcaAssignment)
    const revisedAssignment = { ...pcaAssignment, version: pcaAssignment.version + 1 }
    const hydrated = hydrateAssignmentState(revisedAssignment, saved)

    expect(hydrated.assignmentVersion).toBe(revisedAssignment.version)
    expect(hydrated.questionStates[0].attemptHistory).toEqual([])
  })
})
