import { describe, expect, it } from 'vitest'
import { pcaAssignment } from '../data/pcaAssignment'
import {
  applyAttemptOutcome,
  createInitialAssignmentState,
  getVisibleHints,
  hydrateAssignmentState,
  isAssignmentComplete,
} from '../lib/assignmentState'

describe('assignmentState', () => {
  it('starts with every question available', () => {
    const state = createInitialAssignmentState(pcaAssignment)
    expect(state.questionStates.every((questionState) => questionState.status === 'active')).toBe(true)
  })

  it('unlocks hints based on incorrect attempts and preserves free navigation on success', () => {
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

  it('reopens a completed question when a redo attempt is incorrect', () => {
    let state = createInitialAssignmentState(pcaAssignment)
    state = applyAttemptOutcome(state, pcaAssignment, 0, 'correct', { direction: [0.6, 0.8] })
    state = applyAttemptOutcome(state, pcaAssignment, 0, 'incorrect', { direction: [1, 0] })

    expect(state.questionStates[0].status).toBe('active')
    expect(state.questionStates[0].attempts).toBe(2)
    expect(state.questionStates[0].latestAnswer).toEqual({ direction: [1, 0] })
    expect(state.questionStates[0].attemptHistory.map((attempt) => attempt.outcome)).toEqual([
      'correct',
      'incorrect',
    ])
  })

  it('detects when an assignment is fully resolved', () => {
    let state = createInitialAssignmentState(pcaAssignment)

    for (let index = 0; index < pcaAssignment.questions.length; index += 1) {
      state = applyAttemptOutcome(state, pcaAssignment, index, 'correct', { answer: index })
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

  it('migrates legacy locked questions to available questions', () => {
    const saved = createInitialAssignmentState(pcaAssignment)
    saved.questionStates[1] = { ...saved.questionStates[1], status: 'locked' }

    const hydrated = hydrateAssignmentState(pcaAssignment, saved)

    expect(hydrated.questionStates[1].status).toBe('active')
  })
})
