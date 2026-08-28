import type {
  AssignmentSpec,
  AssignmentState,
  AttemptOutcome,
  JsonValue,
  QuestionSpec,
  QuestionState,
} from '../types'

export const ASSIGNMENT_STATE_VERSION = 2
export const DEFAULT_HINT_SCHEDULE = [3, 6]

export function toJsonValue(value: unknown): JsonValue {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') {
    return value
  }

  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null
  }

  if (Array.isArray(value)) {
    return value.map(toJsonValue)
  }

  if (typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, toJsonValue(entry)]),
    )
  }

  return null
}

function createQuestionState(status: QuestionState['status']): QuestionState {
  return {
    status,
    attempts: 0,
    incorrectAttempts: 0,
    hintsShown: 0,
    resolvedAt: null,
    latestAnswer: null,
    attemptHistory: [],
  }
}

export function createInitialAssignmentState(assignment: AssignmentSpec): AssignmentState {
  return {
    assignmentId: assignment.id,
    assignmentVersion: assignment.version,
    version: ASSIGNMENT_STATE_VERSION,
    questionStates: assignment.questions.map(() => createQuestionState('active')),
    exportedAt: null,
  }
}

function deriveHintsShown(question: QuestionSpec, incorrectAttempts: number) {
  const schedule = question.hintSchedule ?? DEFAULT_HINT_SCHEDULE
  return schedule.filter((attemptThreshold) => incorrectAttempts >= attemptThreshold).length
}

export function hydrateAssignmentState(
  assignment: AssignmentSpec,
  candidate: AssignmentState | null,
): AssignmentState {
  if (
    !candidate ||
    candidate.assignmentId !== assignment.id ||
    candidate.assignmentVersion !== assignment.version ||
    candidate.version !== ASSIGNMENT_STATE_VERSION
  ) {
    return createInitialAssignmentState(assignment)
  }

  if (candidate.questionStates.length !== assignment.questions.length) {
    return createInitialAssignmentState(assignment)
  }

  return {
    ...candidate,
    questionStates: candidate.questionStates.map((questionState) =>
      questionState.status === 'locked'
        ? { ...questionState, status: 'active' }
        : questionState,
    ),
  }
}

export function applyAttemptOutcome(
  state: AssignmentState,
  assignment: AssignmentSpec,
  questionIndex: number,
  outcome: AttemptOutcome,
  answer: unknown = null,
): AssignmentState {
  const question = assignment.questions[questionIndex]
  const current = state.questionStates[questionIndex]

  if (!question || !current) {
    return state
  }

  const incorrectAttempts = current.incorrectAttempts + (outcome === 'incorrect' ? 1 : 0)
  const normalizedAnswer = toJsonValue(answer)
  const nextCurrentState: QuestionState = {
    ...current,
    attempts: current.attempts + 1,
    incorrectAttempts,
    hintsShown: deriveHintsShown(question, incorrectAttempts),
    status: outcome === 'correct' ? 'correct' : 'active',
    resolvedAt: outcome === 'correct' ? new Date().toISOString() : null,
    latestAnswer: normalizedAnswer,
    attemptHistory: [
      ...current.attemptHistory,
      {
        attemptNumber: current.attempts + 1,
        outcome,
        submittedAt: new Date().toISOString(),
        answer: normalizedAnswer,
      },
    ],
  }

  const questionStates = state.questionStates.map<QuestionState>((questionState, index) =>
    index === questionIndex ? nextCurrentState : questionState,
  )

  return {
    ...state,
    questionStates,
  }
}

export function applyGiveUp(
  state: AssignmentState,
  assignment: AssignmentSpec,
  questionIndex: number,
  answer: unknown = null,
): AssignmentState {
  const current = state.questionStates[questionIndex]
  if (!assignment.questions[questionIndex] || !current) {
    return state
  }

  const questionStates = state.questionStates.map<QuestionState>((questionState, index) =>
    index === questionIndex
      ? {
          ...questionState,
          status: 'gave_up',
          resolvedAt: new Date().toISOString(),
          latestAnswer: toJsonValue(answer),
        }
      : questionState,
  )

  return {
    ...state,
    questionStates,
  }
}

export function clearAssignmentProgress(assignment: AssignmentSpec) {
  return createInitialAssignmentState(assignment)
}

export function isAssignmentComplete(state: AssignmentState) {
  return state.questionStates.every(
    (questionState) => questionState.status === 'correct' || questionState.status === 'gave_up',
  )
}

export function getVisibleHints(question: QuestionSpec, questionState: QuestionState) {
  return question.hints.slice(0, questionState.hintsShown)
}

export function getResolvedCount(state: AssignmentState) {
  return state.questionStates.filter(
    (questionState) => questionState.status === 'correct' || questionState.status === 'gave_up',
  ).length
}

export function getCurrentOrFirstActiveIndex(state: AssignmentState) {
  const activeIndex = state.questionStates.findIndex((questionState) => questionState.status === 'active')
  if (activeIndex >= 0) {
    return activeIndex
  }
  return state.questionStates.length - 1
}

export function markAssignmentExported(state: AssignmentState): AssignmentState {
  return {
    ...state,
    exportedAt: new Date().toISOString(),
  }
}
