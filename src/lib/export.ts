import type { AssignmentSpec, AssignmentState, SubmissionExport } from '../types'

export function buildSubmissionExport(
  assignment: AssignmentSpec,
  state: AssignmentState,
): SubmissionExport {
  return {
    formatVersion: 2,
    assignmentId: assignment.id,
    assignmentVersion: assignment.version,
    generatedAt: new Date().toISOString(),
    questions: assignment.questions.map((question, index) => {
      const questionState = state.questionStates[index]

      return {
        id: question.id,
        status: questionState.status === 'correct' ? 'correct' : 'gave_up',
        attempts: questionState.attempts,
        hintsShown: questionState.hintsShown,
        latestAnswer: questionState.latestAnswer,
        attemptHistory: questionState.attemptHistory,
      }
    }),
  }
}

export function downloadSubmissionExport(
  assignment: AssignmentSpec,
  state: AssignmentState,
) {
  const payload = JSON.stringify(buildSubmissionExport(assignment, state), null, 2)
  const blob = new Blob([payload], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `${assignment.id}-submission.json`
  anchor.click()
  URL.revokeObjectURL(url)
}
