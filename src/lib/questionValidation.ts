import type { ValidationResult } from '../types'

export function validateMultipleChoiceSelections(
  correctSelections: Record<string, string>,
  incorrectMessage = 'That selection is not right yet.',
) {
  return (submission: unknown): ValidationResult => {
    if (!submission || typeof submission !== 'object' || Array.isArray(submission)) {
      return { correct: false, message: incorrectMessage }
    }
    const payload = submission as { selectedIds?: unknown }
    if (!payload.selectedIds || typeof payload.selectedIds !== 'object' || Array.isArray(payload.selectedIds)) {
      return { correct: false, message: incorrectMessage }
    }
    const selectedIds = payload.selectedIds as Record<string, unknown>
    const allCorrect = Object.entries(correctSelections).every(
      ([partId, answerId]) => selectedIds[partId] === answerId,
    )

    return {
      correct: allCorrect,
      message: allCorrect ? 'Correct! Good Job!' : incorrectMessage,
    }
  }
}
