import type { ValidationResult } from '../types'

export function validateMultipleChoiceSelections(
  correctSelections: Record<string, string>,
  incorrectMessage = 'That selection is not right yet.',
) {
  return (submission: unknown): ValidationResult => {
    const payload = submission as { selectedIds?: Record<string, string | undefined> }
    const selectedIds = payload.selectedIds ?? {}
    const allCorrect = Object.entries(correctSelections).every(
      ([partId, answerId]) => selectedIds[partId] === answerId,
    )

    return {
      correct: allCorrect,
      message: allCorrect ? 'Correct! Good Job!' : incorrectMessage,
    }
  }
}
