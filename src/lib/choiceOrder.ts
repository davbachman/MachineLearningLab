import type { AssignmentSpec } from '../types'

interface ChoiceField {
  options: { id: string }[]
  correctOptionId: string
}

function seededRandom(seed: string) {
  let state = 2166136261
  for (const character of seed) {
    state = Math.imul(state ^ character.charCodeAt(0), 16777619)
  }
  return () => {
    state += 0x6d2b79f5
    let value = Math.imul(state ^ (state >>> 15), 1 | state)
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value)
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296
  }
}

function shuffle<T>(values: T[], random: () => number): T[] {
  const result = [...values]
  for (let index = result.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1))
    ;[result[index], result[other]] = [result[other], result[index]]
  }
  return result
}

/** Balance answer positions for each option count, then shuffle their sequence. */
function balanceChoices<T extends ChoiceField>(fields: T[], seed: string): T[] {
  const random = seededRandom(seed)
  const positionsBySize = new Map<number, number[]>()
  for (const size of new Set(fields.map((field) => field.options.length))) {
    const count = fields.filter((field) => field.options.length === size).length
    const slots = shuffle(Array.from({ length: size }, (_, index) => index), random)
    positionsBySize.set(size, shuffle(Array.from({ length: count }, (_, index) => slots[index % size]), random))
  }

  return fields.map((field) => {
    const correct = field.options.find((option) => option.id === field.correctOptionId)
    if (!correct) throw new Error(`Missing correct option: ${field.correctOptionId}`)
    const options = shuffle(field.options.filter((option) => option.id !== field.correctOptionId), random)
    const position = positionsBySize.get(field.options.length)!.pop()!
    options.splice(position, 0, correct)
    return { ...field, options }
  })
}

/**
 * Order choices once when the registry loads. Stable seeds keep refreshes and
 * retries consistent; grading and saved submissions continue to use option IDs.
 * Each lab is balanced separately, as are the regular questions in an assignment.
 */
export function withBalancedChoiceOrder(assignment: AssignmentSpec): AssignmentSpec {
  const parts = balanceChoices(
    assignment.questions.flatMap((question) => question.kind === 'multipleChoice' ? question.parts : []),
    `${assignment.id}/multiple-choice`,
  )
  let partIndex = 0

  return {
    ...assignment,
    questions: assignment.questions.map((question) => {
      if (question.kind === 'multipleChoice') {
        return { ...question, parts: question.parts.map(() => parts[partIndex++]) }
      }
      if (question.kind === 'codeLab') {
        const fields = balanceChoices(
          question.stages.flatMap((stage) => stage.fields),
          `${assignment.id}/${question.id}`,
        )
        let fieldIndex = 0
        return {
          ...question,
          stages: question.stages.map((stage) => ({
            ...stage,
            fields: stage.fields.map(() => fields[fieldIndex++]),
          })) as typeof question.stages,
        }
      }
      return question
    }),
  }
}
