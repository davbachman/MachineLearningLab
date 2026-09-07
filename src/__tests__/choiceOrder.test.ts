import { describe, expect, it } from 'vitest'
import { assignmentRegistry } from '../data/assignments'
import { kmeansAssignment } from '../data/kmeansAssignment'
import { withBalancedChoiceOrder } from '../lib/choiceOrder'
import { getCorrectCodeLabSubmission } from '../lib/codeLab'

interface ChoiceField {
  options: { id: string }[]
  correctOptionId: string
}

function expectBalanced(fields: ChoiceField[]) {
  for (const size of new Set(fields.map((field) => field.options.length))) {
    const counts = Array<number>(size).fill(0)
    for (const field of fields.filter((field) => field.options.length === size)) {
      const position = field.options.findIndex((option) => option.id === field.correctOptionId)
      expect(position).toBeGreaterThanOrEqual(0)
      counts[position] += 1
    }
    expect(Math.max(...counts) - Math.min(...counts)).toBeLessThanOrEqual(1)
  }
}

describe('assignment choice ordering', () => {
  for (const assignment of assignmentRegistry) {
    it(`balances correct positions and preserves grading in ${assignment.id}`, () => {
      expectBalanced(assignment.questions.flatMap((question) =>
        question.kind === 'multipleChoice' ? question.parts : [],
      ))
      for (const question of assignment.questions) {
        if (question.kind === 'codeLab') {
          expectBalanced(question.stages.flatMap((stage) => stage.fields))
          expect(question.validator(getCorrectCodeLabSubmission(question)).correct).toBe(true)
        } else if (question.kind === 'multipleChoice') {
          expect(question.validator({
            selectedIds: Object.fromEntries(question.parts.map((part) => [part.id, part.correctOptionId])),
          }).correct).toBe(true)
        }
      }
    })
  }

  it('is deterministic and preserves source choices, IDs, and assignment versions', () => {
    const original = JSON.stringify(kmeansAssignment)
    const first = withBalancedChoiceOrder(kmeansAssignment)
    expect(withBalancedChoiceOrder(kmeansAssignment)).toEqual(first)
    expect(JSON.stringify(kmeansAssignment)).toBe(original)
    expect(first.version).toBe(kmeansAssignment.version)

    for (const [index, question] of first.questions.entries()) {
      const source = kmeansAssignment.questions[index]
      if (question.kind === 'codeLab' && source.kind === 'codeLab') {
        expect(question.variantId).toBe(source.variantId)
        const before = source.stages.flatMap((stage) => stage.fields)
        const after = question.stages.flatMap((stage) => stage.fields)
        after.forEach((field, fieldIndex) => {
          expect(field.correctOptionId).toBe(before[fieldIndex].correctOptionId)
          expect(field.options).toHaveLength(before[fieldIndex].options.length)
          expect(field.options).toEqual(expect.arrayContaining(before[fieldIndex].options))
        })
      }
    }
  })
})
