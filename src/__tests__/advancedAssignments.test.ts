/// <reference types="node" />
import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { advancedAssignments, advancedNotebookFilenames } from '../data/advancedAssignments'
import { publishedAssignments } from '../data/assignments'
import { multipleChoiceDatasets } from '../data/multipleChoiceDatasets'
import { getCorrectCodeLabSubmission } from '../lib/codeLab'
import { createInitialAssignmentState } from '../lib/assignmentState'
import { buildSubmissionExport } from '../lib/export'

describe('homework 11–25 modules', () => {
  it('publishes one complete module per supplied notebook in order', () => {
    expect(advancedAssignments).toHaveLength(15)
    expect(advancedAssignments.map(a => a.displayNumber)).toEqual(Array.from({ length: 15 }, (_, i) => i + 11))
    expect(new Set(publishedAssignments.map(a => a.id)).size).toBe(26)
    for (const assignment of advancedAssignments) {
      expect(assignment.version).toBe(1)
      expect(assignment.published).toBe(true)
      expect(assignment.questions.length).toBeGreaterThanOrEqual(4)
      const filename = advancedNotebookFilenames[assignment.displayNumber!]
      expect(existsSync(resolve('Homework2026', filename))).toBe(true)
      const notebook = JSON.parse(readFileSync(resolve('Homework2026', filename), 'utf8'))
      expect(notebook.cells.length).toBeGreaterThan(0)
      const last = assignment.questions.at(-1)!
      expect(last.kind).toBe('codeLab')
      expect(last.prompt).toContain(filename)
      expect(new Set(assignment.questions.map(q => q.id)).size).toBe(assignment.questions.length)
    }
  })

  for (const assignment of advancedAssignments) {
    it(`${assignment.id}: has self-contained visual warm-ups and small notebook checkpoints`, () => {
      for (const question of assignment.questions.slice(0, -1)) {
        expect(question.kind).toBe('multipleChoice')
        if (question.kind !== 'multipleChoice') throw new Error('Expected concept question')
        expect(multipleChoiceDatasets[question.datasetId]).toBeDefined()
        expect(`${question.prompt} ${question.instructions}`).not.toMatch(/\.ipynb|notebook|code cell/i)
        expect(question.parts.length).toBeGreaterThan(0)
        for (const part of question.parts) {
          expect(part.options.length).toBeGreaterThanOrEqual(3)
          expect(new Set(part.options.map(o => o.id)).size).toBe(part.options.length)
          expect(part.options.filter(o => o.id === part.correctOptionId)).toHaveLength(1)
        }
        for (const malformed of [null, [], {}, { selectedIds: null }, { selectedIds: [] }]) {
          expect(question.validator(malformed).correct).toBe(false)
        }
      }
      const lab = assignment.questions.at(-1)!
      if (lab.kind !== 'codeLab') throw new Error('Expected notebook lab')
      expect(lab.code.trim()).not.toBe('')
      expect(lab.fixture.trim()).not.toBe('')
      expect(lab.invocation.trim()).not.toBe('')
      expect(lab.stages.length).toBeGreaterThanOrEqual(4)
      expect(lab.stages.every(stage => stage.fields.length === 1)).toBe(true)
      expect(new Set(lab.stages.map(s => s.id)).size).toBe(lab.stages.length)
      const good = getCorrectCodeLabSubmission(lab)
      expect(lab.validator(good).correct).toBe(true)
      for (const stage of lab.stages) {
        const field = stage.fields[0]
        expect(field.options.filter(o => o.id === field.correctOptionId)).toHaveLength(1)
        for (const option of field.options) {
          const changed = structuredClone(good)
          changed.stages[stage.id][field.id] = option.id
          expect(lab.validator(changed).correct).toBe(option.id === field.correctOptionId)
        }
      }
    })

    it(`${assignment.id}: exports untouched work without a completion gate`, () => {
      const submission = buildSubmissionExport(assignment, createInitialAssignmentState(assignment))
      expect(submission.assignmentId).toBe(assignment.id)
      expect(submission.assignmentVersion).toBe(1)
      expect(submission.questions).toHaveLength(assignment.questions.length)
      expect(submission.questions.every(q => q.latestAnswer === null && q.attemptHistory.length === 0)).toBe(true)
    })
  }
})
