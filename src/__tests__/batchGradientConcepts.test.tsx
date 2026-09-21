import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { batchGradientDescentAssignment, batchGradientVisualDatasets } from '../data/batchGradientAssignment'
import { batchGradientDescentNotebookLab } from '../data/classificationCodeLabQuestions'
import { BatchGradientVisual } from '../components/questions/BatchGradientVisual'
import { MultipleChoiceQuestion } from '../components/questions/MultipleChoiceQuestion'
import type { QuestionState } from '../types'
import { allBatchRows, batchGradient, batchLearningRate, batchLoss, batchMean, batchOptimum, batchOrders, batchSizes, batchTargets, batchTrajectory, shuffleSchedules, updatesPerEpoch } from '../lib/batchGradientConcepts'

afterEach(cleanup)

describe('conceptual batch gradient descent assignment', () => {
  it('keeps implementation details inside the unchanged notebook lab', () => {
    expect(batchGradientDescentAssignment.version).toBe(2)
    expect(batchGradientDescentAssignment.displayNumber).toBe(7)
    expect(batchGradientDescentAssignment.questions).toHaveLength(6)
    expect(batchGradientDescentAssignment.questions.at(-1)).toBe(batchGradientDescentNotebookLab)
    for (const question of batchGradientDescentAssignment.questions.slice(0, -1)) {
      expect(question.kind).toBe('multipleChoice')
      expect(JSON.stringify(question)).not.toMatch(/notebook|ipynb|numpy|SGDRegressor|self\.|np\./i)
      expect(batchGradientVisualDatasets[question.datasetId!].kind).toBe('batchGradientVisual')
      if (question.kind === 'multipleChoice') {
        const selectedIds = Object.fromEntries(question.parts.map(part => [part.id, part.correctOptionId]))
        expect(question.validator({ selectedIds }).correct).toBe(true)
        expect(question.validator({ selectedIds: {} }).correct).toBe(false)
        for (const malformed of [null, undefined, false, 5, [], {}, { selectedIds: null }, { selectedIds: [] }, { selectedIds: 'answer' }, { selectedIds: { ...selectedIds, [question.parts[0].id]: 1 } }]) {
          expect(question.validator(malformed).correct).toBe(false)
        }
        for (const part of question.parts) for (const option of part.options) {
          const changed = { ...selectedIds, [part.id]: option.id }
          expect(question.validator({ selectedIds: changed }).correct).toBe(option.id === part.correctOptionId)
        }
      }
    }
  })

  it('derives gradients and the optimum from the displayed targets', () => {
    expect(batchMean(allBatchRows)).toBe(4)
    expect(batchOptimum).toBe(4)
    expect(batchLoss(4)).toBe(3.75)
    expect(batchGradient(3, [0, 1, 2, 3])).toBe(1.5)
    expect(batchGradient(3, [4, 5, 6, 7])).toBe(-3.5)
    expect(batchGradient(3)).toBe(-1)
    for (const w of [-1, 0, 3, 4, 6, 9]) for (const rows of [allBatchRows, [0, 1, 2, 3], [4, 5, 6, 7], [6]]) {
      const independentGradient = rows.reduce((sum, row) => sum + w - batchTargets[row], 0) / rows.length
      expect(batchGradient(w, rows)).toBeCloseTo(independentGradient, 12)
      const h = 1e-5
      expect(batchGradient(w, rows)).toBeCloseTo((batchLoss(w + h, rows) - batchLoss(w - h, rows)) / (2 * h), 7)
    }
  })

  it('uses every observation once per epoch and holds shuffled orders fixed across batch sizes', () => {
    for (const order of batchOrders) expect([...order].sort((a, b) => a - b)).toEqual(allBatchRows)
    for (const size of batchSizes) {
      const run = batchTrajectory(size)
      expect(run).toHaveLength(32 / size)
      expect(run.at(-1)!.rowsProcessed).toBe(32)
      for (let epoch = 1; epoch <= 4; epoch++) {
        const steps = run.filter(step => step.epoch === epoch)
        expect(steps.flatMap(step => step.rows)).toEqual(batchOrders[epoch - 1])
        expect(steps).toHaveLength(8 / size)
      }
      // Every selected-batch objective must decrease (or already be stationary)
      // under the fixed rate; the FULL objective need not decrease.
      run.forEach(step => expect(step.batchLossAfter).toBeLessThanOrEqual(step.batchLossBefore + 1e-12))
      run.forEach(step => expect(step.fullLoss).toBeGreaterThanOrEqual(3.75 - 1e-12))
      expect(Math.max(...run.map(step => step.fullLoss))).toBeLessThan(12)
      expect(batchTrajectory(size)).toEqual(run)
    }
  })

  it('independently verifies update 5 and the full-data recurrence', () => {
    // w <- 0.6w + 0.4y for the first five targets 0, 7, 3, 6, 1.
    let w = 0
    const expected = [0, 7, 3, 6, 1].map(target => { w = 0.6 * w + 0.4 * target; return w })
    batchTrajectory(1).slice(0, 5).forEach((step, i) => expect(step.w).toBeCloseTo(expected[i], 12))
    const fifth = batchTrajectory(1)[4]
    expect(fifth.rows).toEqual([1])
    expect(fifth.before).toBeCloseTo(4.128, 12)
    expect(fifth.w).toBeCloseTo(2.8768, 12)
    expect(fifth.batchLossBefore).toBeCloseTo(4.892192, 12)
    expect(fifth.batchLossAfter).toBeCloseTo(1.76118912, 12)
    expect(fifth.fullLossBefore).toBeCloseTo(3.758192, 12)
    expect(fifth.fullLoss).toBeCloseTo(4.38078912, 12)
    expect(fifth.fullLoss).toBeGreaterThan(fifth.fullLossBefore)
    batchTrajectory(8).forEach((step, i) => {
      expect(step.w).toBeCloseTo(4 * (1 - 0.6 ** (i + 1)), 12)
      expect(step.fullLoss).toBeCloseTo(3.75 + 8 * 0.6 ** (2 * (i + 1)), 12)
      expect(step.fullLoss).toBeLessThan(step.fullLossBefore)
    })
  })

  it('validates the equal-epoch counts and the actual shuffled groups', () => {
    expect(8 * updatesPerEpoch(120, 20)).toBe(48)
    expect(8 * updatesPerEpoch(120, 120)).toBe(8)
    expect(updatesPerEpoch(125, 20)).toBe(7)
    expect(48 * 20).toBe(8 * 120)
    const [ordered, shuffled] = shuffleSchedules.map(schedule => [0, 2, 4, 6].map(start => batchMean(schedule.rows.slice(start, start + 2))))
    expect(ordered).toEqual([0.5, 2.5, 5.5, 7.5])
    expect(shuffled).toEqual([3.5, 4.5, 4.5, 3.5])
    expect(shuffled.every(mean => mean >= 3.5 && mean <= 4.5)).toBe(true)
  })

  it('explains continuing stochastic movement at the full-data optimum', () => {
    expect(batchGradient(4)).toBe(0)
    expect(batchGradient(4, [6])).toBe(-3)
    for (let n = 1; n <= 16; n++) {
      const rate = n * 0.05
      const next = 4 - rate * batchGradient(4, [6])
      const half = 4 - rate / 2 * batchGradient(4, [6])
      expect(next).toBeGreaterThan(4)
      expect(batchLoss(next)).toBeGreaterThan(batchLoss(4))
      expect(batchLoss(next, [6])).toBeLessThan(batchLoss(4, [6]))
      expect(half - 4).toBeCloseTo((next - 4) / 2, 12)
    }
  })

  it('lets students move the prediction line and return to the fixed scored situation', () => {
    render(<BatchGradientVisual dataset={batchGradientVisualDatasets.batchDirections} />)
    const control = screen.getByRole('slider', { name: 'Current constant prediction' })
    expect(control).toHaveValue('3')
    fireEvent.change(control, { target: { value: '6.25' } })
    expect(screen.getByRole('img')).toHaveAttribute('aria-label', expect.stringContaining('w = 6.25'))
    fireEvent.click(screen.getByRole('button', { name: 'Return to w = 3 for the questions' }))
    expect(control).toHaveValue('3')
  })

  it('replays the requested noisy step and recomputes traces when batch size changes', () => {
    render(<BatchGradientVisual dataset={batchGradientVisualDatasets.batchExplorer} />)
    expect(screen.getByLabelText('Selected batch rows')).toHaveTextContent('R2 (target 1)')
    expect(screen.getByLabelText('Parameter update')).toHaveTextContent('4.1280 → 2.8768')
    expect(screen.getByText('1.7612')).toBeInTheDocument()
    expect(screen.getByText('4.3808')).toBeInTheDocument()
    fireEvent.change(screen.getByRole('combobox', { name: 'Batch size' }), { target: { value: '8' } })
    expect(screen.getByRole('slider', { name: 'Inspect update' })).toHaveValue('4')
    expect(screen.getByRole('slider', { name: 'Inspect update' })).toHaveAttribute('max', '4')
    expect(screen.getByLabelText('Selected batch rows')).toHaveTextContent('R8 (target 8)')
    fireEvent.change(screen.getByRole('slider', { name: 'Inspect update' }), { target: { value: '1' } })
    expect(screen.getByLabelText('Parameter update')).toHaveTextContent('0.0000 → 1.6000')
    fireEvent.click(screen.getByRole('button', { name: 'Show batch size 1, update 5' }))
    expect(screen.getByRole('combobox', { name: 'Batch size' })).toHaveValue('1')
    expect(screen.getByRole('slider', { name: 'Inspect update' })).toHaveValue('5')
    expect(screen.getByLabelText('Parameter update')).toHaveTextContent('4.1280 → 2.8768')
  })

  it('restarts the one-row update at w=4 whenever the learning rate changes', () => {
    render(<BatchGradientVisual dataset={batchGradientVisualDatasets.batchNearMinimum} />)
    expect(screen.getByLabelText('Single-row updated prediction')).toHaveTextContent('4 → 5.20')
    fireEvent.change(screen.getByRole('slider', { name: 'Single-row learning rate' }), { target: { value: '0.2' } })
    expect(screen.getByLabelText('Single-row updated prediction')).toHaveTextContent('4 → 4.60')
    fireEvent.change(screen.getByRole('slider', { name: 'Single-row learning rate' }), { target: { value: String(batchLearningRate) } })
    expect(screen.getByLabelText('Single-row updated prediction')).toHaveTextContent('4 → 5.20')
  })

  it('renders the budget diagram and both concrete shuffle schedules', () => {
    const view = render(<BatchGradientVisual dataset={batchGradientVisualDatasets.batchBudget} />)
    expect(screen.getByRole('img')).toHaveAttribute('aria-label', expect.stringContaining('120, 6, and 1 updates'))
    view.unmount()
    render(<BatchGradientVisual dataset={batchGradientVisualDatasets.batchShuffle} />)
    expect(screen.getByRole('heading', { name: 'Schedule A' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Schedule B' })).toBeInTheDocument()
    expect(screen.getAllByText('R7: target 7')).toHaveLength(2)
  })

  it('records raw choices independently of exploratory controls and permits saved-answer redo', () => {
    const question = batchGradientDescentAssignment.questions[0]
    if (question.kind !== 'multipleChoice') throw new Error('Expected conceptual multiple choice')
    const state: QuestionState = { status: 'active', attempts: 0, incorrectAttempts: 0, hintsShown: 0, resolvedAt: null, latestAnswer: null, attemptHistory: [] }
    const onAttempt = vi.fn()
    const props = { question, state, questionNumber: 1, totalQuestions: 6, hints: [], onAttempt }
    const view = render(<MultipleChoiceQuestion {...props} />)
    expect(screen.getByRole('button', { name: 'Check answer' })).toBeDisabled()
    for (const part of question.parts) {
      const answer = part.options.find(option => option.id === part.correctOptionId)!
      fireEvent.click(screen.getByRole('radio', { name: answer.title }))
    }
    fireEvent.change(screen.getByRole('slider', { name: 'Current constant prediction' }), { target: { value: '7' } })
    fireEvent.click(screen.getByRole('button', { name: 'Check answer' }))
    const latestAnswer = { selectedIds: { directions: 'batch-left-full-right', average: 'mean' } }
    expect(onAttempt).toHaveBeenLastCalledWith('correct', latestAnswer)
    view.unmount()
    render(<MultipleChoiceQuestion {...props} state={{ ...state, status: 'correct', latestAnswer }} />)
    expect(screen.getByRole('radio', { name: 'Left for the first-four-row batch; right for the full dataset.' })).toBeChecked()
    expect(screen.getByRole('button', { name: 'Check answer' })).toBeEnabled()
    fireEvent.click(screen.getByRole('radio', { name: 'Right for both the first-four-row batch and the full dataset.' }))
    fireEvent.click(screen.getByRole('button', { name: 'Check answer' }))
    expect(onAttempt).toHaveBeenLastCalledWith('incorrect', { selectedIds: { directions: 'both-right', average: 'mean' } })
  })
})
