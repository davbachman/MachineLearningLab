import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { TransformerInterpretabilityVisual } from '../components/questions/TransformerInterpretabilityVisual'
import { transformerInterpretabilityAssignments, transformerInterpretabilityVisualDatasets } from '../data/transformerInterpretabilityAssignments'
import { getCorrectCodeLabSubmission, validateCodeLabStage } from '../lib/codeLab'
import { activationMetrics, addScaled, attentionPattern, attentionValues, normalizeChannels, samplingProbabilities, sparseMetrics, steerResidual } from '../lib/transformerInterpretability'

afterEach(cleanup)
const show = (mode: string) => render(<TransformerInterpretabilityVisual dataset={transformerInterpretabilityVisualDatasets[`ti-${mode}`]} />)
const change = (label: string, value: number) => fireEvent.change(screen.getByLabelText(label), { target: { value: String(value) } })

describe('independent transformer and SAE calculations', () => {
  it('preserves the residual when both updates vanish and composes nonzero updates', () => {
    expect(addScaled(addScaled([2, 1], [-1, 2], 1), [.5, -.5], 1)).toEqual([1.5, 2.5])
    expect(addScaled([2, 1], [-1, 2], 0)).toEqual([2, 1])
  })
  it('normalizes channels with population variance and is invariant to a common offset', () => {
    const values = normalizeChannels([1, 2, 3])
    expect(values.reduce((a, b) => a + b, 0)).toBeCloseTo(0)
    expect(values.reduce((a, b) => a + b * b, 0) / 3).toBeCloseTo(1)
    expect(normalizeChannels([11, 12, 13])).toEqual(values)
    expect(normalizeChannels([2, 2, 2])).toEqual([0, 0, 0])
  })
  it('uses exactly the threshold-based top-k rule, including ties', () => {
    expect(samplingProbabilities([2, 1, 1, 0], 1, 2).filter(x => x > 0)).toHaveLength(3)
    expect(samplingProbabilities([2, 1, 0, -1], 1, 2)[0]).toBeCloseTo(Math.E / (Math.E + 1))
    expect(samplingProbabilities([2, 1, 0, -1], .5, 2)[0]).toBeGreaterThan(samplingProbabilities([2, 1, 0, -1], 1, 2)[0])
    expect(samplingProbabilities([2, 1, 0, -1], 2, 1)).toEqual([1, 0, 0, 0])
  })
  it('has normalized causal attention rows and the independently computed weighted value', () => {
    attentionPattern.forEach((row, q) => {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1)
      expect(row.slice(q + 1).every(x => x === 0)).toBe(true)
    })
    expect(attentionPattern[2].reduce((sum, w, i) => sum + w * attentionValues[i], 0)).toBeCloseTo(2.1)
  })
  it('computes the sparse reconstruction frontier from the same underlying vectors', () => {
    expect(sparseMetrics([0, 0, 2])).toEqual({ reconstruction: [1.2, 1.6], mse: 0, l1: 2, l0: 1 })
    const exact = sparseMetrics([1.2, 1.6, 0]), shrunk = sparseMetrics([0, 0, 1.5])
    expect(exact.mse).toBe(0)
    expect(shrunk.mse).toBeCloseTo(.125)
    expect(exact.mse + .2 * exact.l1).toBeCloseTo(.56)
    expect(shrunk.mse + .2 * shrunk.l1).toBeCloseTo(.425)
  })
  it('counts threshold equality as inactive and distinguishes a dead column from sparse rows', () => {
    const metrics = activationMetrics([[0, 1e-4, 2], [0, 0, 0], [0, 2e-4, 1]])
    expect(metrics.meanL0).toBe(1)
    expect(metrics.deadFraction).toBeCloseTo(1 / 3)
    expect(activationMetrics([[1, 0, .6], [0, 2, 1.6]]).meanL1).toBeCloseTo(2.6)
  })
  it('ablates and amplifies only the selected contribution', () => {
    const ablated = steerResidual([3, 2], 2, [.6, .8], 0)
    expect(ablated[0]).toBeCloseTo(1.8)
    expect(ablated[1]).toBeCloseTo(.4)
    expect(steerResidual([3, 2], 2, [.6, .8], 1)).toEqual([3, 2])
    const amplified = steerResidual([3, 2], 2, [.6, .8], 3)
    expect(amplified[0]).toBeCloseTo(5.4)
    expect(amplified[1]).toBeCloseTo(5.2)
  })
})

describe('meaningful visual controls', () => {
  it('updates both residual branches', () => { show('residual'); change('Attention update scale', 0); change('MLP update scale', 0); expect(screen.getByLabelText('Residual result')).toHaveTextContent('(2.00, 1.00)') })
  it('does not let a later token change the earlier normalized vector', () => { show('normalization'); const original = screen.getByLabelText('Normalized token 0').textContent; change('Second token, first channel', 16); expect(screen.getByLabelText('Normalized token 0').textContent).toBe(original); expect(screen.getByLabelText('Normalized token 1')).toHaveTextContent('(1.22, -1.22, 0.00)') })
  it('slides targets with the input window', () => { show('windows'); expect(screen.getByLabelText('Window targets')).toHaveTextContent('D, E, F'); change('Window start position', 0); expect(screen.getByLabelText('Window targets')).toHaveTextContent('B, C, D') })
  it('changes sampling support without editing logits', () => { show('sampling'); change('Top-k cutoff', 1); expect(screen.getByLabelText('Sampling probabilities')).toHaveTextContent('(1.00, 0.00, 0.00, 0.00)') })
  it('recomputes the selected query row', () => { show('attention'); expect(screen.getByLabelText('Attention weighted value')).toHaveTextContent('2.10'); change('Query position', 0); expect(screen.getByLabelText('Attention weighted value')).toHaveTextContent('2.00') })
  it('rotates the projection while retaining all labeled observations', () => { show('direction'); change('Projection angle in degrees', 90); expect(screen.getByText('Unit probe direction: (0.00, 1.00)')).toBeInTheDocument(); expect(screen.getAllByRole('row')).toHaveLength(5) })
  it('shows the zero-code reconstruction error and active count', () => { show('dictionary'); change('Feature 3 coefficient', 0); expect(screen.getByLabelText('Sparse reconstruction metrics')).toHaveTextContent('MSE 2.000; L1 0.00; active features 0') })
  it('recomputes the fixed candidate objective, not a refit', () => { show('sparsity'); expect(screen.getByText('0.425')).toBeInTheDocument(); change('L1 penalty strength', 0); expect(screen.queryByText('0.425')).not.toBeInTheDocument(); expect(screen.getAllByText('0.125')).toHaveLength(2) })
  it('lets ablation expose the residual remainder', () => { show('steering'); change('Feature contribution multiplier', 0); expect(screen.getByLabelText('Steered residual')).toHaveTextContent('(1.80, 0.40)') })
  it('reports quality cost relative to the same baseline', () => { show('quality'); fireEvent.change(screen.getByLabelText('Steering condition'), { target: { value: '2' } }); expect(screen.getByLabelText('Steering quality cost')).toHaveTextContent('increase above baseline 0.90') })
})

describe('five independent assignments and raw grading', () => {
  it('publishes the correct current notebook numbers and complete stage structure', () => {
    expect(transformerInterpretabilityAssignments.map(a => a.displayNumber)).toEqual([21, 22, 23, 24, 25])
    for (const assignment of transformerInterpretabilityAssignments) {
      expect(assignment.version).toBe(1)
      expect(assignment.published).toBe(true)
      expect(assignment.questions).toHaveLength(4)
      const lab = assignment.questions.at(-1)!
      expect(lab.kind).toBe('codeLab')
      if (lab.kind !== 'codeLab') continue
      expect(lab.stages.length).toBeGreaterThanOrEqual(5)
      expect(lab.stages.length).toBeLessThanOrEqual(6)
      expect(lab.stages.every(s => s.fields.length === 1)).toBe(true)
      const correct = getCorrectCodeLabSubmission(lab)
      expect(lab.validator(correct).correct).toBe(true)
      const first = lab.stages[0]
      delete correct.stages[first.id]
      expect(lab.validator(correct).correct).toBe(false)
      expect(validateCodeLabStage(lab.stages[1], correct.stages[lab.stages[1].id]).correct).toBe(true)
      expect(lab.validator(null).correct).toBe(false)
      expect(lab.validator({ completed: true, score: 100 }).correct).toBe(false)
    }
  })
  it('validates choices using their raw stable IDs', () => {
    for (const assignment of transformerInterpretabilityAssignments) for (const q of assignment.questions) {
      if (q.kind !== 'multipleChoice') continue
      expect(transformerInterpretabilityVisualDatasets[q.datasetId]).toBeDefined()
      const selectedIds = Object.fromEntries(q.parts.map(p => [p.id, p.correctOptionId]))
      expect(q.validator({ selectedIds }).correct).toBe(true)
      expect(q.validator({ selectedIds: {} }).correct).toBe(false)
      expect(q.validator({ selectedIds: [] }).correct).toBe(false)
      for (const p of q.parts) {
        const incorrect = p.options.find(o => o.id !== p.correctOptionId)!.id
        expect(q.validator({ selectedIds: { ...selectedIds, [p.id]: incorrect } }).correct).toBe(false)
      }
    }
  })
})
