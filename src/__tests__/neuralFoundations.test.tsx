import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { NeuralFoundationsVisual } from '../components/questions/NeuralFoundationsVisual'
import { neuralFoundationsAssignments, neuralFoundationsVisualDatasets } from '../data/neuralFoundationsAssignments'
import { neuralFoundationsNotebookLabs } from '../data/neuralFoundationsCodeLabs'
import { getCorrectCodeLabSubmission, validateCodeLabStage } from '../lib/codeLab'
import { affineRows, allPairsMse, batchGroups, contrastImage, contrastKernel, denseParameterCount, dropoutValues, exponentialMoments, hingeValue, logRoot, logRootDerivative, maxPool, pairedMse, poolingImage, sharedBranchTrace, squaredErrorGrid, twoBranchValue, validCorrelation } from '../lib/neuralFoundations'

afterEach(cleanup)

describe('neural foundations assignment structure and answer contracts', () => {
  it('publishes five numbered modules with self-contained warmups and a final independently scored lab', () => {
    expect(neuralFoundationsAssignments.map(a => a.displayNumber)).toEqual([11, 12, 13, 14, 15])
    expect(neuralFoundationsAssignments.map(a => a.id)).toEqual(['mlps', 'autograd', 'pytorch', 'optimizers', 'convolutions'])
    for (const assignment of neuralFoundationsAssignments) {
      expect(assignment.version).toBe(1)
      expect(assignment.published).toBe(true)
      expect(assignment.questions).toHaveLength(4)
      const warmups = assignment.questions.slice(0, -1)
      expect(new Set(warmups.map(q => q.datasetId)).size).toBeGreaterThanOrEqual(2)
      for (const question of warmups) {
        expect(question.kind).toBe('multipleChoice')
        expect(JSON.stringify(question)).not.toMatch(/notebook|ipynb|numpy|self\.|np\.|torch\./i)
        expect(neuralFoundationsVisualDatasets[question.datasetId!]).toBeDefined()
        if (question.kind !== 'multipleChoice') continue
        expect(question.validator({ selectedIds: Object.fromEntries(question.parts.map(part => [part.id, part.correctOptionId])) }).correct).toBe(true)
        expect(question.validator({}).correct).toBe(false)
        expect(question.validator(null).correct).toBe(false)
        for (const part of question.parts) {
          expect(part.options.filter(option => option.id === part.correctOptionId)).toHaveLength(1)
          expect(new Set(part.options.map(option => option.title)).size).toBe(part.options.length)
        }
      }
      const final = assignment.questions.at(-1)!
      expect(final.kind).toBe('codeLab')
      if (final.kind !== 'codeLab') continue
      expect(final.stages.length).toBeGreaterThanOrEqual(4)
      expect(final.stages.length).toBeLessThanOrEqual(6)
      expect(final.stages.every(stage => stage.fields.length === 1)).toBe(true)
      expect(final.prompt).toContain(`Homework2026/${assignment.displayNumber}`)
    }
  })

  it('validates labs from raw choices and isolates incomplete, incorrect, and malformed stage answers', () => {
    for (const lab of Object.values(neuralFoundationsNotebookLabs)) {
      const correct = getCorrectCodeLabSubmission(lab)
      expect(lab.validator(correct).correct).toBe(true)
      expect(lab.validator(null).correct).toBe(false)
      expect(lab.validator({ ...correct, variantId: 'other' }).correct).toBe(false)
      const first = lab.stages[0]
      const wrong = { ...correct, stages: { ...correct.stages, [first.id]: { answer: 'unrecognized' } } }
      expect(lab.validator(wrong).correct).toBe(false)
      expect(validateCodeLabStage(first, wrong.stages[first.id]).incorrectFieldIds).toEqual(['answer'])
      expect(validateCodeLabStage(first, {}).missingFieldIds).toEqual(['answer'])
      for (const next of lab.stages.slice(1)) expect(validateCodeLabStage(next, wrong.stages[next.id]).correct).toBe(true)
    }
  })

  it('labels the flat-target counterexample honestly while preserving the PyTorch grading contract', () => {
    const lab = neuralFoundationsNotebookLabs.pytorch
    expect(lab.instructions).toContain('Unlike the corrected notebook')
    expect(lab.instructions).toContain('intentional counterexample')
    expect(lab.instructions).not.toMatch(/undefined mpg|Most network examples.*commented out/)
    expect(lab.code).toContain('network=Sequential(\n    Linear(2,3),\n    ReLU(),\n    Linear(3,1)\n)')
    expect(lab.code).toContain('mpg=torch.tensor(cars.mpg.to_numpy(),dtype=torch.float32).reshape(-1,1)')
    expect(lab.code).toContain('Corrected notebook excerpts; dataset/model setup is omitted here:')
    // The corrected source excerpt and the deliberately incorrect fixed fixture
    // must stay distinct; no old submission is reinterpreted under a new key.
    expect(lab.fixture).toContain('mpg = torch.tensor([2., 0.])')
    expect(lab.invocation).toContain('error_shape = tuple((prediction - mpg).shape)')
    expect(lab.invocation).toContain('broadcast_mse = torch.nn.MSELoss()(prediction, mpg).item()')
    expect(lab.variantId).toBe('pytorch-notebook-lab-v1')
    expect(lab.stages.map(stage => [stage.id, stage.fields[0].correctOptionId])).toEqual([
      ['weight', 'three-two'], ['count', 'thirteen'], ['output', 'fifteen-one'],
      ['broadcast', 'two-two'], ['mse', 'three'], ['classes', 'zero-one'],
    ])
    expect(lab.reveal.explanation).toContain('corrected notebook keeps the cars target column-shaped')
  })
})

describe('independent mechanism and fixture calculations', () => {
  it('rectifies before composing the next dense layer and counts shared parameters', () => {
    expect([hingeValue(-1, 1), hingeValue(2, 1)]).toEqual([0, 1])
    expect([-2, 2].map(x => twoBranchValue(x, true))).toEqual([2, 2])
    expect([-2, 2].map(x => twoBranchValue(x, false))).toEqual([0, 0])
    expect(denseParameterCount([2, 2, 1])).toBe(9)
    expect(denseParameterCount([2, 3, 1])).toBe(13)
    const preactivation = affineRows([[1, -1], [0, 2]], [[1, -2, 0], [2, 1, -1]], [0, 1, 1])
    expect(preactivation).toEqual([[-1, -2, 2], [4, 3, -1]])
    const relu = preactivation.map(row => row.map(x => Math.max(0, x)))
    expect(affineRows(relu, [[1], [-1], [2]], [.5])).toEqual([[4.5], [1.5]])
    expect(affineRows(preactivation, [[1], [-1], [2]], [.5])[0][0]).toBe(5.5)
  })

  it('verifies shared-path and composed derivatives by finite differences', () => {
    const h = 1e-5
    const trace = sharedBranchTrace(2, 3)
    expect(trace).toEqual({ product: 6, value: 8, productPath: 3, directPath: 1, da: 4, db: 2 })
    expect((sharedBranchTrace(2 + h, 3).value - sharedBranchTrace(2 - h, 3).value) / (2 * h)).toBeCloseTo(trace.da, 8)
    expect((sharedBranchTrace(2, 3 + h).value - sharedBranchTrace(2, 3 - h).value) / (2 * h)).toBeCloseTo(trace.db, 8)
    for (const u of [1, 2, 4, 7]) expect((logRoot(u + h) - logRoot(u - h)) / (2 * h)).toBeCloseTo(logRootDerivative(u), 8)
    expect(logRootDerivative(4)).toBe(.125)
  })

  it('distinguishes pairwise loss from an unintended all-pairs mean', () => {
    expect(squaredErrorGrid([1, 3], [2, 0])).toEqual([[1, 1], [1, 9]])
    expect(pairedMse([1, 3], [2, 0])).toBe(5)
    expect(allPairsMse([1, 3], [2, 0])).toBe(3)
  })

  it('replays moment memory, complete permutations, final short batches, and dropout modes', () => {
    expect(exponentialMoments([2, -2, 2, -2], .5)).toEqual([1, -.5, .75, -.625])
    expect(exponentialMoments([2, -2, 2, -2], 0)).toEqual([2, -2, 2, -2])
    const order = [8, 2, 11, 0, 5, 7, 1, 10, 4, 9, 3, 6]
    const groups = batchGroups(order, 5)
    expect(groups.map(group => group.length)).toEqual([5, 5, 2])
    expect(groups.flat()).toEqual(order)
    expect(new Set(groups.flat()).size).toBe(12)
    expect(batchGroups(Array.from({ length: 1437 }, (_, i) => i), 100).map(group => group.length)).toEqual([...Array(14).fill(100), 37])
    expect(dropoutValues([1, 2, 3, 4], [true, false, true, true], .25, true)).toEqual([4 / 3, 0, 4, 16 / 3])
    expect(dropoutValues([1, 2, 3, 4], [true, false, true, true], .25, false)).toEqual([1, 2, 3, 4])
  })

  it('computes local contrast with shared weights and cancels uniform brightness', () => {
    const response = validCorrelation(contrastImage, contrastKernel)
    expect(response).toEqual([[2, -5, 0], [3, 0, -5], [1, 5, -3]])
    expect(validCorrelation(contrastImage.map(row => row.map(x => x + 5)), contrastKernel)).toEqual(response)
    expect(maxPool(poolingImage, [2, 2])).toEqual([[5, 6]])
    const rearranged = poolingImage.map(row => [...row])
    rearranged[0][0] = 5
    rearranged[0][1] = 1
    expect(maxPool(rearranged, [2, 2])).toEqual([[5, 6]])
  })

  it('traces all convolution lab windows, kernel orientation, and discarded edges', () => {
    const image = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    const response = validCorrelation(image, [[1, 2], [0, -1]])
    expect(response).toEqual([[-2, 0, 2, 4], [8, 10, 12, 14]])
    expect(maxPool(image, [2, 2])).toEqual([[7, 9]])
    expect(maxPool(response, [2, 2])).toEqual([[10, 14]])
    expect(validCorrelation(image, [[-1, 0], [2, 1]])[0][0]).toBe(18)
    expect(maxPool([...image.slice(0, 2), [1000, 1000, 1000, 1000, 1000]], [2, 2])).toEqual([[7, 9]])
  })
})

describe('interactive neural mechanisms preserve fixed scored references', () => {
  it('moves a hinge and restores its reference', () => {
    render(<NeuralFoundationsVisual dataset={neuralFoundationsVisualDatasets.neuralHinge} />)
    expect(screen.getByLabelText('Hinge probe result')).toHaveTextContent('output: 1')
    fireEvent.change(screen.getByRole('slider', { name: 'Activation threshold' }), { target: { value: '-2' } })
    expect(screen.getByLabelText('Hinge probe result')).toHaveTextContent('output: 4')
    expect(screen.getByText(/Scored reference: threshold 1/)).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Reset hinge' }))
    expect(screen.getByLabelText('Hinge probe result')).toHaveTextContent('output: 1')
  })

  it('toggles hidden nonlinearities without changing branch weights', () => {
    render(<NeuralFoundationsVisual dataset={neuralFoundationsVisualDatasets.neuralBranches} />)
    expect(screen.getByLabelText('Two-branch output')).toHaveTextContent('Sum: 2')
    fireEvent.click(screen.getByRole('checkbox'))
    expect(screen.getByLabelText('Two-branch output')).toHaveTextContent('Sum: 0')
  })

  it('changes batch dimensions while retaining parameter sharing', () => {
    render(<NeuralFoundationsVisual dataset={neuralFoundationsVisualDatasets.neuralShapes} />)
    fireEvent.change(screen.getByRole('slider', { name: 'Observation count' }), { target: { value: '6' } })
    expect(screen.getByLabelText('Shared network dimensions')).toHaveTextContent('6 rows')
    expect(screen.getByLabelText('Shared network dimensions')).toHaveTextContent('parameters: 9')
  })

  it('updates shared-path gradient contributions', () => {
    render(<NeuralFoundationsVisual dataset={neuralFoundationsVisualDatasets.neuralGraph} />)
    fireEvent.change(screen.getByRole('slider', { name: 'Graph input b' }), { target: { value: '4' } })
    expect(screen.getByLabelText('Graph gradient contributions')).toHaveTextContent('product path 4 + direct path 1 = 5')
    fireEvent.click(screen.getByRole('button', { name: 'Reset graph reference' }))
    expect(screen.getByLabelText('Graph gradient contributions')).toHaveTextContent('product path 3 + direct path 1 = 4')
  })

  it('recomputes chain derivatives at the movable probe', () => {
    render(<NeuralFoundationsVisual dataset={neuralFoundationsVisualDatasets.neuralChain} />)
    expect(screen.getByLabelText('Chain local derivatives')).toHaveTextContent('total: 0.125')
    fireEvent.change(screen.getByRole('slider', { name: 'Chain-rule input u' }), { target: { value: '2' } })
    expect(screen.getByLabelText('Chain local derivatives')).toHaveTextContent('total: 0.25')
  })

  it('highlights paired versus all-pairs loss terms', () => {
    render(<NeuralFoundationsVisual dataset={neuralFoundationsVisualDatasets.neuralPairedLoss} />)
    expect(screen.getByLabelText('Comparison loss')).toHaveTextContent('Paired mean squared error: 5')
    fireEvent.click(screen.getByRole('button', { name: 'All possible pairs' }))
    expect(screen.getByLabelText('Comparison loss')).toHaveTextContent('All-pairs mean squared error: 3')
    fireEvent.click(screen.getByRole('button', { name: 'Matching observations' }))
    expect(screen.getByLabelText('Comparison loss')).toHaveTextContent('Paired mean squared error: 5')
  })

  it('changes score confidence without changing its class ordering', () => {
    render(<NeuralFoundationsVisual dataset={neuralFoundationsVisualDatasets.neuralLogits} />)
    const initial = screen.getByLabelText('Score probability comparison').textContent
    fireEvent.change(screen.getByRole('slider', { name: 'Class score multiplier' }), { target: { value: '2' } })
    expect(screen.getByLabelText('Score probability comparison').textContent).not.toBe(initial)
    expect(screen.getByLabelText('Score probability comparison')).toHaveTextContent('Predicted class: A')
  })

  it('replays moment summaries from zero when beta changes', () => {
    render(<NeuralFoundationsVisual dataset={neuralFoundationsVisualDatasets.neuralMoments} />)
    expect(screen.getByLabelText('Running gradient moments')).toHaveTextContent('(1, -0.5, 0.75, -0.625)')
    fireEvent.change(screen.getByRole('slider', { name: 'Gradient memory beta' }), { target: { value: '0' } })
    expect(screen.getByLabelText('Running gradient moments')).toHaveTextContent('(2, -2, 2, -2)')
  })

  it('preserves observation coverage when shuffling and resizing batches', () => {
    render(<NeuralFoundationsVisual dataset={neuralFoundationsVisualDatasets.neuralBatches} />)
    expect(screen.getByLabelText('Batch size summary')).toHaveTextContent('(5, 5, 2)')
    fireEvent.click(screen.getByRole('button', { name: 'Use a shuffled order' }))
    expect(screen.getByLabelText('Batch size summary')).toHaveTextContent('distinct IDs: 12')
    fireEvent.change(screen.getByRole('slider', { name: 'Mini-batch size' }), { target: { value: '4' } })
    expect(screen.getByLabelText('Batch size summary')).toHaveTextContent('(4, 4, 4)')
  })

  it('switches dropout mode and restores its fixed training mask', () => {
    render(<NeuralFoundationsVisual dataset={neuralFoundationsVisualDatasets.neuralDropout} />)
    expect(screen.getByLabelText('Dropout output')).toHaveTextContent('(1.333, 0, 4, 5.333)')
    fireEvent.click(screen.getByRole('button', { name: 'Evaluation mode' }))
    expect(screen.getByLabelText('Dropout output')).toHaveTextContent('(1, 2, 3, 4)')
    fireEvent.click(screen.getByRole('button', { name: 'Training mask' }))
    expect(screen.getByLabelText('Dropout output')).toHaveTextContent('(1.333, 0, 4, 5.333)')
  })

  it('selects a filter patch with both grid buttons and range controls', () => {
    render(<NeuralFoundationsVisual dataset={neuralFoundationsVisualDatasets.neuralCorrelation} />)
    expect(screen.getByLabelText('Selected filter calculation')).toHaveTextContent('= 2')
    fireEvent.click(screen.getByRole('button', { name: 'Inspect patch row 0 column 1' }))
    expect(screen.getByLabelText('Selected filter calculation')).toHaveTextContent('= -5')
    fireEvent.change(screen.getByRole('slider', { name: 'Patch start row' }), { target: { value: '2' } })
    expect(screen.getByLabelText('Selected filter calculation')).toHaveTextContent('= 5')
  })

  it('demonstrates pooling invariance to a within-block permutation', () => {
    render(<NeuralFoundationsVisual dataset={neuralFoundationsVisualDatasets.neuralPooling} />)
    const initial = screen.getByLabelText('Pooled output values').textContent
    fireEvent.click(screen.getByRole('button', { name: 'Move peak within first block' }))
    expect(screen.getByLabelText('Pooled output values').textContent).toBe(initial)
    expect(screen.getByRole('button', { name: 'Restore peak position' })).toBeVisible()
  })
})
