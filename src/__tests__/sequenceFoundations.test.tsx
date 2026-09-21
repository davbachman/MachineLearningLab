import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { spawnSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { SequenceFoundationsVisual } from '../components/questions/SequenceFoundationsVisual'
import { sequenceFoundationsAssignments, sequenceFoundationsVisualDatasets } from '../data/sequenceFoundationsAssignments'
import { sequenceNotebookLabs } from '../data/sequenceFoundationsCodeLabs'
import { getCorrectCodeLabSubmission, validateCodeLabStage } from '../lib/codeLab'
import { attentionMatrix, causalRow, cleanPixels, convSize, edgeImage, edgeKernel, filterAt, headShapes, mse, noisyPixels, normalizedWeights, pairCompression, poolTwo, poolingImage, restoredPixels, retrieve, shiftedWindow, simpleTokens, transposeConvSize } from '../lib/sequenceFoundations'

afterEach(cleanup)

describe('sequence foundations modules', () => {
  it('provides five complete published modules with conceptual warmups and six independent lab stages', () => {
    expect(sequenceFoundationsAssignments.map(a => a.displayNumber)).toEqual([16, 17, 18, 19, 20])
    const ids = new Set<string>()
    for (const assignment of sequenceFoundationsAssignments) {
      expect(assignment.version).toBe(1)
      expect(assignment.published).toBe(true)
      expect(assignment.questions).toHaveLength(4)
      for (const question of assignment.questions.slice(0, -1)) {
        expect(JSON.stringify(question)).not.toMatch(/notebook|ipynb|torch|nn\./i)
        expect(question.kind).toBe('multipleChoice')
        expect(sequenceFoundationsVisualDatasets[question.datasetId!]).toBeDefined()
        expect(ids.has(question.id)).toBe(false)
        ids.add(question.id)
        if (question.kind !== 'multipleChoice') continue
        const selectedIds = Object.fromEntries(question.parts.map(p => [p.id, p.correctOptionId]))
        expect(question.validator({ selectedIds }).correct).toBe(true)
        expect(question.validator({ selectedIds: {} }).correct).toBe(false)
        expect(question.validator(null).correct).toBe(false)
        for (const part of question.parts) {
          expect(new Set(part.options.map(o => o.id)).size).toBe(part.options.length)
          expect(part.options.filter(o => o.id === part.correctOptionId)).toHaveLength(1)
          for (const option of part.options.filter(o => o.id !== part.correctOptionId)) {
            expect(question.validator({ selectedIds: { ...selectedIds, [part.id]: option.id } }).correct).toBe(false)
          }
        }
      }
      const lab = sequenceNotebookLabs[assignment.id]
      expect(assignment.questions.at(-1)).toBe(lab)
      expect(lab.stages).toHaveLength(6)
      expect(lab.validator(getCorrectCodeLabSubmission(lab)).correct).toBe(true)
      for (const stage of lab.stages) {
        expect(stage.fields).toHaveLength(1)
        expect(validateCodeLabStage(stage, { answer: stage.fields[0].correctOptionId }).correct).toBe(true)
        expect(validateCodeLabStage(stage, {}).correct).toBe(false)
        expect(validateCodeLabStage(stage, { answer: 'forged' }).correct).toBe(false)
      }
      expect(lab.validator({ ...getCorrectCodeLabSubmission(lab), variantId: 'other' }).correct).toBe(false)
    }
  })

  it('independently computes convolution, pooling, and both architectures’ spatial sizes', () => {
    expect([0, 1, 2].map(column => filterAt(edgeImage, edgeKernel, 0, column))).toEqual([0, 2, 0])
    expect(filterAt(edgeImage, edgeKernel, 2, 1)).toBe(2)
    expect(poolTwo(poolingImage, 'max')).toEqual([[4, 5], [3, 6]])
    expect(poolTwo(poolingImage, 'mean')).toEqual([[2.5, 2], [1.25, 2.5]])
    expect(convSize(37, 3, 1)).toBe(37)
    expect([50, 37].map(n => convSize(convSize(convSize(n, 2, 0, 2), 2, 0, 2), 2, 0, 2))).toEqual([6, 4])
    expect(128 * 6 * 4).toBe(3072)
    expect([12, 9].map(n => transposeConvSize(n, 3, 1, 2, 1))).toEqual([24, 18])
    expect([12, 9].map(n => transposeConvSize(n, 3, 1, 2, 0))).toEqual([23, 17])
  })

  it('verifies denoising losses and a genuine representation collision', () => {
    expect(mse(noisyPixels, cleanPixels)).toBeCloseTo(.16, 14)
    expect(mse(restoredPixels, cleanPixels)).toBeCloseTo(.015, 14)
    expect(mse(noisyPixels, noisyPixels)).toBe(0)
    expect(mse(restoredPixels, noisyPixels)).toBeCloseTo(.095, 14)
    const first = pairCompression([0, 1, .2, .8]), second = pairCompression([1, 0, .8, .2])
    expect(first).toEqual({ code: [.5, .5], reconstruction: [.5, .5, .5, .5] })
    expect(second).toEqual(first)
    expect(.1 * 0 + .9 * (.9 / .9)).toBeCloseTo(.9, 14)
  })

  it('verifies tokenizer units and every window start shown in the panel', () => {
    expect([...'blue kite!']).toHaveLength(10)
    expect(simpleTokens('blue kite!')).toEqual(['blue', 'kite', '!'])
    expect(simpleTokens("Ava's kite 12!")).toEqual(["Ava's", 'kite', '12', '!'])
    expect(shiftedWindow(0)).toEqual({ inputs: ['<bos>', 'Ava', 'found', 'a'], targets: ['Ava', 'found', 'a', 'blue'] })
    for (const start of [0, 1, 2, 3]) {
      const window = shiftedWindow(start)
      expect(window.inputs).toHaveLength(4)
      expect(window.targets).toHaveLength(4)
      expect(window.inputs.slice(1)).toEqual(window.targets.slice(0, -1))
    }
  })

  it('independently checks attention scores, value-only changes, mask zeros and head dimensions', () => {
    const original = retrieve([1, 0]), changed = retrieve([1, 0], [2, 8, 9])
    const exponential = Math.exp(1 / Math.sqrt(2)), denominator = 2 * exponential + 1
    expect(original.weights).toEqual([exponential / denominator, 1 / denominator, exponential / denominator])
    expect(changed.weights).toEqual(original.weights)
    expect(changed.result - original.result).toBeCloseTo(4 * original.weights[2], 14)
    expect(attentionMatrix[1][1]).toBeCloseTo(attentionMatrix[1][2], 14)
    expect(attentionMatrix[1][0]).toBeLessThan(attentionMatrix[1][1])
    for (const row of attentionMatrix) expect(row.reduce((s, x) => s + x, 0)).toBeCloseTo(1, 14)
    expect(causalRow(0)).toEqual([1, 0, 0, 0])
    for (const index of [0, 1, 2, 3]) {
      const row = causalRow(index)
      expect(row.slice(index + 1).every(x => x === 0)).toBe(true)
      expect(row.reduce((s, x) => s + x, 0)).toBeCloseTo(1, 14)
    }
    expect(causalRow(0, false).every(x => x > 0)).toBe(true)
    expect(normalizedWeights([1000, 1000])).toEqual([.5, .5])
    expect(() => normalizedWeights([1], [false])).toThrow()
    expect(headShapes(3, 6, 8, 2)).toEqual({ headSize: 4, split: [3, 2, 6, 4], weights: [3, 2, 6, 6], output: [3, 6, 8] })
    expect(() => headShapes(3, 6, 8, 3)).toThrow()
  })

  it('exercises both image interactions and denoising comparisons', () => {
    let view = render(<SequenceFoundationsVisual dataset={sequenceFoundationsVisualDatasets.cnnFilter} />)
    fireEvent.change(screen.getByRole('slider', { name: 'Patch left column' }), { target: { value: '1' } })
    expect(screen.getByLabelText('Filter response')).toHaveTextContent('= 2')
    view.unmount()
    view = render(<SequenceFoundationsVisual dataset={sequenceFoundationsVisualDatasets.cnnPool} />)
    fireEvent.change(screen.getByRole('combobox', { name: 'Pooling rule' }), { target: { value: 'mean' } })
    expect(screen.getByRole('img', { name: 'Mean output; stride 2' })).toBeInTheDocument()
    view.unmount()
    view = render(<SequenceFoundationsVisual dataset={sequenceFoundationsVisualDatasets.aeDenoise} />)
    expect(screen.getByLabelText('Reconstruction losses')).toHaveTextContent('A = 0.160; B = 0.015')
    fireEvent.change(screen.getByRole('combobox', { name: 'Reconstruction target' }), { target: { value: 'noisy' } })
    expect(screen.getByLabelText('Reconstruction losses')).toHaveTextContent('A = 0.000; B = 0.095')
    view.unmount()
    render(<SequenceFoundationsVisual dataset={sequenceFoundationsVisualDatasets.aeBottleneck} />)
    fireEvent.change(screen.getByRole('combobox', { name: 'Compression input' }), { target: { value: 'B' } })
    expect(screen.getByLabelText('Compressed representation')).toHaveTextContent('[0.5, 0.5]')
    expect(screen.getByRole('img', { name: 'Image B: 1, 0, 0.8, 0.2' })).toBeInTheDocument()
  })

  it('exercises tokenization, shifted windows, and position embeddings', () => {
    let view = render(<SequenceFoundationsVisual dataset={sequenceFoundationsVisualDatasets.tokensUnits} />)
    expect(screen.getByLabelText('Token count')).toHaveTextContent('3')
    fireEvent.change(screen.getByRole('combobox', { name: 'Tokenization units' }), { target: { value: 'characters' } })
    expect(screen.getByLabelText('Token count')).toHaveTextContent('10')
    view.unmount()
    view = render(<SequenceFoundationsVisual dataset={sequenceFoundationsVisualDatasets.tokensWindows} />)
    fireEvent.change(screen.getByRole('slider', { name: 'Window start index' }), { target: { value: '3' } })
    expect(screen.getAllByRole('row')).toHaveLength(5)
    expect(screen.getByRole('cell', { name: '<eos>' })).toBeInTheDocument()
    view.unmount()
    render(<SequenceFoundationsVisual dataset={sequenceFoundationsVisualDatasets.tokensEmbedding} />)
    fireEvent.change(screen.getByRole('slider', { name: 'Token position' }), { target: { value: '1' } })
    expect(screen.getByRole('slider', { name: 'Token position' })).toHaveValue('1')
    expect(screen.getByRole('img', { name: /Rows: token vector/ })).toHaveTextContent('3')
  })

  it('exercises retrieval, row inspection, causal masks and multi-head controls', () => {
    let view = render(<SequenceFoundationsVisual dataset={sequenceFoundationsVisualDatasets.attentionRetrieve} />)
    const initialBars = screen.getByRole('img').getAttribute('aria-label')
    fireEvent.change(screen.getByRole('slider', { name: 'Value at key 2' }), { target: { value: '9' } })
    expect(screen.getByRole('img').getAttribute('aria-label')).toBe(initialBars)
    expect(screen.getByLabelText('Retrieved value')).toHaveTextContent(retrieve([1, 0], [2, 8, 9]).result.toFixed(3))
    fireEvent.change(screen.getByRole('slider', { name: 'Query first coordinate' }), { target: { value: '0' } })
    expect(screen.getByRole('img').getAttribute('aria-label')).not.toBe(initialBars)
    view.unmount()
    view = render(<SequenceFoundationsVisual dataset={sequenceFoundationsVisualDatasets.attentionMatrix} />)
    fireEvent.change(screen.getByRole('slider', { name: 'Inspect query row' }), { target: { value: '1' } })
    expect(screen.getByLabelText('Selected attention output')).toHaveTextContent('query 1')
    view.unmount()
    view = render(<SequenceFoundationsVisual dataset={sequenceFoundationsVisualDatasets.causalMask} />)
    fireEvent.change(screen.getByRole('slider', { name: 'Query position' }), { target: { value: '0' } })
    expect(screen.getByRole('img', { name: 'Key probabilities: 1.000, 0.000, 0.000, 0.000' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('checkbox', { name: 'Apply causal mask' }))
    expect(screen.queryByRole('img', { name: 'Key probabilities: 1.000, 0.000, 0.000, 0.000' })).not.toBeInTheDocument()
    view.unmount()
    render(<SequenceFoundationsVisual dataset={sequenceFoundationsVisualDatasets.causalHeads} />)
    fireEvent.change(screen.getByRole('combobox', { name: 'Attention head count' }), { target: { value: '4' } })
    expect(screen.getByLabelText('Head shapes')).toHaveTextContent('(3, 4, 6, 2)')
    expect(screen.getByLabelText('Head shapes')).toHaveTextContent('(3, 6, 8)')
  })

  it('quotes the current CNN, autoencoder, and custom-module implementations without silently repairing them', () => {
    const source = (name: string) => {
      const notebook = JSON.parse(readFileSync(`Homework2026/${name}.ipynb`, 'utf8')) as { cells: { cell_type: string; source: string[] }[] }
      return notebook.cells.filter(cell => cell.cell_type === 'code').map(cell => cell.source.join('')).join('\n')
    }
    expect(source('16CNNs')).toContain(sequenceNotebookLabs.cnns.code.split('\n\n')[1])
    expect(source('17AutoEncoders')).toContain(sequenceNotebookLabs.autoencoders.code.split('\n\n').slice(1).join('\n\n'))
    for (const [filename, id, classes] of [
      ['19Attention', 'attention', ['TinyMLP', 'SelfAttentionHead']],
      ['20CausalAttention', 'causal-attention', ['CausalSelfAttention']],
    ] as const) {
      for (const name of classes) {
        const code = sequenceNotebookLabs[id].code.split(`class ${name}`)[1].split('\n\nclass ')[0]
        expect(source(filename)).toContain(`class ${name}${code}`)
      }
    }
  })
})

const tracePython = process.env.ML_NOTEBOOK_PYTHON
describe.skipIf(!tracePython)('isolated PyTorch execution of all displayed lab text', () => {
  const checks: Record<string, string> = {
    cnns: `assert tuple(first_pool.shape) == (2,32,25,18)
assert tuple(second_pool.shape) == (2,64,12,9)
assert tuple(flat.shape) == (2,3072)
assert tuple(logits.shape) == (2,7)
assert tuple(predicted_ids.shape) == (2,)
assert dropout_active is False`,
    autoencoders: `assert tuple(encoded.shape) == (2,64,12,9)
assert tuple(first_expansion.shape) == (2,32,24,18)
assert tuple(reconstructed.shape) == (2,1,48,36)
assert torch.allclose(eval_pixel, torch.tensor([0.9]))
assert alternate_shape == (2,32,23,17)`,
    tokenization: `assert punctuation_probe == ["kite", "!", "kite", "."]
assert period_id == 4
assert unknown_tokens == ["<bos>", "Ava", "found", "a", "<unk>", ".", "<eos>"]
assert tuple(input_vectors.shape) == (1,12,8)
assert target_tokens == ["Ava", "found", "a", "blue", "kite", "."]
assert overlap_matches`,
    attention: `assert bridge_outputs_match
assert custom_parameter_names == ["first.weight", "first.bias", "second.weight", "second.bias"]
assert custom_parameter_count == 58
assert tuple(learned_weights.shape) == (1,4,4)
assert tuple(learned_output.shape) == (1,4,2)
assert torch.allclose(row_sums, torch.ones(1,4))`,
    'causal-attention': `assert causal_mask[2].int().tolist() == [1,1,1,0]
assert causal_weights[0].tolist() == [1.,0.,0.,0.]
assert attention.head_size == 4
assert tuple(attention_weights.shape) == (3,2,6,6)
assert mask_is_parameter is False
assert prefix_matches`,
  }
  for (const [id, lab] of Object.entries(sequenceNotebookLabs)) {
    it(`executes the exact displayed ${id} fixture and verifies every numeric or Boolean checkpoint`, () => {
      const script = [lab.code, lab.fixture, lab.invocation, checks[id]].join('\n\n')
      const result = spawnSync(tracePython!, ['-c', script], { encoding: 'utf8', timeout: 60000 })
      expect(result.stderr, `Python stderr for ${id}`).toBe('')
      expect(result.status, `Python status for ${id}: ${result.stderr}`).toBe(0)
    }, 60000)
  }
})
