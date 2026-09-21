/// <reference types="node" />
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

type Cell = { cell_type: string; source: string[]; execution_count?: number | null; outputs?: unknown[] }
const notebook = JSON.parse(readFileSync('Homework2026/13PyTorch.ipynb', 'utf8')) as {
  nbformat: number; cells: Cell[]; metadata: { kernelspec: { language: string } }
}
const code = notebook.cells.filter(cell => cell.cell_type === 'code').map(cell => cell.source.join(''))
const prose = notebook.cells.filter(cell => cell.cell_type === 'markdown').map(cell => cell.source.join('')).join('\n')

describe('corrected Homework 13 notebook', () => {
  it('keeps the completed examples executable with clean Colab-compatible notebook state', () => {
    expect(notebook.nbformat).toBe(4)
    expect(notebook.metadata.kernelspec.language).toBe('python')
    for (const cell of notebook.cells.filter(cell => cell.cell_type === 'code')) {
      expect(cell.execution_count).toBeNull()
      expect(cell.outputs).toEqual([])
      expect(cell.source.some(line => line.trim() && !line.trim().startsWith('#'))).toBe(true)
    }
    expect(code.some(source => source.startsWith('network=Sequential('))).toBe(true)
    expect(code.join('\n')).toContain('torch.manual_seed(158)')
  })

  it('pairs regression targets by row and uses safe explicit gradient updates', () => {
    const training = code.filter(source => source.startsWith('for i in range(10000)'))
    expect(training).toHaveLength(3)
    expect(training[0]).toContain('MSELoss()(prediction,target)')
    expect(training[0]).not.toContain('mpg')
    expect(code.join('\n')).toContain('mpg=torch.tensor(cars.mpg.to_numpy(),dtype=torch.float32).reshape(-1,1)')
    expect(code.join('\n')).toContain('DWG=(DWG-DWG_mean)/DWG_std')
    expect(training[1]).toContain('assert prediction.shape==mpg.shape')
    for (const source of training) {
      expect(source).toContain('with torch.no_grad():')
      expect(source).not.toContain('param.data')
    }
  })

  it('computes final results instead of overwriting them with cached answers', () => {
    const source = code.join('\n')
    for (const name of ['final_mse', 'class1_prob', 'class2_prob', 'accuracy']) {
      expect(source).not.toMatch(new RegExp(`^\\s*${name}\\s*=\\s*[0-9]`, 'm'))
    }
    expect(source).toContain('final_mse=torch.nn.MSELoss()(network(DWG),mpg).item()')
    expect(source).toContain('accuracy=(predictions==y).float().mean().item()')
    expect(prose).toContain('two hidden layers, each with 10 neurons')
    expect(prose).toContain('training metrics, not estimates of performance on unseen observations')
  })
})
