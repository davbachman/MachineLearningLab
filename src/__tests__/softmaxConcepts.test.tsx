import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import { SoftmaxVisual } from '../components/questions/SoftmaxVisual'
import { softmaxAssignment, softmaxVisualDatasets } from '../data/softmaxAssignment'
import { softmaxNotebookLab } from '../data/classificationCodeLabQuestions'
import { baseSoftmaxScores, boundaryScores, categoricalLoss, comparisonPredictions, scoreGradient, softmaxProbabilities, transformedScores, updateProbabilities } from '../lib/softmaxConcepts'

afterEach(cleanup)
describe('softmax conceptual assignment',()=>{
  it('keeps the supplied notebook lab last and uses only conceptual warm-ups',()=>{
    expect(softmaxAssignment.questions.at(-1)).toBe(softmaxNotebookLab)
    expect(softmaxAssignment.version).toBe(2)
    for(const q of softmaxAssignment.questions.slice(0,-1)) {
      expect(q.kind).toBe('multipleChoice')
      expect(JSON.stringify(q)).not.toMatch(/notebook|ipynb|numpy|self\.|np\./i)
    }
  })
  it('normalizes, preserves ranking, and is invariant to common shifts even for large scores',()=>{
    const base=softmaxProbabilities(baseSoftmaxScores)
    expect(base.reduce((a,b)=>a+b,0)).toBeCloseTo(1,12)
    for(const shift of [-1000,-3,0,3,1000]) {
      softmaxProbabilities(transformedScores(shift,1)).forEach((p,i)=>expect(p).toBeCloseTo(base[i],12))
    }
    expect(softmaxProbabilities(transformedScores(0,2))[0]).toBeGreaterThan(base[0])
    expect(softmaxProbabilities([0,0,0])).toEqual([1/3,1/3,1/3])
  })
  it('distinguishes score ties from ties for the largest score',()=>{
    const p=boundaryScores(1,1),q=boundaryScores(-1,-1)
    expect(p).toEqual([1,1,0])
    expect(q).toEqual([-1,-1,0])
    expect(softmaxProbabilities(p)[0]).toBe(softmaxProbabilities(p)[1])
    expect(softmaxProbabilities(q)[2]).toBeGreaterThan(softmaxProbabilities(q)[0])
    // Along positive A–B line, a slight crossing changes the leading class; not on its negative extension.
    for(const x of [.5,1,1.5]) {
      const sides=[boundaryScores(x-.01,x),boundaryScores(x+.01,x)]
      expect(sides.map(s=>s.indexOf(Math.max(...s)))).toEqual([1,0])
    }
    expect(boundaryScores(-1+.01,-1).indexOf(0)).toBe(2)
  })
  it('uses observed-class probability and the true cross-entropy score gradient',()=>{
    const [u,v]=comparisonPredictions.map(m=>categoricalLoss(m.probabilities,1))
    expect(u).toBe(v)
    expect(u).toBeCloseTo(-Math.log(.6),12)
    const grad=scoreGradient(updateProbabilities,1)
    expect(grad).toEqual([.5,-.7,.2])
    const scores=updateProbabilities.map(p=>Math.log(p)),h=1e-5
    for(let i=0;i<3;i++) {
      const plus=scores.map((s,j)=>s+(i===j?h:0)), minus=scores.map((s,j)=>s-(i===j?h:0))
      expect((categoricalLoss(softmaxProbabilities(plus),1)-categoricalLoss(softmaxProbabilities(minus),1))/(2*h)).toBeCloseTo(grad[i],6)
    }
    const next=softmaxProbabilities(scores.map((s,i)=>s-.2*grad[i]))
    expect(categoricalLoss(next,1)).toBeLessThan(categoricalLoss(updateProbabilities,1))
  })
  it('supports score experiments without changing the graded comparison conditions',()=>{
    render(<SoftmaxVisual dataset={softmaxVisualDatasets.softmaxExplorer} />)
    const initial=screen.getByLabelText('Current class probabilities').textContent
    fireEvent.change(screen.getByRole('slider',{name:'Common score shift'}),{target:{value:'3'}})
    expect(screen.getByLabelText('Current class probabilities').textContent).toBe(initial)
    fireEvent.change(screen.getByRole('slider',{name:'Score spread multiplier'}),{target:{value:'2'}})
    expect(screen.getByLabelText('Current class probabilities').textContent).not.toBe(initial)
    fireEvent.click(screen.getByRole('button',{name:'Reset scores'}))
    expect(screen.getByLabelText('Current class probabilities').textContent).toBe(initial)
  })
  it('lets students probe both marked boundary points',()=>{
    render(<SoftmaxVisual dataset={softmaxVisualDatasets.softmaxBoundary} />)
    fireEvent.click(screen.getByRole('button',{name:'Inspect P'}))
    expect(screen.getByLabelText('Leading probe classes')).toHaveTextContent('A and B (tie)')
    fireEvent.click(screen.getByRole('button',{name:'Inspect Q'}))
    expect(screen.getByLabelText('Leading probe classes')).toHaveTextContent('Highest score: C')
    fireEvent.change(screen.getByRole('slider',{name:'Probe feature 1'}),{target:{value:'2'}})
    expect(screen.getByLabelText('Leading probe classes')).toHaveTextContent('Highest score: A')
  })
  it('replays one score update and exposes a decreasing loss',()=>{
    render(<SoftmaxVisual dataset={softmaxVisualDatasets.softmaxUpdates} />)
    const initial=screen.getByLabelText('Score update cross entropy').textContent
    fireEvent.change(screen.getByRole('slider',{name:'Score update learning rate'}),{target:{value:'.5'}})
    expect(screen.getByLabelText('Score changes')).toHaveTextContent('(-0.250, 0.350, -0.100)')
    expect(screen.getByLabelText('Score update cross entropy').textContent).not.toBe(initial)
  })
})
