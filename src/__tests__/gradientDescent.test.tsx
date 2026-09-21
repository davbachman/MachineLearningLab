import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { gradientDescentAssignment, learningRateQuestion } from '../data/gradientDescentAssignment'
import { gradientDescentNotebookLab } from '../data/regressionCodeLabQuestions'
import { multipleChoiceDatasets } from '../data/multipleChoiceDatasets'
import { descentArrows, descentGradient, descentLoss, descentRuns, descentStart, descentSteps, descentTargetLoss, descentTrajectory, nonconvexGradient, nonconvexLoss } from '../lib/gradientDescent'
import { GradientDescentQuestion } from '../components/questions/GradientDescentQuestion'
import { GradientVisual } from '../components/questions/GradientVisual'
import type { QuestionState } from '../types'

afterEach(cleanup)
const state: QuestionState = { status: 'active', attempts: 0, incorrectAttempts: 0, hintsShown: 0, resolvedAt: null, latestAnswer: null, attemptHistory: [] }

describe('conceptual gradient descent assignment', () => {
  it('leaves the notebook lab intact and makes every warm-up self-contained and graphical', () => {
    expect(gradientDescentAssignment.version).toBe(2)
    expect(gradientDescentAssignment.questions).toHaveLength(6)
    expect(gradientDescentAssignment.questions.at(-1)).toBe(gradientDescentNotebookLab)
    for (const question of gradientDescentAssignment.questions.slice(0, -1)) {
      expect(JSON.stringify(question)).not.toMatch(/notebook|ipynb|numpy|GDRegressor|fGD|self\.|np\./i)
      if (question.kind === 'multipleChoice') expect(multipleChoiceDatasets[question.datasetId].kind).toBe('gradientVisual')
    }
  })

  it('uses the true gradient and simultaneous updates of the plotted bowl', () => {
    expect(descentGradient(descentStart)).toEqual([-6, 16])
    const run = descentTrajectory(.1)
    expect(run).toHaveLength(21)
    expect(run[0].loss).toBe(26)
    expect(run[1].point[0]).toBeCloseTo(-1.4)
    expect(run[1].point[1]).toBeCloseTo(-.6)
    run.forEach(({point:[a,b],loss},step) => {
      // Independent closed-form solution for this diagonal quadratic.
      expect(a).toBeCloseTo(1 - 3 * .8 ** step, 10)
      expect(b).toBeCloseTo(-1 + 2 * .2 ** step, 10)
      expect(loss).toBeCloseTo(1 + 9 * .8 ** (2*step) + 16 * .2 ** (2*step), 10)
    })
    for (const point of [[-2,1],[.5,-.6],[1,-1]] as [number,number][]) {
      const g = descentGradient(point), h=1e-5
      expect(g[0]).toBeCloseTo((descentLoss([point[0]+h,point[1]])-descentLoss([point[0]-h,point[1]]))/(2*h),6)
      expect(g[1]).toBeCloseTo((descentLoss([point[0],point[1]+h])-descentLoss([point[0],point[1]-h]))/(2*h),6)
    }
  })

  it('chooses D by directional derivative, not by aiming at the center', () => {
    const g = descentGradient(descentStart)
    const rates = descentArrows.map(a => ({label:a.label, slope:(g[0]*a.vector[0]+g[1]*a.vector[1])/Math.hypot(...a.vector)}))
    expect(rates.sort((a,b)=>a.slope-b.slope)[0].label).toBe('D')
    expect(rates.find(a=>a.label==='C')!.slope).toBe(0)
    expect(rates.find(a=>a.label==='B')!.slope).toBeGreaterThan(0)
  })

  it('accepts any successful slider rate, rejects bad input, and does not trust claimed loss', () => {
    let accepted = 0
    for(let n=1;n<=30;n++) {
      const rate=n/100, final=1+9*(1-2*rate)**40+16*(1-8*rate)**40
      expect(learningRateQuestion.validator({learningRate:rate}).correct).toBe(final<=descentTargetLoss+1e-10)
      if(final<=descentTargetLoss) accepted++
    }
    expect(accepted).toBeGreaterThan(5)
    for(const answer of [null,{},[],{learningRate:'0.12'},{learningRate:NaN},{learningRate:Infinity},{learningRate:0},{learningRate:.31},{learningRate:.125},{learningRate:.3,finalLoss:1,steps:0,finalParameters:[1,-1]}]) {
      expect(learningRateQuestion.validator(answer).correct).toBe(false)
    }
  })

  it('supports the convergence diagnoses and keeps the loss charts within their fixed axes', () => {
    const [a,b,c]=descentRuns.map(r=>r.trajectory.map(p=>p.loss))
    expect(a.every((v,i)=>i===0 || v<a[i-1])).toBe(true)
    expect(a[20]).toBeGreaterThan(5)
    expect(b[20]).toBeLessThan(1.01)
    expect(c[20]).toBeGreaterThan(c[0])
    expect(c.slice(5).every((v,i)=>i===0 || v>c[i+4])).toBe(true)
    expect(Math.max(...a,...b,...c)).toBeLessThan(90)
  })

  it('distinguishes a local minimum, a stationary maximum, and a global minimum', () => {
    for(const w of [-1,0,2]) expect(nonconvexGradient(w)).toBeCloseTo(0,12)
    expect(nonconvexLoss(2)).toBeLessThan(nonconvexLoss(-1))
    expect(nonconvexLoss(.01)).toBeLessThan(nonconvexLoss(0))
    expect(nonconvexLoss(-.01)).toBeLessThan(nonconvexLoss(0))
    let w=-1.7
    for(let i=0;i<1000;i++) w-=.01*nonconvexGradient(w)
    expect(w).toBeCloseTo(-1,6)
  })

  it('renders each concept plot and supplies exact loss checkpoints', () => {
    for(const id of ['gradientSlopes','gradientContours','gradientRuns','gradientMinima']) {
      const dataset=multipleChoiceDatasets[id]
      if(dataset.kind!=='gradientVisual') throw new Error('Expected gradient visual')
      const view=render(<GradientVisual dataset={dataset} />)
      expect(screen.getAllByRole('img').length).toBe(id==='gradientRuns'?3:1)
      if(id==='gradientRuns') expect(screen.getAllByRole('row')).toHaveLength(6)
      view.unmount()
    }
  })

  it('updates the trajectory, replays steps, and records raw answers for checking and redo', () => {
    const onAttempt=vi.fn()
    const view=render(<GradientDescentQuestion question={learningRateQuestion} state={state} questionNumber={3} totalQuestions={6} hints={[]} onAttempt={onAttempt} />)
    fireEvent.change(screen.getByRole('slider',{name:'Learning rate'}),{target:{value:'.3'}})
    expect(screen.getByText(/path leaves the plotted area/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button',{name:'Check learning rate'}))
    expect(onAttempt).toHaveBeenLastCalledWith('incorrect',expect.objectContaining({learningRate:.3}))
    fireEvent.change(screen.getByRole('slider',{name:'Learning rate'}),{target:{value:'.12'}})
    const final=descentTrajectory(.12)[descentSteps]
    expect(screen.getByLabelText('Final loss')).toHaveTextContent(final.loss.toFixed(4))
    fireEvent.change(screen.getByRole('slider',{name:'Inspect step'}),{target:{value:'0'}})
    expect(screen.getByLabelText('Displayed loss')).toHaveTextContent('26.0000')
    expect(screen.getByLabelText('Final loss')).toHaveTextContent(final.loss.toFixed(4))
    fireEvent.click(screen.getByRole('button',{name:'Check learning rate'}))
    expect(onAttempt).toHaveBeenLastCalledWith('correct',{learningRate:.12,steps:20,finalParameters:final.point,finalLoss:final.loss})
    view.unmount()
    render(<GradientDescentQuestion question={learningRateQuestion} state={{...state,status:'correct',latestAnswer:{learningRate:.12}}} questionNumber={3} totalQuestions={6} hints={[]} onAttempt={onAttempt} />)
    expect(screen.getByRole('slider',{name:'Learning rate'})).toHaveValue('0.12')
    expect(screen.getByRole('button',{name:'Check learning rate'})).toBeEnabled()
  })
})
