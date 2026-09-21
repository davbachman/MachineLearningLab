import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { regularizationAssignment, regularizationTuningQuestion, regularizationVisualDatasets } from '../data/regularizationAssignment'
import { regularizationNotebookLab } from '../data/classificationCodeLabQuestions'
import { RegularizationVisual } from '../components/questions/RegularizationVisual'
import { RegularizationTuningQuestion } from '../components/questions/RegularizationTuningQuestion'
import { bestRidgeStrengthIndex, penaltyCandidates, penaltyPath, ridgeFits, ridgePrediction, ridgeStrengths, ridgeTrainingPoints, ridgeValidationPoints } from '../lib/regularizationConcepts'
import type { QuestionState } from '../types'

afterEach(cleanup)
const state: QuestionState = { status: 'active', attempts: 0, incorrectAttempts: 0, hintsShown: 0, resolvedAt: null, latestAnswer: null, attemptHistory: [] }

describe('conceptual regularization assignment', () => {
  it('keeps the lab intact after five self-contained warm-ups', () => {
    expect(regularizationAssignment.version).toBe(2)
    expect(regularizationAssignment.questions).toHaveLength(6)
    expect(regularizationAssignment.questions.at(-1)).toBe(regularizationNotebookLab)
    for (const question of regularizationAssignment.questions.slice(0,-1)) {
      expect(JSON.stringify(question)).not.toMatch(/notebook|ipynb|numpy|SGDRegressor|self\.|np\./i)
      if (question.kind==='multipleChoice') {
        expect(regularizationVisualDatasets[question.datasetId].kind).toBe('regularizationVisual')
        expect(question.validator({selectedIds:Object.fromEntries(question.parts.map(part=>[part.id,part.correctOptionId]))}).correct).toBe(true)
        for (const part of question.parts) expect(part.options.every(option=>option.description==='')).toBe(true)
      }
    }
  })

  it('uses real least-squares ridge optima with an unpenalized intercept', () => {
    expect(ridgeFits.map(fit=>fit.strength)).toEqual(ridgeStrengths)
    ridgeFits.forEach(fit => {
      // Independently verify every component of the first-order optimality
      // condition for MSE + lambda * ||w[1:]||², using the displayed points.
      for(let j=0;j<=8;j++) {
        const halfDataGradient=ridgeTrainingPoints.reduce((sum,[x,y])=>
          sum+(ridgePrediction(fit.weights,x)-y)*(x/2)**j,0)/ridgeTrainingPoints.length
        expect(halfDataGradient+(j===0?0:fit.strength*fit.weights[j])).toBeCloseTo(0,9)
      }
      const independentMSE=(points: [number,number][])=>points.reduce((sum,[x,y])=> {
        const prediction=fit.weights.reduce((total,weight,power)=>total+weight*(x/2)**power,0)
        return sum+(prediction-y)**2
      },0)/points.length
      expect(fit.trainingMSE).toBeCloseTo(independentMSE(ridgeTrainingPoints),12)
      expect(fit.validationMSE).toBeCloseTo(independentMSE(ridgeValidationPoints),12)
      for(let i=0;i<=400;i++) expect(ridgePrediction(fit.weights,-2+4*i/400)).toBeGreaterThanOrEqual(0)
      for(let i=0;i<=400;i++) expect(ridgePrediction(fit.weights,-2+4*i/400)).toBeLessThanOrEqual(5)
    })
    expect(ridgeFits.every((fit,i)=>i===0||fit.trainingMSE>ridgeFits[i-1].trainingMSE)).toBe(true)
    expect(ridgeFits.every((fit,i)=>i===0||fit.squaredWeights<ridgeFits[i-1].squaredWeights)).toBe(true)
    expect(bestRidgeStrengthIndex).toBe(1)
    expect(ridgeFits[1].validationMSE).toBeCloseTo(.08464554554171448,10)
    expect(ridgeFits[0].validationMSE).toBeGreaterThan(ridgeFits[1].validationMSE)
    expect(ridgeFits.at(-1)!.validationMSE).toBeGreaterThan(.8)
  })

  it('grades only the raw allowed slider choice, never claimed metrics', () => {
    ridgeFits.forEach((_,index)=>expect(regularizationTuningQuestion.validator({strengthIndex:index}).correct).toBe(index===1))
    for (const answer of [null,[],{}, {strengthIndex:'1'}, {strengthIndex:1.1}, {strengthIndex:-1}, {strengthIndex:6}, {strengthIndex:NaN}, {strengthIndex:Infinity}, {strengthIndex:0,validationMSE:0,strength:.001}]) {
      expect(regularizationTuningQuestion.validator(answer).correct).toBe(false)
    }
  })

  it('distinguishes the candidate objective from training error and minimum coefficient norm', () => {
    expect(penaltyCandidates.map(model=>Number(model.trainingMSE.toFixed(2)))).toEqual([.45,2.25,7.65])
    expect(penaltyCandidates.map(model=>Number(model.objective.toFixed(2)))).toEqual([6.21,3.69,7.65])
    expect([...penaltyCandidates].sort((a,b)=>a.objective-b.objective)[0].name).toBe('B')
  })

  it('uses exact sparse L1 and shrinking L2 paths for the specified objectives', () => {
    expect(penaltyPath('l1',.6)).toEqual([1.4,-.4,0])
    expect(penaltyPath('l2',.6)).toEqual([1.25,-.625,.25])
    const z=[2,-1,.4]
    for(let i=0;i<=20;i++) {
      const lambda=i/10
      for (const kind of ['l1','l2'] as const) {
        const weights=penaltyPath(kind,lambda)
        weights.forEach((w,j)=>{
          if(kind==='l2') expect(w-z[j]+lambda*w).toBeCloseTo(0,12)
          else if(w!==0) expect(w-z[j]+lambda*Math.sign(w)).toBeCloseTo(0,12)
          else expect(Math.abs(z[j])).toBeLessThanOrEqual(lambda+1e-12)
        })
      }
    }
  })

  it('renders every concept visual and explores coefficient paths without changing the fixed reference', () => {
    for(const dataset of Object.values(regularizationVisualDatasets)) {
      const view=render(<RegularizationVisual dataset={dataset} />)
      expect(screen.getAllByRole('img').length).toBeGreaterThan(0)
      if(dataset.mode==='paths') {
        fireEvent.change(screen.getByRole('slider',{name:'Explore coefficient paths'}),{target:{value:'1'}})
        expect(screen.getByLabelText('Plot A inspected coefficients')).toHaveTextContent('(1.000, 0.000, 0.000)')
        expect(screen.getByRole('table')).toHaveAccessibleName('Fixed reference for the question: λ = 0.6')
        expect(screen.getByRole('table')).toHaveTextContent('1.400')
      }
      if(dataset.mode==='fit') {
        fireEvent.click(screen.getByRole('checkbox',{name:'Show vertical training residuals'}))
        expect(screen.getByRole('checkbox')).toBeChecked()
      }
      view.unmount()
    }
  })

  it('updates both MSE readouts and saves a raw choice that can be revisited and redone', () => {
    const onAttempt=vi.fn()
    const props={question:regularizationTuningQuestion,state,questionNumber:2,totalQuestions:6,hints:[],onAttempt}
    const view=render(<RegularizationTuningQuestion {...props} />)
    expect(screen.getByLabelText('Training MSE')).toHaveTextContent('0.0252')
    fireEvent.change(screen.getByRole('slider',{name:'L2 strength'}),{target:{value:'1'}})
    expect(screen.getByLabelText('Training MSE')).toHaveTextContent('0.0658')
    expect(screen.getByLabelText('Validation MSE')).toHaveTextContent('0.0846')
    fireEvent.click(screen.getByRole('button',{name:'Check penalty strength'}))
    expect(onAttempt).toHaveBeenLastCalledWith('correct',expect.objectContaining({strengthIndex:1,strength:.001}))
    const answer=onAttempt.mock.calls[0][1]
    view.unmount()
    render(<RegularizationTuningQuestion {...props} state={{...state,status:'correct',latestAnswer:answer}} />)
    expect(screen.getByRole('slider',{name:'L2 strength'})).toHaveValue('1')
    fireEvent.change(screen.getByRole('slider',{name:'L2 strength'}),{target:{value:'5'}})
    fireEvent.click(screen.getByRole('button',{name:'Check penalty strength'}))
    expect(onAttempt).toHaveBeenLastCalledWith('incorrect',expect.objectContaining({strengthIndex:5,strength:10}))
  })
})
