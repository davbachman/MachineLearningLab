import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { PolynomialDegreeQuestion } from '../components/questions/PolynomialDegreeQuestion'
import { degreeCoefficients, polynomialDegreeQuestion, polynomialFit, predictPolynomial, trainingPoints } from '../data/polynomialDegreeData'
import type { QuestionState } from '../types'

afterEach(cleanup)
const state: QuestionState = { status: 'active', attempts: 0, incorrectAttempts: 0, hintsShown: 0, resolvedAt: null, latestAnswer: null, attemptHistory: [] }

describe('polynomial degree exploration', () => {
  it('uses least-squares fits with unique, different training and validation minima', () => {
    // The least-squares residual is orthogonal to every design-matrix column.
    degreeCoefficients.forEach((coefficients, i) => {
      for (let power = 0; power <= i + 1; power++) {
        const dot = trainingPoints.reduce((sum, [x, y]) => sum + x ** power * (y - predictPolynomial(coefficients, x)), 0)
        expect(Math.abs(dot)).toBeLessThan(1e-7)
      }
      expect(polynomialDegreeQuestion('training').validator({ degree: i + 1 }).correct).toBe(i + 1 === 8)
      expect(polynomialDegreeQuestion('validation').validator({ degree: i + 1 }).correct).toBe(i + 1 === 2)
      if (i > 0) expect(polynomialFit(i + 1).trainingMse).toBeLessThan(polynomialFit(i).trainingMse)
    })
    for (const target of ['training', 'validation'] as const) {
      for (const answer of [null, {}, { degree: 0 }, { degree: 9 }, { degree: 2.5 }, { degree: '2' }, { degree: NaN }, { degree: 1, trainingMse: 0, validationMse: 0 }]) {
        expect(polynomialDegreeQuestion(target).validator(answer).correct).toBe(false)
      }
    }
  })

  it.each(['training', 'validation'] as const)('updates the curve and both MSEs and records the %s answer', target => {
    const onAttempt = vi.fn()
    render(<PolynomialDegreeQuestion question={polynomialDegreeQuestion(target)} state={state}
      questionNumber={2} totalQuestions={7} hints={[]} onAttempt={onAttempt} />)
    const initial = screen.getByLabelText('Training MSE').textContent
    const degree = target === 'training' ? 8 : 2
    fireEvent.change(screen.getByRole('slider', { name: 'Polynomial degree' }), { target: { value: String(degree) } })
    const fit = polynomialFit(degree)
    expect(screen.getByLabelText('Training MSE').textContent).not.toBe(initial)
    expect(screen.getByLabelText('Training MSE')).toHaveTextContent(fit.trainingMse.toFixed(4))
    expect(screen.getByLabelText('Validation MSE')).toHaveTextContent(fit.validationMse.toFixed(4))
    expect(screen.getByRole('img')).toHaveAttribute('aria-label', expect.stringContaining(`Degree ${degree}`))
    fireEvent.click(screen.getByRole('button', { name: 'Check degree' }))
    expect(onAttempt).toHaveBeenLastCalledWith('correct', { degree, trainingMse: fit.trainingMse, validationMse: fit.validationMse })
    fireEvent.change(screen.getByRole('slider'), { target: { value: '1' } })
    fireEvent.click(screen.getByRole('button', { name: 'Check degree' }))
    expect(onAttempt).toHaveBeenLastCalledWith('incorrect', expect.objectContaining({ degree: 1 }))
  })

  it('restores submitted degrees and allows revisiting a completed question', () => {
    render(<PolynomialDegreeQuestion question={polynomialDegreeQuestion('validation')}
      state={{ ...state, status: 'correct', latestAnswer: { degree: 2 } }} questionNumber={3}
      totalQuestions={7} hints={[]} onAttempt={vi.fn()} />)
    expect(screen.getByRole('slider')).toHaveValue('2')
    expect(screen.getByRole('button', { name: 'Check degree' })).toBeEnabled()
    fireEvent.change(screen.getByRole('slider'), { target: { value: '8' } })
    expect(screen.getByLabelText('Training MSE')).toHaveTextContent('0.0252')
    expect(screen.getByLabelText('Validation MSE')).toHaveTextContent('0.1220')
  })
})
