import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { LinearFitQuestion } from '../components/questions/LinearFitQuestion'
import { linearFitQuestion } from '../data/linearFitQuestion'
import { fitLeastSquaresLine, residualSumOfSquares } from '../lib/linearRegression'

afterEach(cleanup)

describe('RSS line fitting', () => {
  it('accepts near-optimal parameters, rejects excess error, and ignores a claimed RSS', () => {
    const best = fitLeastSquaresLine(linearFitQuestion.points)
    // Shifting the intercept by d adds n*d^2 to the least-squares RSS.
    const boundary = Math.sqrt(best.rss * 0.05 / linearFitQuestion.points.length)
    expect(linearFitQuestion.validator(best).correct).toBe(true)
    expect(linearFitQuestion.validator({ slope: best.slope, intercept: best.intercept + 0.99 * boundary }).correct).toBe(true)
    expect(linearFitQuestion.validator({ slope: best.slope, intercept: best.intercept + 1.01 * boundary, rss: 0 }).correct).toBe(false)
    for (const answer of [{ slope: NaN, intercept: 0 }, { slope: 1, intercept: Infinity }, {}, null]) {
      expect(linearFitQuestion.validator(answer).correct).toBe(false)
    }
  })

  it('updates RSS with sliders and handles and records raw parameters on check', () => {
    const onAttempt = vi.fn()
    render(<LinearFitQuestion question={linearFitQuestion}
      state={{ status: 'active', attempts: 0, incorrectAttempts: 0, hintsShown: 0,
        resolvedAt: null, latestAnswer: null, attemptHistory: [] }}
      questionNumber={1} totalQuestions={2} hints={[]} onAttempt={onAttempt} />)
    const initial = screen.getByLabelText('Residual sum of squares').textContent
    fireEvent.keyDown(screen.getByRole('button', { name: /Right line handle/ }), { key: 'ArrowUp' })
    expect(screen.getByLabelText('Residual sum of squares').textContent).not.toBe(initial)
    fireEvent.change(screen.getByRole('slider', { name: 'Slope' }), { target: { value: '0.9' } })
    fireEvent.change(screen.getByRole('slider', { name: 'Intercept' }), { target: { value: '2.73' } })
    const rss = residualSumOfSquares(linearFitQuestion.points, 0.9, 2.73)
    expect(screen.getByLabelText('Residual sum of squares')).toHaveTextContent(rss.toFixed(3))
    fireEvent.click(screen.getByRole('button', { name: 'Check line' }))
    expect(onAttempt).toHaveBeenLastCalledWith('correct', { slope: 0.9, intercept: 2.73, rss })
    fireEvent.change(screen.getByRole('slider', { name: 'Slope' }), { target: { value: '-1' } })
    fireEvent.click(screen.getByRole('button', { name: 'Check line' }))
    expect(onAttempt).toHaveBeenLastCalledWith('incorrect', expect.objectContaining({ slope: -1 }))
  })
})
