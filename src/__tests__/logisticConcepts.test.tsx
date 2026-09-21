import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { logisticRegressionAssignment, logisticThresholdQuestion, logisticVisualDatasets } from '../data/logisticAssignment'
import { logisticRegressionNotebookLab } from '../data/classificationCodeLabQuestions'
import { binaryCrossEntropy, isLogisticThreshold, logisticBoundaryPoints, logisticBoundaryScore, logisticLossLabels, logisticLossModels, logisticThresholdMetrics, logisticThresholdObservations, logisticThresholdValues, sigmoid } from '../lib/logisticConcepts'
import { LogisticThresholdQuestion } from '../components/questions/LogisticThresholdQuestion'
import { LogisticVisual } from '../components/questions/LogisticVisual'
import type { QuestionState } from '../types'

afterEach(cleanup)
const state: QuestionState = { status: 'active', attempts: 0, incorrectAttempts: 0, hintsShown: 0, resolvedAt: null, latestAnswer: null, attemptHistory: [] }

describe('conceptual logistic regression assignment', () => {
  it('preserves the supplied notebook lab and keeps all warm-ups self-contained', () => {
    expect(logisticRegressionAssignment.version).toBe(2)
    expect(logisticRegressionAssignment.questions).toHaveLength(5)
    expect(logisticRegressionAssignment.questions.at(-1)).toBe(logisticRegressionNotebookLab)
    for (const question of logisticRegressionAssignment.questions.slice(0, -1)) {
      expect(JSON.stringify(question)).not.toMatch(/notebook|ipynb|numpy|self\.|np\./i)
      if (question.kind === 'multipleChoice') expect(logisticVisualDatasets[question.datasetId].kind).toBe('logisticVisual')
    }
  })

  it('maps scores monotonically with symmetry and saturation, preserving the exact boundary', () => {
    expect(sigmoid(0)).toBe(0.5)
    expect(sigmoid(-2)).toBeCloseTo(0.119202922, 8)
    expect(sigmoid(2)).toBeCloseTo(0.880797078, 8)
    expect(sigmoid(4)).toBeCloseTo(0.98201379, 8)
    expect(sigmoid(4) - sigmoid(2)).toBeLessThan(sigmoid(2) - sigmoid(0))
    expect(sigmoid(-1000)).toBe(0)
    expect(sigmoid(1000)).toBe(1)
    for (const score of [-5, -1, 0, 1, 5]) expect(sigmoid(score) + sigmoid(-score)).toBeCloseTo(1, 14)
  })

  it('supports the geometric key including the boundary tie and parallel threshold shift', () => {
    expect(logisticBoundaryPoints.map(point => logisticBoundaryScore(point.point))).toEqual([-1, 0.5, 0, 1.5])
    expect(logisticBoundaryPoints.filter(point => sigmoid(logisticBoundaryScore(point.point)) > 0.5).map(point => point.id)).toEqual(['B', 'D'])
    expect(logisticBoundaryPoints.filter(point => sigmoid(logisticBoundaryScore(point.point)) > 0.75).map(point => point.id)).toEqual(['D'])
    for (const threshold of [0.25, 0.5, 0.75]) {
      const scoreCutoff = Math.log(threshold / (1 - threshold))
      const points: [number, number][] = [0, 1, 2].map(x1 => [x1, (4 + scoreCutoff - x1) / 2])
      points.forEach(point => expect(sigmoid(logisticBoundaryScore(point))).toBeCloseTo(threshold, 12))
      expect(points[1][1] - points[0][1]).toBeCloseTo(-0.5, 12)
    }
  })

  it('independently checks all slider choices and accepts both equally good thresholds', () => {
    const positiveProbabilities = [0.28, 0.47, 0.54, 0.72, 0.91]
    const negativeProbabilities = [0.08, 0.17, 0.38, 0.61, 0.84]
    const accepted: number[] = []
    for (const threshold of logisticThresholdValues) {
      const truePositives = positiveProbabilities.filter(p => p > threshold).length
      const falsePositives = negativeProbabilities.filter(p => p > threshold).length
      const correct = truePositives >= 4 && truePositives / (truePositives + falsePositives) >= 2 / 3
      expect(logisticThresholdMetrics(threshold)).toEqual({
        truePositives, falsePositives, falseNegatives: 5 - truePositives, trueNegatives: 5 - falsePositives,
        precision: truePositives + falsePositives === 0 ? null : truePositives / (truePositives + falsePositives), recall: truePositives / 5,
      })
      expect(logisticThresholdQuestion.validator({ threshold }).correct).toBe(correct)
      if (correct) accepted.push(threshold)
    }
    expect(accepted).toEqual([0.4, 0.45])
    expect(logisticThresholdMetrics(0.47).truePositives).toBe(3) // Strict >, not >=, even exactly on a probability.
    expect(logisticThresholdMetrics(0.95).precision).toBeNull()
    expect(logisticThresholdObservations).toHaveLength(10)
  })

  it('rejects malformed thresholds and never trusts claimed confusion counts or metrics', () => {
    for (const answer of [null, {}, [], { threshold: '0.4' }, { threshold: NaN }, { threshold: Infinity }, { threshold: -0.1 }, { threshold: 0 }, { threshold: 1 }, { threshold: 0.425 }, { threshold: 0.75, precision: 1, recall: 1, truePositives: 5, falsePositives: 0 }]) {
      expect(logisticThresholdQuestion.validator(answer).correct).toBe(false)
    }
    expect(isLogisticThreshold(0.4)).toBe(true)
    expect(logisticThresholdQuestion.validator({ threshold: 0.4, precision: 0, recall: 0 }).correct).toBe(true)
  })

  it('compares cross-entropy separately from hard-label accuracy and threshold selection', () => {
    const losses = logisticLossModels.map(model => model.probabilities.map((p, index) => binaryCrossEntropy(p, logisticLossLabels[index])))
    const predictions = logisticLossModels.map(model => model.probabilities.map(p => Number(p > 0.5)))
    expect(predictions[0]).toEqual([1, 1, 0, 1])
    expect(predictions[1]).toEqual(predictions[0])
    const means = losses.map(values => values.reduce((sum, loss) => sum + loss, 0) / values.length)
    expect(means[0]).toBeCloseTo(-(2 * Math.log(0.7) + Math.log(0.6) + Math.log(0.4)) / 4, 12)
    expect(means[1]).toBeCloseTo(-(3 * Math.log(0.95) + Math.log(0.01)) / 4, 12)
    expect(means[0]).toBeLessThan(means[1])
    expect(means[0]).toBeCloseTo(0.535, 3)
    expect(means[1]).toBeCloseTo(1.190, 3)
    expect(losses[1][3]).toBeCloseTo(4.60517, 5)
  })

  it('renders all plots and lets students explore scores and thresholds without moving reference points', () => {
    let view = render(<LogisticVisual dataset={logisticVisualDatasets.logisticScores} />)
    fireEvent.change(screen.getByRole('slider', { name: 'Explore linear score' }), { target: { value: '4' } })
    expect(screen.getByLabelText('Explored probability')).toHaveTextContent('p = 0.982')
    for (const id of ['A', 'B', 'C']) expect(screen.getByText(id)).toBeInTheDocument()
    view.unmount()
    view = render(<LogisticVisual dataset={logisticVisualDatasets.logisticBoundary} />)
    fireEvent.change(screen.getByRole('slider', { name: 'Explore boundary threshold' }), { target: { value: '0.75' } })
    expect(screen.getByLabelText('Explored boundary')).toHaveTextContent('1.099')
    expect(screen.getByText('p = 0.5')).toBeInTheDocument()
    view.unmount()
    render(<LogisticVisual dataset={logisticVisualDatasets.logisticLoss} />)
    expect(screen.getAllByRole('img')).toHaveLength(2)
    expect(screen.getAllByRole('row')).toHaveLength(5)
  })

  it('updates live counts, records the raw threshold, and restores saved answers for redo', () => {
    const onAttempt = vi.fn()
    const view = render(<LogisticThresholdQuestion question={logisticThresholdQuestion} state={state} questionNumber={3} totalQuestions={5} hints={[]} onAttempt={onAttempt} />)
    expect(screen.getByLabelText('Recall')).toHaveTextContent('60.0%')
    fireEvent.click(screen.getByRole('button', { name: 'Check threshold' }))
    expect(onAttempt).toHaveBeenLastCalledWith('incorrect', expect.objectContaining({ threshold: 0.5 }))
    fireEvent.change(screen.getByRole('slider', { name: 'Probability threshold' }), { target: { value: '0.95' } })
    expect(screen.getByLabelText('Precision')).toHaveTextContent('undefined')
    fireEvent.change(screen.getByRole('slider', { name: 'Probability threshold' }), { target: { value: '0.45' } })
    expect(screen.getByLabelText('Recall')).toHaveTextContent('80.0%')
    expect(screen.getByLabelText('Precision')).toHaveTextContent('66.7%')
    fireEvent.click(screen.getByRole('button', { name: 'Check threshold' }))
    expect(onAttempt).toHaveBeenLastCalledWith('correct', { threshold: 0.45, truePositives: 4, falsePositives: 2, falseNegatives: 1, trueNegatives: 3, precision: 2 / 3, recall: 0.8 })
    view.unmount()
    render(<LogisticThresholdQuestion question={logisticThresholdQuestion} state={{ ...state, status: 'correct', latestAnswer: { threshold: 0.45 } }} questionNumber={3} totalQuestions={5} hints={[]} onAttempt={onAttempt} />)
    expect(screen.getByRole('slider', { name: 'Probability threshold' })).toHaveValue('0.45')
    expect(screen.getByRole('button', { name: 'Check threshold' })).toBeEnabled()
    fireEvent.change(screen.getByRole('slider', { name: 'Probability threshold' }), { target: { value: '0.4' } })
    fireEvent.click(screen.getByRole('button', { name: 'Check threshold' }))
    expect(onAttempt).toHaveBeenLastCalledWith('correct', expect.objectContaining({ threshold: 0.4 }))
  })
})
