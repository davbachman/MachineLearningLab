import { useState } from 'react'
import type { AttemptOutcome, QuestionState } from '../../types'
import type { LogisticThresholdQuestionSpec } from '../../data/logisticAssignment'
import { isLogisticThreshold, logisticThresholdMetrics, logisticThresholdObservations } from '../../lib/logisticConcepts'
import { QuestionFrame } from '../QuestionFrame'

interface Props {
  question: LogisticThresholdQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
}

function ThresholdPlot({ threshold }: { threshold: number }) {
  const x = (probability: number) => 86 + probability * 500
  return <svg viewBox="0 0 625 345" style={{ width: '100%' }} role="img" aria-label={`Fixed validation observations, with class 1 predicted when probability is greater than ${threshold.toFixed(2)}`}>
    <rect x={x(threshold)} y="48" width={x(1) - x(threshold)} height="228" fill="#e1ecea" />
    <text x="86" y="25" fontSize="16">Left: predict 0 · right: predict 1</text>
    {[0, 0.2, 0.4, 0.6, 0.8, 1].map(probability => <g key={probability}>
      <line x1={x(probability)} x2={x(probability)} y1="48" y2="276" stroke="#ded9cf" />
      <text x={x(probability)} y="301" textAnchor="middle" fontSize="15">{probability.toFixed(1)}</text>
    </g>)}
    {[0, 1].map(label => <g key={label}>
      <text x="72" y={label ? 105 : 220} textAnchor="end" fontSize="15">True {label}</text>
      <line x1="86" x2="586" y1={label ? 100 : 215} y2={label ? 100 : 215} stroke="#ded9cf" />
    </g>)}
    <line x1={x(threshold)} x2={x(threshold)} y1="40" y2="281" stroke="#77326b" strokeWidth="3" strokeDasharray="6 4" />
    {logisticThresholdObservations.map(observation => {
      const cy = observation.label ? 100 : 215, cx = x(observation.probability)
      return <g key={observation.id}>
        {observation.label ? <circle cx={cx} cy={cy} r="7" fill="#1c7b8a" /> : <rect x={cx - 7} y={cy - 7} width="14" height="14" fill="#b9601b" />}
        <text x={cx} y={cy - 18} textAnchor="middle" fontSize="16" fontWeight="700">{observation.id}</text>
        <text x={cx} y={cy + 29} textAnchor="middle" fontSize="14">{observation.probability.toFixed(2)}</text>
      </g>
    })}
    <text x="336" y="337" textAnchor="middle" fontSize="17">Predicted probability p of class 1</text>
  </svg>
}

const percentage = (value: number | null) => value === null ? 'undefined' : `${(value * 100).toFixed(1)}%`

export function LogisticThresholdQuestion({ question, state, questionNumber, totalQuestions, hints, onAttempt }: Props) {
  const [threshold, setThreshold] = useState(() => {
    const saved = state.latestAnswer as { threshold?: unknown } | null
    return isLogisticThreshold(saved?.threshold) ? saved.threshold : question.initialThreshold
  })
  const [feedback, setFeedback] = useState('')
  const metrics = logisticThresholdMetrics(threshold)
  return <QuestionFrame question={question} state={state} questionNumber={questionNumber} totalQuestions={totalQuestions}
    hints={hints} feedback={feedback} controls={<button className="button" type="button" onClick={() => {
      const answer = { threshold, ...metrics }
      const result = question.validator(answer)
      setFeedback(result.message ?? '')
      onAttempt(result.correct ? 'correct' : 'incorrect', answer)
    }}>Check threshold</button>}>
    <div className="visual-grid" style={{ gridTemplateColumns: 'minmax(0, 1fr)' }}>
      <section className="visual-panel">
        <h3 className="panel-title">Ten held-out validation observations</h3>
        <ThresholdPlot threshold={threshold} />
        <p className="panel-note">Circles: true class 1. Squares: true class 0. The purple dashed line is the threshold; observations to its right are predicted positive. Every observation and probability stays fixed as you move it.</p>
      </section>
      <section className="projection-panel" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem' }}>
        <h3 className="panel-title" style={{ gridColumn: '1 / -1' }}>Catch at least 80% of the positives</h3>
        <label className="slider-shell" style={{ gridColumn: '1 / -1' }}><span>Probability threshold: {threshold.toFixed(2)}</span>
          <input className="range-input" type="range" min="0.05" max="0.95" step="0.05" aria-label="Probability threshold"
            value={threshold} onChange={event => { setThreshold(Number(event.target.value)); setFeedback('') }} />
          <span className="panel-note">0.05 — 0.95 · increments of 0.05</span>
        </label>
        <div className="status-panel" aria-live="polite" aria-atomic="true">
          <p><output aria-label="Precision" style={{ fontSize: '1.25rem', fontWeight: 700 }}>Precision: {percentage(metrics.precision)}</output></p>
          <p><output aria-label="Recall" style={{ fontSize: '1.25rem', fontWeight: 700 }}>Recall: {percentage(metrics.recall)}</output></p>
          <p className="panel-note">{metrics.recall !== null && metrics.recall >= 0.8 ? 'Recall requirement met. Can you improve precision?' : 'Recall requirement not met.'}</p>
        </div>
        <table className="table-grid">
          <caption>Live counts for the current threshold</caption>
          <thead><tr><th></th><th>Predicted 1</th><th>Predicted 0</th></tr></thead>
          <tbody><tr><th>True 1</th><td>{metrics.truePositives} TP</td><td>{metrics.falseNegatives} FN</td></tr>
            <tr><th>True 0</th><td>{metrics.falsePositives} FP</td><td>{metrics.trueNegatives} TN</td></tr></tbody>
        </table>
        <p className="panel-note" style={{ gridColumn: '1 / -1' }}>Precision = TP / (TP + FP). Recall = TP / (TP + FN). A positive prediction is class 1. Maximize precision among thresholds meeting the recall requirement.</p>
      </section>
    </div>
  </QuestionFrame>
}
