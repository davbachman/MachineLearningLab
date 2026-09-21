import { useState } from 'react'
import type { AttemptOutcome, GradientDescentQuestionSpec, QuestionState } from '../../types'
import { descentSteps, descentTargetLoss, descentTrajectory, formatLoss, isDescentLearningRate } from '../../lib/gradientDescent'
import { QuestionFrame } from '../QuestionFrame'
import { ContourPlot } from './GradientVisual'

interface Props {
  question: GradientDescentQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
}

export function GradientDescentQuestion({ question, state, questionNumber, totalQuestions, hints, onAttempt }: Props) {
  const [rate, setRate] = useState(() => {
    const saved = state.latestAnswer as { learningRate?: unknown } | null
    return isDescentLearningRate(saved?.learningRate) ? saved.learningRate : question.initialLearningRate
  })
  const [step, setStep] = useState(descentSteps)
  const [feedback, setFeedback] = useState('')
  const trajectory = descentTrajectory(rate)
  const current = trajectory[step], final = trajectory[descentSteps]
  const visible = trajectory.slice(0, step + 1)
  const outside = visible.some(({point:[a,b]}) => a < -5 || a > 7 || b < -7 || b > 5)
  return <QuestionFrame question={question} state={state} questionNumber={questionNumber} totalQuestions={totalQuestions}
    hints={hints} feedback={feedback} controls={<button className="button" type="button" onClick={() => {
      const answer = { learningRate: rate, steps: descentSteps, finalParameters: final.point, finalLoss: final.loss }
      const result = question.validator(answer)
      setFeedback(result.message ?? '')
      onAttempt(result.correct ? 'correct' : 'incorrect', answer)
    }}>Check learning rate</button>}>
    <div className="visual-grid">
      <section className="visual-panel">
        <h3 className="panel-title">Path through step {step}</h3>
        <ContourPlot points={visible.map(p => p.point)} />
        <p className="panel-note">Orange: update path. Purple dot: displayed step. ×: minimum. Contour labels are loss values; axes are model parameters.</p>
        {outside && <p className="feedback" role="status">The path leaves the plotted area. Off-screen steps still count and are included in the loss readouts.</p>}
      </section>
      <section className="projection-panel">
        <h3 className="panel-title">One fixed rate, 20 updates</h3>
        <p className="panel-note">L(a, b) = 1 + (a − 1)² + 4(b + 1)². Each update subtracts learning rate × gradient from the current parameter pair.</p>
        <label className="slider-shell"><span>Learning rate: {rate.toFixed(2)}</span>
          <input className="range-input" type="range" min="0.01" max="0.30" step="0.01" aria-label="Learning rate" value={rate}
            onChange={e => { setRate(Number(e.target.value)); setStep(descentSteps); setFeedback('') }} />
          <span className="panel-note">0.01 — 0.30</span>
        </label>
        <div className="status-panel" aria-live="polite" aria-atomic="true">
          <p><output aria-label="Final loss" style={{fontSize:'1.3rem',fontWeight:700}}>Loss after 20 steps: {formatLoss(final.loss)}</output></p>
          <p className="panel-note">Target: {descentTargetLoss.toFixed(2)} or below. Minimum possible: 1.</p>
        </div>
        <label className="slider-shell"><span>Inspect step: {step}</span>
          <input className="range-input" type="range" min="0" max={descentSteps} step="1" aria-label="Inspect step" value={step} onChange={e => setStep(Number(e.target.value))} />
        </label>
        <p><output aria-label="Displayed parameters">(a, b) = ({current.point[0].toFixed(3)}, {current.point[1].toFixed(3)})</output></p>
        <p><output aria-label="Displayed loss">Loss at step {step}: {formatLoss(current.loss)}</output></p>
        <p className="panel-note">The step viewer only replays this run; your answer is always evaluated after all 20 updates.</p>
      </section>
    </div>
  </QuestionFrame>
}
