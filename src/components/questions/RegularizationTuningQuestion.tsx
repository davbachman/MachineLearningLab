import { useState } from 'react'
import type { AttemptOutcome, QuestionState } from '../../types'
import type { RegularizationTuningQuestionSpec } from '../../data/regularizationAssignment'
import { isRidgeStrengthIndex, ridgeFits, ridgeStrengths } from '../../lib/regularizationConcepts'
import { QuestionFrame } from '../QuestionFrame'
import { RidgeModelPlot } from './RegularizationVisual'

interface Props {
  question: RegularizationTuningQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
}

export function RegularizationTuningQuestion({ question, state, questionNumber, totalQuestions, hints, onAttempt }: Props) {
  const [index, setIndex] = useState(() => {
    const saved = state.latestAnswer as { strengthIndex?: unknown } | null
    return isRidgeStrengthIndex(saved?.strengthIndex) ? saved.strengthIndex : question.initialStrengthIndex
  })
  const [feedback, setFeedback] = useState('')
  const fit = ridgeFits[index]
  return <QuestionFrame question={question} state={state} questionNumber={questionNumber} totalQuestions={totalQuestions}
    hints={hints} feedback={feedback} controls={<button type="button" className="button" onClick={() => {
      const answer = { strengthIndex: index, strength: fit.strength, trainingMSE: fit.trainingMSE, validationMSE: fit.validationMSE }
      const result = question.validator(answer)
      setFeedback(result.message ?? '')
      onAttempt(result.correct?'correct':'incorrect', answer)
    }}>Check penalty strength</button>}>
    <div className="visual-grid">
      <section className="visual-panel">
        <h3 className="panel-title">Same degree, different penalty</h3>
        <RidgeModelPlot weights={fit.weights} />
        <p className="panel-note">The fitted degree-8 curve uses training circles only. Validation crosses are held out from coefficient fitting. Both datasets and the axis scales stay fixed.</p>
      </section>
      <section className="projection-panel">
        <h3 className="panel-title">Choose by validation MSE</h3>
        <label className="slider-shell"><span>L2 strength λ: {fit.strength}</span>
          <input type="range" className="range-input" min="0" max={ridgeStrengths.length-1} step="1" value={index} aria-label="L2 strength" onChange={e => { setIndex(Number(e.target.value));setFeedback('') }} />
          <span className="panel-note">Settings: {ridgeStrengths.join(' · ')} (uneven numerical spacing)</span>
        </label>
        <div className="status-panel" aria-live="polite" aria-atomic="true">
          <p><output aria-label="Training MSE">Training MSE: {fit.trainingMSE.toFixed(4)}</output></p>
          <p><output aria-label="Validation MSE" style={{fontWeight:700,fontSize:'1.2rem'}}>Validation MSE: {fit.validationMSE.toFixed(4)}</output></p>
        </div>
        <p className="panel-note">The readouts are prediction errors only: they do not include the coefficient penalty. All models use features (x/2), (x/2)², …, (x/2)⁸, plus an unpenalized intercept.</p>
        <p className="panel-note">Each setting shows its fully fitted model, not an intermediate gradient-descent step. Degree and feature scaling are held fixed while you tune λ.</p>
      </section>
    </div>
  </QuestionFrame>
}
