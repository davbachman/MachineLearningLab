import { useId, useState } from 'react'
import type { AttemptOutcome, PolynomialDegreeQuestionSpec, QuestionState } from '../../types'
import { isPolynomialDegree, polynomialFit, predictPolynomial, trainingPoints, validationPoints } from '../../data/polynomialDegreeData'
import { QuestionFrame } from '../QuestionFrame'

interface Props {
  question: PolynomialDegreeQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
}

const WIDTH = 560, HEIGHT = 430, PAD = 44
const toX = (x: number) => PAD + (x + 2.2) / 4.4 * (WIDTH - 2 * PAD)
const toY = (y: number) => HEIGHT - PAD - y / 5 * (HEIGHT - 2 * PAD)

export function PolynomialDegreeQuestion({ question, state, questionNumber, totalQuestions, hints, onAttempt }: Props) {
  const clipId = useId()
  const [degree, setDegree] = useState(() => {
    const saved = state.latestAnswer as { degree?: unknown } | null
    return isPolynomialDegree(saved?.degree) ? saved.degree : question.initialDegree
  })
  const [feedback, setFeedback] = useState('')
  const fit = polynomialFit(degree)
  const path = Array.from({ length: 401 }, (_, i) => {
    const x = -2 + i / 100
    return `${i ? 'L' : 'M'}${toX(x)},${toY(predictPolynomial(fit.coefficients, x))}`
  }).join(' ')

  return <QuestionFrame question={question} state={state} questionNumber={questionNumber}
    totalQuestions={totalQuestions} hints={hints} feedback={feedback}
    controls={<button className="button" type="button" onClick={() => {
      const answer = { degree, trainingMse: fit.trainingMse, validationMse: fit.validationMse }
      const result = question.validator(answer)
      setFeedback(result.message ?? '')
      onAttempt(result.correct ? 'correct' : 'incorrect', answer)
    }}>Check degree</button>}>
    <div className="visual-grid">
      <section className="visual-panel">
        <h3 className="panel-title">Degree {degree} polynomial</h3>
        <svg className="svg-stage" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img"
          aria-label={`Degree ${degree} fitted curve, training circles, and validation crosses`}>
          <defs><clipPath id={clipId}><rect x={PAD} y={PAD} width={WIDTH - 2 * PAD} height={HEIGHT - 2 * PAD} /></clipPath></defs>
          {[-2, -1, 0, 1, 2].map(x => <g key={`x${x}`}>
            <line x1={toX(x)} x2={toX(x)} y1={PAD} y2={HEIGHT - PAD} stroke="#ded9cf" />
            <text x={toX(x)} y={HEIGHT - PAD + 22} textAnchor="middle" fontSize="13">{x}</text>
          </g>)}
          {[0, 1, 2, 3, 4, 5].map(y => <g key={`y${y}`}>
            <line x1={PAD} x2={WIDTH - PAD} y1={toY(y)} y2={toY(y)} stroke="#ded9cf" />
            <text x={PAD - 12} y={toY(y) + 4} textAnchor="end" fontSize="13">{y}</text>
          </g>)}
          <text x={WIDTH / 2} y={HEIGHT - 5} textAnchor="middle">x</text>
          <text x="12" y="30">y</text>
          <g clipPath={`url(#${clipId})`}>
            <path d={path} stroke="#77326b" strokeWidth="2.5" fill="none" />
            {trainingPoints.map(([x, y], i) => <circle key={`t${i}`} cx={toX(x)} cy={toY(y)} r="5" fill="#1c7b8a" stroke="white" strokeWidth="1.5" />)}
            {validationPoints.map(([x, y], i) => <path key={`v${i}`} d={`M${toX(x) - 5},${toY(y) - 5}l10,10m0,-10l-10,10`} stroke="#bd5e16" strokeWidth="2.5" />)}
          </g>
        </svg>
        <p className="panel-note">Teal circles: training data. Orange crosses: validation data. Purple curve: predictions.</p>
      </section>
      <section className="projection-panel">
        <h3 className="panel-title">Goal: minimize {question.target} MSE</h3>
        <label className="slider-shell">
          <span>Polynomial degree: {degree}</span>
          <input className="range-input" type="range" aria-label="Polynomial degree" min="1" max="8" step="1" value={degree}
            onChange={event => { setDegree(Number(event.target.value)); setFeedback('') }} />
          <span className="panel-note">1 (straight line) — 8 (most flexible)</span>
        </label>
        <div className="status-panel" aria-live="polite" aria-atomic="true">
          <p><output aria-label="Training MSE" style={{ fontSize: '1.3rem', fontWeight: 700 }}>Training MSE = {fit.trainingMse.toFixed(4)}</output></p>
          <p><output aria-label="Validation MSE" style={{ fontSize: '1.3rem', fontWeight: 700 }}>Validation MSE = {fit.validationMse.toFixed(4)}</output></p>
        </div>
        <p className="panel-note">MSE is the average squared vertical distance from a point to the curve, computed separately for each dataset. Smaller is better.</p>
        <p className="panel-note">Only training points determine the fitted curve. Validation points measure how well it predicts held-out observations.</p>
      </section>
    </div>
  </QuestionFrame>
}
