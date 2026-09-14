import { useId, useRef, useState } from 'react'
import type { AttemptOutcome, LinearFitQuestionSpec, QuestionState } from '../../types'
import { residualSumOfSquares } from '../../lib/linearRegression'
import { QuestionFrame } from '../QuestionFrame'

interface Props {
  question: LinearFitQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
}

const SIZE = 480
const PAD = 42
const toX = (x: number) => PAD + (x + 5) / 10 * (SIZE - 2 * PAD)
const toY = (y: number) => SIZE - PAD - (y + 7) / 20 * (SIZE - 2 * PAD)
const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value))

export function LinearFitQuestion({ question, state, questionNumber, totalQuestions, hints, onAttempt }: Props) {
  const clipId = useId()
  const svgRef = useRef<SVGSVGElement>(null)
  const drag = useRef<{ x: number; otherY: number } | null>(null)
  const [fit, setFit] = useState(() => {
    const saved = state.latestAnswer as { slope?: unknown; intercept?: unknown } | null
    return saved && typeof saved.slope === 'number' && Number.isFinite(saved.slope) &&
      typeof saved.intercept === 'number' && Number.isFinite(saved.intercept)
      ? { slope: saved.slope, intercept: saved.intercept }
      : { slope: question.initialSlope, intercept: question.initialIntercept }
  })
  const [feedback, setFeedback] = useState('')
  const rss = residualSumOfSquares(question.points, fit.slope, fit.intercept)
  const prediction = (x: number) => fit.slope * x + fit.intercept
  const moveHandle = (x: number, y: number, otherY: number) => {
    const slope = clamp((y - otherY) / (2 * x), -2, 3)
    const intercept = clamp(otherY + slope * x, -2, 6)
    setFit({ slope, intercept })
    setFeedback('')
  }

  return (
    <QuestionFrame question={question} state={state} questionNumber={questionNumber}
      totalQuestions={totalQuestions} hints={hints} feedback={feedback}
      controls={<button className="button" type="button" onClick={() => {
        const result = question.validator(fit)
        setFeedback(result.message ?? '')
        onAttempt(result.correct ? 'correct' : 'incorrect', { ...fit, rss })
      }}>Check line</button>}
    >
      <div className="visual-grid">
        <section className="visual-panel">
          <h3 className="panel-title">Fit a line to the observations</h3>
          <svg ref={svgRef} className="svg-stage" viewBox={`0 0 ${SIZE} ${SIZE}`}
            style={{ touchAction: 'none' }} role="img" aria-label="Observations, adjustable regression line, and vertical residuals"
            onPointerMove={(event) => {
              if (!drag.current || !svgRef.current) return
              const rect = svgRef.current.getBoundingClientRect()
              const stageY = (event.clientY - rect.top) / rect.height * SIZE
              const y = (SIZE - PAD - stageY) / (SIZE - 2 * PAD) * 20 - 7
              moveHandle(drag.current.x, y, drag.current.otherY)
            }}
            onPointerUp={() => { drag.current = null }}
            onPointerCancel={() => { drag.current = null }}
          >
            <defs><clipPath id={clipId}><rect x={PAD} y={PAD} width={SIZE - 2 * PAD} height={SIZE - 2 * PAD} /></clipPath></defs>
            {[-4, -2, 0, 2, 4].map((x) => <g key={`x-${x}`}>
              <line x1={toX(x)} x2={toX(x)} y1={PAD} y2={SIZE - PAD} stroke="#ded9cf" />
              <text x={toX(x)} y={SIZE - PAD + 20} textAnchor="middle" fontSize="12">{x}</text>
            </g>)}
            {[-6, -3, 0, 3, 6, 9, 12].map((y) => <g key={`y-${y}`}>
              <line x1={PAD} x2={SIZE - PAD} y1={toY(y)} y2={toY(y)} stroke="#ded9cf" />
              <text x={PAD - 10} y={toY(y) + 4} textAnchor="end" fontSize="12">{y}</text>
            </g>)}
            <text x={SIZE / 2} y={SIZE - 6} textAnchor="middle" fontSize="15">x</text>
            <text x="12" y={PAD - 10} fontSize="15">y</text>
            <g clipPath={`url(#${clipId})`}>
              <line x1={toX(-5)} y1={toY(prediction(-5))} x2={toX(5)} y2={toY(prediction(5))} stroke="#cc7c31" strokeWidth="3" />
              {question.points.map(([x, y], index) => <g key={index}>
                <line x1={toX(x)} x2={toX(x)} y1={toY(y)} y2={toY(prediction(x))} stroke="#77326b" strokeWidth="2.5" />
                <line x1={toX(x) - 4} x2={toX(x) + 4} y1={toY(prediction(x))} y2={toY(prediction(x))} stroke="#77326b" strokeWidth="2" />
                <circle cx={toX(x)} cy={toY(y)} r="6" fill="#1c7b8a" stroke="#102027" />
              </g>)}
            </g>
            {[-2, 2].map((x) => <circle key={x}
              cx={toX(x)} cy={toY(prediction(x))} r="10" fill="white" stroke="#cc7c31" strokeWidth="3"
              role="button" tabIndex={0} aria-label={`${x < 0 ? 'Left' : 'Right'} line handle: drag or use up and down arrows`}
              style={{ cursor: 'ns-resize' }}
              onPointerDown={(event) => {
                event.preventDefault()
                drag.current = { x, otherY: prediction(-x) }
                event.currentTarget.setPointerCapture(event.pointerId)
              }}
              onKeyDown={(event) => {
                if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return
                event.preventDefault()
                moveHandle(x, prediction(x) + (event.key === 'ArrowUp' ? 0.1 : -0.1), prediction(-x))
              }}
            />)}
          </svg>
          <p className="panel-note">Teal dots: observations. Orange line: predictions. Purple vertical segments: residuals.</p>
        </section>
        <section className="projection-panel">
          <h3 className="panel-title">Adjust the line</h3>
          <div className="status-panel">
            <p className="reveal-copy">y = {fit.slope.toFixed(2)}x {fit.intercept < 0 ? '−' : '+'} {Math.abs(fit.intercept).toFixed(2)}</p>
            <p><output aria-label="Residual sum of squares" style={{ fontSize: '1.6rem', fontWeight: 700 }}>RSS = {rss.toFixed(3)}</output></p>
            <p className="panel-note">RSS = Σ(y − predicted y)². Smaller is better.</p>
          </div>
          {(['slope', 'intercept'] as const).map((parameter) => <label className="slider-shell" key={parameter}>
            <span>{parameter === 'slope' ? 'Slope' : 'Intercept'}: {fit[parameter].toFixed(2)}</span>
            <input className="range-input" type="range" aria-label={parameter === 'slope' ? 'Slope' : 'Intercept'}
              min="-2" max={parameter === 'slope' ? 3 : 6} step="0.01" value={fit[parameter]}
              onChange={(event) => {
                setFit({ ...fit, [parameter]: Number(event.target.value) })
                setFeedback('')
              }} />
          </label>)}
          <p className="panel-note">Adjust both parameters until you cannot lower RSS much further. Then check your line.</p>
        </section>
      </div>
    </QuestionFrame>
  )
}
