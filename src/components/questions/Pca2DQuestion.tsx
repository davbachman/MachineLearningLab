import { useRef, useState } from 'react'
import { QuestionFrame } from '../QuestionFrame'
import type { AttemptOutcome, Pca2DLineQuestionSpec, QuestionState, Vec2 } from '../../types'
import { pca2dDatasets } from '../../data/pcaDatasets'
import {
  degrees,
  explainedVarianceForLine,
  maxAbsCoordinate2,
  projectPointOntoLine2D,
  projectScalarOntoLine2D,
  radians,
} from '../../lib/pcaMath'
import { DEFAULT_HINT_SCHEDULE } from '../../lib/assignmentState'

interface Pca2DQuestionProps {
  question: Pca2DLineQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
  onGiveUp: (answer: unknown) => void
}

const VIEW_SIZE = 360
const VIEW_PADDING = 32

function axisLine(domain: number, direction: Vec2) {
  return {
    x1: -direction[0] * domain,
    y1: -direction[1] * domain,
    x2: direction[0] * domain,
    y2: direction[1] * domain,
  }
}

export function Pca2DQuestion({
  question,
  state,
  questionNumber,
  totalQuestions,
  hints,
  onAttempt,
  onGiveUp,
}: Pca2DQuestionProps) {
  const dataset = pca2dDatasets[question.datasetId]
  const svgRef = useRef<SVGSVGElement | null>(null)
  const [angle, setAngle] = useState(radians(question.initialAngleDeg))
  const [feedback, setFeedback] = useState('')
  const [dragging, setDragging] = useState(false)
  const resolved = state.status === 'correct' || state.status === 'gave_up'

  const answerAngle = Math.atan2(dataset.answerDirection[1], dataset.answerDirection[0])
  const displayAngle = resolved ? answerAngle : angle
  const direction: Vec2 = [Math.cos(displayAngle), Math.sin(displayAngle)]
  const projectedPoints = dataset.points.map((point) => projectPointOntoLine2D(point, direction))
  const projectedScalars = dataset.points.map((point) => projectScalarOntoLine2D(point, direction))
  const optimalScore = explainedVarianceForLine(
    dataset.points,
    dataset.answerDirection,
    dataset.scoring,
  )
  const chosenScore = explainedVarianceForLine(dataset.points, direction, dataset.scoring)
  const scoreRatio = chosenScore / optimalScore
  const domain = Math.max(
    5.5,
    maxAbsCoordinate2([...dataset.points, ...projectedPoints, dataset.answerDirection]),
  ) + 0.75
  const lineExtent = domain * 1.18
  const projectionDomain =
    Math.max(...dataset.points.map((point) => Math.hypot(point[0], point[1])), 1) * 1.08
  const scoreLabel =
    dataset.scoring === 'secondMoment'
      ? 'Uncentered PCA score'
      : 'Projected variance'
  const projectionNote =
    dataset.scoring === 'secondMoment'
      ? 'This view keeps the origin fixed. In the uncentered case, the best line pushes the projections far from the origin on average, even if they stay tightly clustered.'
      : ''

  const toSvgX = (value: number) => (value / domain) * ((VIEW_SIZE - VIEW_PADDING * 2) / 2) + VIEW_SIZE / 2
  const toSvgY = (value: number) => VIEW_SIZE / 2 - (value / domain) * ((VIEW_SIZE - VIEW_PADDING * 2) / 2)

  const updateAngleFromEvent = (clientX: number, clientY: number) => {
    if (!svgRef.current || resolved) {
      return
    }

    const rect = svgRef.current.getBoundingClientRect()
    const normalizedX = ((clientX - rect.left) / rect.width - 0.5) * 2
    const normalizedY = (0.5 - (clientY - rect.top) / rect.height) * 2
    setAngle(Math.atan2(normalizedY, normalizedX))
  }

  const handlePointerDown = (event: React.PointerEvent<SVGSVGElement>) => {
    if (resolved) {
      return
    }

    event.currentTarget.setPointerCapture(event.pointerId)
    setDragging(true)
    updateAngleFromEvent(event.clientX, event.clientY)
  }

  const handlePointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!dragging) {
      return
    }

    updateAngleFromEvent(event.clientX, event.clientY)
  }

  const handlePointerUp = (event: React.PointerEvent<SVGSVGElement>) => {
    if (dragging) {
      event.currentTarget.releasePointerCapture(event.pointerId)
      setDragging(false)
    }
  }

  const currentSchedule = question.hintSchedule ?? DEFAULT_HINT_SCHEDULE

  const checkDirection = () => {
    const result = question.validator({ direction })

    if (result.correct) {
      setAngle(answerAngle)
      setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
      onAttempt('correct', { direction })
      return
    }

    const nextIncorrectAttempts = state.incorrectAttempts + 1
    const unlockedHint = currentSchedule.includes(nextIncorrectAttempts)
    setFeedback(
      unlockedHint
        ? 'Not yet. A new hint appeared below.'
        : result.message ??
            'That direction is still leaving variance on the table. Keep rotating.',
    )
    onAttempt('incorrect', { direction })
  }

  const revealAnswer = () => {
    setAngle(answerAngle)
    setFeedback('Answer revealed. Compare the highlighted answer with your last guess.')
    onGiveUp({ direction })
  }

  const projectionX = (value: number) =>
    VIEW_SIZE / 2 + (value / projectionDomain) * ((VIEW_SIZE - 68) / 2)

  const userLine = axisLine(lineExtent, direction)
  const answerLine = axisLine(lineExtent, dataset.answerDirection)
  const comparisonLine = dataset.comparisonDirection
    ? axisLine(lineExtent, dataset.comparisonDirection)
    : null

  return (
    <QuestionFrame
      question={question}
      questionNumber={questionNumber}
      totalQuestions={totalQuestions}
      state={state}
      feedback={feedback}
      hints={hints}
      controls={
        <>
          <button type="button" className="button" onClick={checkDirection} disabled={resolved}>
            Check this direction
          </button>
          <button type="button" className="button-secondary" onClick={revealAnswer} disabled={resolved}>
            Give up
          </button>
          <span className="pill">
            {scoreLabel}: {(scoreRatio * 100).toFixed(1)}%
          </span>
          <span className="pill">Angle: {degrees(displayAngle).toFixed(1)}°</span>
        </>
      }
    >
      <div className="visual-grid">
        <section className="visual-panel">
          <h3 className="panel-title">Point cloud and projection line</h3>
          <p className="panel-note">Drag anywhere on the panel to rotate the line through the origin.</p>
          <svg
            ref={svgRef}
            className="svg-stage"
            viewBox={`0 0 ${VIEW_SIZE} ${VIEW_SIZE}`}
            role="img"
            aria-label={`${dataset.label} with a draggable projection line`}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerUp}
          >
            <rect x="0" y="0" width={VIEW_SIZE} height={VIEW_SIZE} rx="18" fill="transparent" />
            <line x1="0" y1={VIEW_SIZE / 2} x2={VIEW_SIZE} y2={VIEW_SIZE / 2} stroke="#d7d0c3" strokeWidth="1.4" />
            <line x1={VIEW_SIZE / 2} y1="0" x2={VIEW_SIZE / 2} y2={VIEW_SIZE} stroke="#d7d0c3" strokeWidth="1.4" />

            {resolved ? (
              <line
                x1={toSvgX(answerLine.x1)}
                y1={toSvgY(answerLine.y1)}
                x2={toSvgX(answerLine.x2)}
                y2={toSvgY(answerLine.y2)}
                stroke="#1c7b8a"
                strokeWidth="5"
                strokeLinecap="round"
                opacity="0.9"
              />
            ) : null}

            {resolved && comparisonLine ? (
              <line
                x1={toSvgX(comparisonLine.x1)}
                y1={toSvgY(comparisonLine.y1)}
                x2={toSvgX(comparisonLine.x2)}
                y2={toSvgY(comparisonLine.y2)}
                stroke="#cc7c31"
                strokeWidth="4"
                strokeDasharray="10 8"
                strokeLinecap="round"
                opacity="0.74"
              />
            ) : null}

            <line
              x1={toSvgX(userLine.x1)}
              y1={toSvgY(userLine.y1)}
              x2={toSvgX(userLine.x2)}
              y2={toSvgY(userLine.y2)}
              stroke="#212f36"
              strokeWidth="3"
              strokeLinecap="round"
            />

            {dataset.points.map((point, index) => {
              const projection = projectedPoints[index]
              return (
                <g key={`${question.id}-projection-${index}`}>
                  <line
                    x1={toSvgX(point[0])}
                    y1={toSvgY(point[1])}
                    x2={toSvgX(projection[0])}
                    y2={toSvgY(projection[1])}
                    stroke="rgba(111, 128, 135, 0.42)"
                    strokeWidth="1.4"
                  />
                  <circle cx={toSvgX(projection[0])} cy={toSvgY(projection[1])} r="4.2" fill="#cc7c31" />
                  <circle cx={toSvgX(point[0])} cy={toSvgY(point[1])} r="5.1" fill="#21313b" opacity="0.95" />
                </g>
              )
            })}

            <circle
              cx={toSvgX(direction[0] * lineExtent * 0.85)}
              cy={toSvgY(direction[1] * lineExtent * 0.85)}
              r="8"
              fill="#f6b26b"
              stroke="#7a4c21"
              strokeWidth="2"
            />
          </svg>

          <div className="legend-row">
            <span className="legend-chip">
              <span className="legend-swatch" style={{ background: '#21313b' }} />
              <span className="legend-label">Original points</span>
            </span>
            <span className="legend-chip">
              <span className="legend-swatch" style={{ background: '#cc7c31' }} />
              <span className="legend-label">Projected points</span>
            </span>
            {resolved && comparisonLine ? (
              <span className="legend-chip">
                <span className="legend-swatch" style={{ background: '#cc7c31', opacity: 0.8 }} />
                <span className="legend-label">{dataset.comparisonLabel}</span>
              </span>
            ) : null}
          </div>
        </section>

        <section className="projection-panel">
          <h3 className="panel-title">Projection as a one-dimensional view</h3>
          {projectionNote ? <p className="panel-note">{projectionNote}</p> : null}
          <svg className="projection-stage" viewBox={`0 0 ${VIEW_SIZE} ${VIEW_SIZE}`} role="img" aria-label="Projected points on a horizontal line">
            <line
              x1={VIEW_SIZE / 2}
              y1="42"
              x2={VIEW_SIZE / 2}
              y2={VIEW_SIZE - 42}
              stroke="#d7d0c3"
              strokeWidth="1.6"
              strokeDasharray="8 6"
            />
            <line
              x1="28"
              y1={VIEW_SIZE / 2}
              x2={VIEW_SIZE - 28}
              y2={VIEW_SIZE / 2}
              stroke="#20313a"
              strokeWidth="3"
              strokeLinecap="round"
            />
            <text
              x={VIEW_SIZE / 2}
              y={VIEW_SIZE / 2 - 16}
              textAnchor="middle"
              fontSize="12"
              fontWeight="700"
              fill="#7b847e"
            >
              origin
            </text>
            {projectedScalars.map((value, index) => (
              <circle
                key={`${question.id}-scalar-${index}`}
                cx={projectionX(value)}
                cy={VIEW_SIZE / 2}
                r="7"
                fill="#cc7c31"
              />
            ))}
          </svg>
        </section>
      </div>
    </QuestionFrame>
  )
}
