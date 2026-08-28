import { Canvas } from '@react-three/fiber'
import { useEffect, useState } from 'react'
import * as THREE from 'three'
import { QuestionFrame } from '../QuestionFrame'
import type { AttemptOutcome, Pca3DPlaneQuestionSpec, QuestionState, Vec2, Vec3 } from '../../types'
import { pca3dDatasets } from '../../data/pcaDatasets'
import {
  degrees,
  directionFromSpherical,
  explainedVarianceForPlane,
  maxAbsCoordinate2,
  orthonormalBasisAround,
  projectPointOntoPlaneBasis,
  radians,
  rollForSecondDirection,
  signInvariantAngle3,
  sphericalFromDirection,
} from '../../lib/pcaMath'
import { DEFAULT_HINT_SCHEDULE } from '../../lib/assignmentState'

interface Pca3DQuestionProps {
  question: Pca3DPlaneQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
}

function PlaneMesh({ first, second, color, opacity }: { first: Vec3; second: Vec3; color: string; opacity: number }) {
  const size = 4.6
  const corners = [
    [
      -size * first[0] - size * second[0],
      -size * first[1] - size * second[1],
      -size * first[2] - size * second[2],
    ],
    [
      size * first[0] - size * second[0],
      size * first[1] - size * second[1],
      size * first[2] - size * second[2],
    ],
    [
      size * first[0] + size * second[0],
      size * first[1] + size * second[1],
      size * first[2] + size * second[2],
    ],
    [
      -size * first[0] + size * second[0],
      -size * first[1] + size * second[1],
      -size * first[2] + size * second[2],
    ],
  ] as const

  const fillVertices = new Float32Array([
    ...corners[0],
    ...corners[1],
    ...corners[2],
    ...corners[0],
    ...corners[2],
    ...corners[3],
  ])
  const outlineVertices = new Float32Array([
    ...corners[0],
    ...corners[1],
    ...corners[2],
    ...corners[3],
  ])

  return (
    <>
      <mesh>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[fillVertices, 3]} />
        </bufferGeometry>
        <meshStandardMaterial color={color} opacity={opacity} transparent side={THREE.DoubleSide} />
      </mesh>
      <lineLoop>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[outlineVertices, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color={color} />
      </lineLoop>
    </>
  )
}

function AxisLine({ direction, color, length }: { direction: Vec3; color: string; length: number }) {
  const vertices = new Float32Array([0, 0, 0, direction[0] * length, direction[1] * length, direction[2] * length])
  return (
    <line>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[vertices, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color={color} linewidth={2} />
    </line>
  )
}

function PointCloudScene({
  points,
  first,
  second,
  answerFirst,
  answerSecond,
  showAnswer,
}: {
  points: Vec3[]
  first: Vec3
  second: Vec3
  answerFirst: Vec3
  answerSecond: Vec3
  showAnswer: boolean
}) {
  return (
    <>
      <ambientLight intensity={0.78} />
      <directionalLight intensity={1.05} position={[6, 7, 5]} />
      <group>
        {points.map((point, index) => (
          <mesh key={`point-${index}`} position={point}>
            <sphereGeometry args={[0.12, 18, 18]} />
            <meshStandardMaterial color="#1d2f37" />
          </mesh>
        ))}
      </group>
      <PlaneMesh first={first} second={second} color="#1c7b8a" opacity={0.18} />
      <AxisLine direction={first} color="#c95e1d" length={5.2} />
      <AxisLine direction={second} color="#2f7f5a" length={4.3} />
      {showAnswer ? <PlaneMesh first={answerFirst} second={answerSecond} color="#cc7c31" opacity={0.1} /> : null}
      {showAnswer ? <AxisLine direction={answerFirst} color="#cc7c31" length={5.2} /> : null}
    </>
  )
}

export function Pca3DQuestion({
  question,
  state,
  questionNumber,
  totalQuestions,
  hints,
  onAttempt,
}: Pca3DQuestionProps) {
  const dataset = pca3dDatasets[question.datasetId]
  const [azimuth, setAzimuth] = useState(radians(question.initialAzimuthDeg))
  const [elevation, setElevation] = useState(radians(question.initialElevationDeg))
  const [roll, setRoll] = useState(radians(question.initialRollDeg))
  const [feedback, setFeedback] = useState('')
  const [dragging, setDragging] = useState(false)
  const showAnswer = state.status === 'correct' || state.status === 'gave_up'
  const answerAngles = sphericalFromDirection(dataset.answerFirst)
  const answerRoll = rollForSecondDirection(dataset.answerFirst, dataset.answerSecond)
  const displayAzimuth = azimuth
  const displayElevation = elevation
  const displayRoll = roll
  const first = directionFromSpherical(displayAzimuth, displayElevation)
  const second = orthonormalBasisAround(first, displayRoll).second

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'ArrowLeft') {
        event.preventDefault()
        setRoll((current) => current - radians(7))
      }

      if (event.key === 'ArrowRight') {
        event.preventDefault()
        setRoll((current) => current + radians(7))
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const updateFromDrag = (movementX: number, movementY: number) => {
    setAzimuth((current) => current + movementX * 0.012)
    setElevation((current) => {
      const next = current - movementY * 0.01
      return Math.max(radians(-78), Math.min(radians(78), next))
    })
  }

  const checkPlane = () => {
    const result = question.validator({ first, second })

    if (result.correct) {
      setAzimuth(answerAngles.azimuth)
      setElevation(answerAngles.elevation)
      setRoll(answerRoll)
      setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
      onAttempt('correct', { first, second })
      return
    }

    const schedule = question.hintSchedule ?? DEFAULT_HINT_SCHEDULE
    const nextIncorrectAttempts = state.incorrectAttempts + 1
    setFeedback(
      schedule.includes(nextIncorrectAttempts)
        ? 'Not yet. A new hint appeared below.'
        : result.message ?? 'The plane can still capture more variance.',
    )
    onAttempt('incorrect', { first, second })
  }

  const projectedPoints = dataset.points.map((point) => projectPointOntoPlaneBasis(point, first, second))
  const domain = Math.max(
    4.2,
    maxAbsCoordinate2(projectedPoints.map((point) => [point[0], point[1]] as Vec2)),
  ) + 0.5
  const viewSize = 360
  const toSvgX = (value: number) => (value / domain) * 140 + viewSize / 2
  const toSvgY = (value: number) => viewSize / 2 - (value / domain) * 140
  const retainedVariance =
    explainedVarianceForPlane(dataset.points, first, second) /
    explainedVarianceForPlane(dataset.points, dataset.answerFirst, dataset.answerSecond)
  const firstDirectionError = degrees(signInvariantAngle3(first, dataset.answerFirst))

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
          <button type="button" className="button" onClick={checkPlane}>
            Check this plane
          </button>
          <span className="pill">Variance kept: {(retainedVariance * 100).toFixed(1)}%</span>
          <span className="pill">PC1 error: {firstDirectionError.toFixed(1)}°</span>
          <span className="pill">Roll: {degrees(displayRoll).toFixed(1)}°</span>
        </>
      }
    >
      <div className="visual-grid">
        <section className="visual-panel">
          <h3 className="panel-title">3D cloud and candidate plane</h3>
          <p className="panel-note">
            Drag to rotate the red first direction. Use the left and right arrow keys to roll the green
            second direction around it.
          </p>
          <div className="plane-stage-shell">
            <div
              className="plane-canvas"
              onPointerDown={() => setDragging(true)}
              onPointerMove={(event) => {
                if (dragging) {
                  updateFromDrag(event.movementX, event.movementY)
                }
              }}
              onPointerUp={() => setDragging(false)}
              onPointerLeave={() => setDragging(false)}
            >
              <Canvas camera={{ position: [6.3, 5.6, 6.5], fov: 42 }}>
                <PointCloudScene
                  points={dataset.points}
                  first={first}
                  second={second}
                  answerFirst={dataset.answerFirst}
                  answerSecond={dataset.answerSecond}
                  showAnswer={showAnswer}
                />
              </Canvas>
            </div>
            <div className="plane-overlay" />
            <div className="plane-guides">
              <p className="stage-caption">Controls</p>
              <span>Drag: rotate PC1</span>
              <span>Left / Right: roll PC2</span>
            </div>
          </div>

          <div className="plane-legend">
            <span className="legend-chip">
              <span className="legend-swatch" style={{ background: '#c95e1d' }} />
              <span className="legend-label">First principal direction</span>
            </span>
            <span className="legend-chip">
              <span className="legend-swatch" style={{ background: '#2f7f5a' }} />
              <span className="legend-label">Second principal direction</span>
            </span>
            <span className="legend-chip">
              <span className="legend-swatch" style={{ background: '#1c7b8a' }} />
              <span className="legend-label">Current plane</span>
            </span>
          </div>
        </section>

        <section className="projection-panel">
          <h3 className="panel-title">Orthogonal view of the projection plane</h3>
          <p className="panel-note">
            This view removes the third dimension so you can judge how much spread the plane keeps.
          </p>
          <svg className="plane-right-stage" viewBox={`0 0 ${viewSize} ${viewSize}`} role="img" aria-label="Projected points in the PCA plane">
            <line x1="24" y1={viewSize / 2} x2={viewSize - 24} y2={viewSize / 2} stroke="#ddd3c4" strokeWidth="1.6" />
            <line x1={viewSize / 2} y1="24" x2={viewSize / 2} y2={viewSize - 24} stroke="#ddd3c4" strokeWidth="1.6" />
            {projectedPoints.map((point, index) => (
              <circle
                key={`${question.id}-plane-point-${index}`}
                cx={toSvgX(point[0])}
                cy={toSvgY(point[1])}
                r="6.2"
                fill="#1c7b8a"
              />
            ))}
          </svg>
        </section>
      </div>
    </QuestionFrame>
  )
}
