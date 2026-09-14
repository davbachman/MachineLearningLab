import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { AttemptOutcome, PlaneFitQuestionSpec, QuestionState, Vec3 } from '../../types'
import { planeResidualSumOfSquares, type PlaneCoefficients } from '../../lib/linearRegression'
import { QuestionFrame } from '../QuestionFrame'

interface Props {
  question: PlaneFitQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
}

// Three.js cameras are mutable scene objects, not React state.
/* eslint-disable react-hooks/immutability */
function ViewControls({ reset }: { reset: number }) {
  const { camera, gl } = useThree()
  const controlsRef = useRef<OrbitControls | null>(null)
  useEffect(() => {
    const controls = new OrbitControls(camera, gl.domElement)
    controlsRef.current = controls
    camera.position.set(9, 7, 10)
    if (camera instanceof THREE.OrthographicCamera) {
      camera.zoom = 38
      camera.updateProjectionMatrix()
    }
    controls.target.set(0, 1, 0)
    controls.enablePan = false
    controls.enableDamping = true
    controls.minZoom = 15
    controls.maxZoom = 100
    controls.update()
    return () => { controls.dispose(); controlsRef.current = null }
  }, [camera, gl, reset])
  useFrame(() => controlsRef.current?.update())
  return null
}
/* eslint-enable react-hooks/immutability */

function Segment({ start, end, color }: { start: Vec3; end: Vec3; color: string }) {
  return <line><bufferGeometry>
    <bufferAttribute attach="attributes-position" args={[new Float32Array([...start, ...end]), 3]} />
  </bufferGeometry><lineBasicMaterial color={color} /></line>
}

function AxisLabel({ label, position, color }: { label: string; position: Vec3; color: string }) {
  const texture = useMemo(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 128
    canvas.height = 64
    const context = canvas.getContext('2d')!
    context.font = 'bold 40px sans-serif'
    context.fillStyle = color
    context.textAlign = 'center'
    context.fillText(label, 64, 46)
    return new THREE.CanvasTexture(canvas)
  }, [label, color])
  useEffect(() => () => texture.dispose(), [texture])
  return <sprite position={position} scale={[1, 0.5, 1]}>
    <spriteMaterial map={texture} depthTest={false} />
  </sprite>
}

function PlaneScene({ points, fit }: { points: Vec3[]; fit: PlaneCoefficients }) {
  // Stored observations are [x1, x2, y]. World Y is the vertical response axis.
  const predict = (x1: number, x2: number) => fit.slope1 * x1 + fit.slope2 * x2 + fit.intercept
  const corner = (x1: number, x2: number): Vec3 => [x1, predict(x1, x2), x2]
  const corners = [corner(-3, -3), corner(3, -3), corner(3, 3), corner(-3, 3)]
  return <>
    <ambientLight intensity={1.5} />
    <directionalLight position={[5, 8, 6]} intensity={2} />
    <mesh>
      <bufferGeometry><bufferAttribute attach="attributes-position"
        args={[new Float32Array([...corners[0], ...corners[1], ...corners[2], ...corners[0], ...corners[2], ...corners[3]]), 3]} /></bufferGeometry>
      <meshBasicMaterial color="#cc7c31" transparent opacity={0.25} side={THREE.DoubleSide} depthWrite={false} />
    </mesh>
    {[-3, -2, -1, 0, 1, 2, 3].map((v) => <group key={v}>
      <Segment start={corner(v, -3)} end={corner(v, 3)} color="#c48c58" />
      <Segment start={corner(-3, v)} end={corner(3, v)} color="#c48c58" />
    </group>)}
    <Segment start={[-3.6, 0, 0]} end={[3.6, 0, 0]} color="#a7392f" />
    <Segment start={[0, 0, -3.6]} end={[0, 0, 3.6]} color="#287448" />
    <Segment start={[0, -3, 0]} end={[0, 5.3, 0]} color="#285aa0" />
    <AxisLabel label="x₁" position={[3.9, 0, 0]} color="#a7392f" />
    <AxisLabel label="x₂" position={[0, 0, 3.9]} color="#287448" />
    <AxisLabel label="y" position={[0, 5.6, 0]} color="#285aa0" />
    {points.map(([x1, x2, y], index) => {
      const predicted = predict(x1, x2)
      return <group key={index}>
        <mesh position={[x1, y, x2]}><sphereGeometry args={[0.095, 16, 16]} /><meshStandardMaterial color="#1c7b8a" /></mesh>
        {Math.abs(y - predicted) > 1e-6 && <mesh position={[x1, (y + predicted) / 2, x2]}>
          <cylinderGeometry args={[0.018, 0.018, Math.abs(y - predicted), 8]} /><meshBasicMaterial color="#77326b" />
        </mesh>}
        <mesh position={[x1, predicted, x2]}><sphereGeometry args={[0.04, 10, 10]} /><meshBasicMaterial color="#77326b" /></mesh>
      </group>
    })}
  </>
}

export function PlaneFitQuestion({ question, state, questionNumber, totalQuestions, hints, onAttempt }: Props) {
  const [fit, setFit] = useState<PlaneCoefficients>(() => {
    const saved = state.latestAnswer as Partial<PlaneCoefficients> | null
    return saved && typeof saved.slope1 === 'number' && Number.isFinite(saved.slope1) &&
      typeof saved.slope2 === 'number' && Number.isFinite(saved.slope2) &&
      typeof saved.intercept === 'number' && Number.isFinite(saved.intercept)
      ? { slope1: saved.slope1, slope2: saved.slope2, intercept: saved.intercept }
      : question.initialCoefficients
  })
  const [feedback, setFeedback] = useState('')
  const [viewReset, setViewReset] = useState(0)
  const rss = planeResidualSumOfSquares(question.points, fit)
  const signed = (value: number) => `${value < 0 ? '−' : '+'} ${Math.abs(value).toFixed(2)}`
  return <QuestionFrame question={question} state={state} questionNumber={questionNumber}
    totalQuestions={totalQuestions} hints={hints} feedback={feedback}
    controls={<button className="button" type="button" onClick={() => {
      const result = question.validator(fit)
      setFeedback(result.message ?? '')
      onAttempt(result.correct ? 'correct' : 'incorrect', { ...fit, rss })
    }}>Check plane</button>}>
    <div className="visual-grid">
      <section className="visual-panel">
        <h3 className="panel-title">Fit a plane to the observations</h3>
        <div style={{ height: 480, touchAction: 'none' }} role="img" aria-label="Rotatable 3D observations, regression plane, and residuals parallel to the y axis">
          <Canvas orthographic camera={{ position: [9, 7, 10], zoom: 38, near: 0.1, far: 100 }}
            fallback={<p>This question requires a browser with WebGL enabled.</p>}>
            <PlaneScene points={question.points} fit={fit} />
            <ViewControls reset={viewReset} />
          </Canvas>
        </div>
        <button type="button" className="button-ghost" onClick={() => setViewReset((v) => v + 1)}>Reset view</button>
        <p className="panel-note">Teal dots: observations. Orange plane: predictions. Purple segments: residuals parallel to y.</p>
      </section>
      <section className="projection-panel">
        <h3 className="panel-title">Adjust the plane</h3>
        <div className="status-panel">
          <p className="reveal-copy">y = {fit.slope1.toFixed(2)}x₁ {signed(fit.slope2)}x₂ {signed(fit.intercept)}</p>
          <p><output aria-label="Residual sum of squares" style={{ fontSize: '1.6rem', fontWeight: 700 }}>RSS = {rss.toFixed(3)}</output></p>
          <p className="panel-note">RSS = Σ(y − predicted y)². Smaller is better.</p>
        </div>
        {(['slope1', 'slope2', 'intercept'] as const).map((parameter) => {
          const label = parameter === 'slope1' ? 'Slope a (x₁)' : parameter === 'slope2' ? 'Slope b (x₂)' : 'Intercept c'
          return <label className="slider-shell" key={parameter}>
            <span>{label}: {fit[parameter].toFixed(2)}</span>
            <input className="range-input" type="range" aria-label={label}
              min={parameter === 'intercept' ? -1 : -1.5} max={parameter === 'intercept' ? 3 : 1.5} step="0.01" value={fit[parameter]}
              onChange={(event) => { setFit({ ...fit, [parameter]: Number(event.target.value) }); setFeedback('') }} />
          </label>
        })}
        <p className="panel-note">Drag to rotate the view; scroll or pinch to zoom. Only the sliders change the plane and its RSS.</p>
      </section>
    </div>
  </QuestionFrame>
}
