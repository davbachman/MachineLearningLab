import { useRef, useState, type MouseEvent, type PointerEvent } from 'react'
import { QuestionFrame } from '../QuestionFrame'
import type {
  AttemptOutcome,
  Kmeans2DQuestionSpec,
  QuestionState,
  Vec2,
} from '../../types'
import { kmeansInteractiveDatasets } from '../../data/kmeansDatasets'
import { DEFAULT_HINT_SCHEDULE } from '../../lib/assignmentState'
import {
  assignPointsToCentroids,
  formatObjective,
  withinClusterSumOfSquares,
} from '../../lib/kmeansMath'

interface Kmeans2DQuestionProps {
  question: Kmeans2DQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
}

const STAGE_SIZE = 420
const STAGE_PADDING = 28
const REGION_STEPS = 18
const CLUSTER_COLORS = ['#1c7b8a', '#cc7c31', '#c44a3d', '#7e5da9']

function cloneCentroids(centroids: Vec2[] | undefined) {
  return (centroids ?? []).map((centroid) => [...centroid] as Vec2)
}

function baseAssignments(length: number) {
  return Array.from({ length }, () => 0)
}

function distanceSquared(a: Vec2, b: Vec2) {
  return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function computeDomain(points: Vec2[], centroids: Vec2[]) {
  const values = [...points, ...centroids]
  const xs = values.map((value) => value[0])
  const ys = values.map((value) => value[1])
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const span = Math.max(maxX - minX, maxY - minY, 1)
  const padding = span * 0.18

  return {
    minX: minX - padding,
    maxX: maxX + padding,
    minY: minY - padding,
    maxY: maxY + padding,
    span: span + padding * 2,
  }
}

function metricLabel(metric: 'euclidean' | 'manhattan') {
  return metric === 'manhattan' ? 'Manhattan' : 'Euclidean'
}

export function Kmeans2DQuestion({
  question,
  state,
  questionNumber,
  totalQuestions,
  hints,
  onAttempt,
}: Kmeans2DQuestionProps) {
  const dataset = kmeansInteractiveDatasets[question.datasetId]
  const svgRef = useRef<SVGSVGElement | null>(null)
  const dragPointerIdRef = useRef<number | null>(null)
  const [centroids, setCentroids] = useState(() =>
    cloneCentroids(dataset.initialCentroids ?? dataset.fixedCentroids ?? dataset.targetCentroids),
  )
  const [assignments, setAssignments] = useState(() => baseAssignments(dataset.points.length))
  const iterationAssignments = dataset.iterationAssignments ?? []
  const iterationCentroids = dataset.iterationCentroids ?? []
  const totalLloydIterations = Math.max(iterationAssignments.length, iterationCentroids.length, 1)
  const [stepIndex, setStepIndex] = useState(0)
  const [feedback, setFeedback] = useState('')
  const [draggingCentroidIndex, setDraggingCentroidIndex] = useState<number | null>(null)
  const showAnswer = state.status === 'correct' || state.status === 'gave_up'
  const currentSchedule = question.hintSchedule ?? DEFAULT_HINT_SCHEDULE
  const unlockedHintCount = hints.length
  const isLloydIteration = question.interactionMode === 'lloydIteration'
  const isMetricComparison = question.interactionMode === 'metricComparison'
  const currentIteration = Math.min(Math.floor(stepIndex / 2), totalLloydIterations - 1)
  const currentPhase = isLloydIteration
    ? stepIndex % 2 === 0
      ? 'assignments'
      : 'centroids'
    : isMetricComparison
      ? stepIndex === 0
        ? 'euclidean'
        : 'manhattan'
    : 'centroids'
  const displayedHints =
    isLloydIteration && question.hints.length
      ? unlockedHintCount === 0
        ? []
        : currentIteration === 0
          ? currentPhase === 'assignments'
            ? [question.hints[0]]
            : question.hints[1]
              ? [question.hints[1]]
              : [question.hints[0]]
          : question.hints[2]
            ? [question.hints[2]]
            : [question.hints[question.hints.length - 1]]
      : isMetricComparison && question.hints.length
        ? unlockedHintCount === 0
          ? []
          : currentPhase === 'euclidean'
            ? [question.hints[0]]
            : question.hints[1]
              ? [question.hints[1]]
              : [question.hints[0]]
      : hints
  const activeMetric =
    question.interactionMode === 'metricComparison'
      ? currentPhase === 'manhattan'
        ? (dataset.comparisonMetric ?? 'manhattan')
        : dataset.metric
      : dataset.metric

  const shownCentroids = cloneCentroids(dataset.fixedCentroids ?? centroids)

  const shownAssignments =
    question.interactionMode === 'assignPoints'
      ? assignments
      : question.interactionMode === 'lloydIteration'
        ? currentPhase === 'assignments'
          ? assignments
          : (iterationAssignments[currentIteration] ?? assignments)
        : question.interactionMode === 'metricComparison'
          ? assignments
        : dataset.displayAssignments ??
          (shownCentroids.length === dataset.k
            ? assignPointsToCentroids(dataset.points, shownCentroids, dataset.metric)
            : Array.from({ length: dataset.points.length }, () => -1))

  const domain = computeDomain(
    dataset.points,
    [
      ...shownCentroids,
      ...cloneCentroids(dataset.targetCentroids),
      ...cloneCentroids(dataset.comparisonCentroids),
      ...cloneCentroids(dataset.initialCentroids),
    ],
  )

  const toStageX = (value: number) =>
    STAGE_PADDING + ((value - domain.minX) / (domain.maxX - domain.minX)) * (STAGE_SIZE - STAGE_PADDING * 2)
  const toStageY = (value: number) =>
    STAGE_SIZE -
    STAGE_PADDING -
    ((value - domain.minY) / (domain.maxY - domain.minY)) * (STAGE_SIZE - STAGE_PADDING * 2)

  const fromEventToWorld = (clientX: number, clientY: number): Vec2 | null => {
    if (!svgRef.current) {
      return null
    }

    const rect = svgRef.current.getBoundingClientRect()
    if (!rect.width || !rect.height) {
      return null
    }

    const normalizedX = clamp((clientX - rect.left) / rect.width, 0, 1)
    const normalizedY = clamp((clientY - rect.top) / rect.height, 0, 1)

    return [
      domain.minX + normalizedX * (domain.maxX - domain.minX),
      domain.maxY - normalizedY * (domain.maxY - domain.minY),
    ]
  }

  const clickablePoints =
    (question.interactionMode === 'assignPoints' ||
      question.interactionMode === 'metricComparison' ||
      (question.interactionMode === 'lloydIteration' && currentPhase === 'assignments'))

  const draggableCentroids =
    (question.interactionMode === 'placeCentroids' ||
      question.interactionMode === 'updateCentroids' ||
      (question.interactionMode === 'lloydIteration' && currentPhase === 'centroids'))

  const showRegions =
    shownCentroids.length === dataset.k &&
    (dataset.regionVisibility === 'always' ||
      (dataset.regionVisibility === 'resolved' && showAnswer))

  const currentObjective =
    shownCentroids.length === dataset.k && shownAssignments.every((assignment) => assignment >= 0)
      ? withinClusterSumOfSquares(dataset.points, shownAssignments, shownCentroids)
      : null
  const invalidAttempt = (message: string, answer: unknown) => {
    const nextIncorrectAttempts = state.incorrectAttempts + 1
    setFeedback(currentSchedule.includes(nextIncorrectAttempts) ? 'Not yet. A new hint appeared below.' : message)
    onAttempt('incorrect', answer)
  }

  const checkWork = () => {
    if (question.interactionMode === 'placeCentroids' || question.interactionMode === 'updateCentroids') {
      const submission = { centroids }
      const result = question.validator(submission)
      if (!result.correct) {
        invalidAttempt(result.message ?? 'That placement is not right yet.', submission)
        return
      }

      if (dataset.targetCentroids) {
        setCentroids(cloneCentroids(dataset.targetCentroids))
      }
      setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
      onAttempt('correct', submission)
      return
    }

    if (question.interactionMode === 'assignPoints') {
      const submission = { assignments }
      const result = question.validator(submission)
      if (!result.correct) {
        invalidAttempt(result.message ?? 'At least one point is still assigned incorrectly.', submission)
        return
      }

      setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
      onAttempt('correct', submission)
      return
    }

    if (question.interactionMode === 'metricComparison') {
      const step = currentPhase === 'manhattan' ? 'manhattan' : 'euclidean'
      const submission = { step, assignments }
      const result = question.validator(submission)
      if (!result.correct) {
        invalidAttempt(result.message ?? 'At least one point is still assigned incorrectly.', submission)
        return
      }

      if (currentPhase === 'euclidean') {
        setAssignments(dataset.targetAssignments ?? assignments)
        setFeedback('Euclidean assignments confirmed. Now recolor the same points using Manhattan distance.')
        setStepIndex(1)
        onAttempt('progress', submission)
        return
      }

      setAssignments(dataset.comparisonAssignments ?? assignments)
      setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
      onAttempt('correct', submission)
      return
    }

    if (currentPhase === 'assignments') {
      const submission = {
        step: 'assignments',
        iteration: currentIteration,
        assignments,
        referenceCentroids: centroids,
      }
      const result = question.validator(submission)
      if (!result.correct) {
        invalidAttempt(result.message ?? 'At least one point is assigned to the wrong centroid.', submission)
        return
      }

      setFeedback(
        `Iteration ${currentIteration + 1} assignments confirmed. Now drag the centroids to the means of the colored points.`,
      )
      setStepIndex((current) => current + 1)
      onAttempt('progress', submission)
      return
    }

    const submission = {
      step: 'centroids',
      iteration: currentIteration,
      centroids,
      referenceCentroids: centroids,
      referenceAssignments: assignments,
    }
    const result = question.validator(submission)
    if (!result.correct) {
      invalidAttempt(result.message ?? 'Those centroid positions are not correct yet.', submission)
      return
    }

    const isFinalCentroidStep = currentIteration >= totalLloydIterations - 1
    if (isFinalCentroidStep) {
      setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
      onAttempt('correct', submission)
      return
    }

    setFeedback(
      `Iteration ${currentIteration + 1} centroid update confirmed. Start iteration ${currentIteration + 2} by reassigning the points.`,
    )
    setStepIndex((current) => current + 1)
    onAttempt('progress', submission)
  }

  const updateCentroidFromPointer = (clientX: number, clientY: number) => {
    const world = fromEventToWorld(clientX, clientY)
    if (world === null || draggingCentroidIndex === null) {
      return
    }

    setCentroids((current) =>
      current.map((centroid, index) => (index === draggingCentroidIndex ? world : centroid)),
    )
  }

  const handleStageClick = (event: MouseEvent<SVGSVGElement>) => {
    if (question.interactionMode !== 'placeCentroids') {
      return
    }

    const world = fromEventToWorld(event.clientX, event.clientY)
    if (world === null) {
      return
    }

    setCentroids((current) => {
      if (current.length < dataset.k) {
        return [...current, world]
      }

      const nearestIndex = current.reduce(
        (best, centroid, index) => {
          const distance = distanceSquared(world, centroid)
          return distance < best.distance ? { distance, index } : best
        },
        { distance: Number.POSITIVE_INFINITY, index: 0 },
      ).index

      return current.map((centroid, index) => (index === nearestIndex ? world : centroid))
    })
  }

  const handlePointClick = (pointIndex: number) => {
    if (!clickablePoints) {
      return
    }

    setAssignments((current) =>
      current.map((assignment, index) => (index === pointIndex ? (assignment + 1) % dataset.k : assignment)),
    )
  }

  const beginCentroidDrag = (event: PointerEvent<SVGCircleElement>, centroidIndex: number) => {
    if (!draggableCentroids) {
      return
    }

    event.stopPropagation()
    dragPointerIdRef.current = event.pointerId
    svgRef.current?.setPointerCapture(event.pointerId)
    setDraggingCentroidIndex(centroidIndex)
    updateCentroidFromPointer(event.clientX, event.clientY)
  }

  const handlePointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    if (draggingCentroidIndex === null) {
      return
    }

    updateCentroidFromPointer(event.clientX, event.clientY)
  }

  const handlePointerUp = () => {
    if (dragPointerIdRef.current !== null && svgRef.current?.hasPointerCapture(dragPointerIdRef.current)) {
      svgRef.current.releasePointerCapture(dragPointerIdRef.current)
    }
    if (dragPointerIdRef.current !== null) {
      dragPointerIdRef.current = null
    }
    setDraggingCentroidIndex(null)
  }

  const checkButtonLabel =
    question.interactionMode === 'assignPoints'
      ? 'Check assignments'
      : question.interactionMode === 'metricComparison'
        ? currentPhase === 'euclidean'
          ? 'Check Euclidean assignments'
          : 'Check Manhattan assignments'
      : question.interactionMode === 'lloydIteration'
        ? currentPhase === 'assignments'
          ? 'Check assignments'
          : 'Check centroids'
        : 'Check centroids'

  const phaseSummary =
    question.interactionMode === 'lloydIteration'
      ? currentPhase === 'assignments'
        ? `Iteration ${currentIteration + 1}, step 1 of 2: assign each point to its nearest centroid.`
        : `Iteration ${currentIteration + 1}, step 2 of 2: move each centroid to the mean of its colored cluster.`
      : question.interactionMode === 'metricComparison'
        ? currentPhase === 'euclidean'
          ? 'Step 1 of 2: color each point by the nearest centroid using Euclidean distance.'
          : 'Step 2 of 2: using the same centroids, recolor the points by the nearest centroid under Manhattan distance.'
      : question.interactionMode === 'assignPoints'
        ? 'Click points to cycle their cluster color until every assignment is correct.'
        : question.interactionMode === 'placeCentroids'
          ? "Place centroids at starting positions so that Lloyd's algorithm does not converge to the obvious clusters."
          : 'Drag each centroid to the arithmetic mean of the points in its cluster.'

  const regionCells = showRegions
    ? Array.from({ length: REGION_STEPS * REGION_STEPS }, (_, cellIndex) => {
        const columnIndex = cellIndex % REGION_STEPS
        const rowIndex = Math.floor(cellIndex / REGION_STEPS)
        const x0 = domain.minX + (columnIndex / REGION_STEPS) * (domain.maxX - domain.minX)
        const x1 = domain.minX + ((columnIndex + 1) / REGION_STEPS) * (domain.maxX - domain.minX)
        const y0 = domain.minY + (rowIndex / REGION_STEPS) * (domain.maxY - domain.minY)
        const y1 = domain.minY + ((rowIndex + 1) / REGION_STEPS) * (domain.maxY - domain.minY)
        const assignment = assignPointsToCentroids(
          [[(x0 + x1) / 2, (y0 + y1) / 2] as Vec2],
          shownCentroids,
          dataset.metric,
        )[0]
        return {
          x: toStageX(x0),
          y: toStageY(y1),
          width: toStageX(x1) - toStageX(x0),
          height: toStageY(y0) - toStageY(y1),
          fill: CLUSTER_COLORS[assignment % CLUSTER_COLORS.length],
        }
      })
    : []

  return (
    <QuestionFrame
      question={question}
      questionNumber={questionNumber}
      totalQuestions={totalQuestions}
      state={state}
      feedback={feedback}
      hints={displayedHints}
      controls={
        <>
          <button type="button" className="button" onClick={checkWork}>
            {checkButtonLabel}
          </button>
          <span className="pill">Metric: {metricLabel(activeMetric)}</span>
          {currentObjective !== null ? (
            <span className="pill">Current objective: {formatObjective(currentObjective)}</span>
          ) : null}
        </>
      }
    >
      <div className="visual-grid">
        <section className="visual-panel">
          <h3 className="panel-title">{dataset.label}</h3>
          <p className="panel-note">{phaseSummary}</p>
          <svg
            ref={svgRef}
            className="svg-stage kmeans-stage"
            viewBox={`0 0 ${STAGE_SIZE} ${STAGE_SIZE}`}
            role="img"
            aria-label={`${dataset.label} K-means interaction`}
            onClick={handleStageClick}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerUp}
          >
            <rect x="0" y="0" width={STAGE_SIZE} height={STAGE_SIZE} rx="18" fill="transparent" />

            {regionCells.map((cell, index) => (
              <rect
                key={`${question.id}-region-${index}`}
                x={cell.x}
                y={cell.y}
                width={cell.width}
                height={cell.height}
                fill={cell.fill}
                opacity="0.08"
              />
            ))}

            <line
              x1={STAGE_PADDING}
              y1={toStageY(0)}
              x2={STAGE_SIZE - STAGE_PADDING}
              y2={toStageY(0)}
              stroke="#d7d0c3"
              strokeWidth="1.2"
            />
            <line
              x1={toStageX(0)}
              y1={STAGE_PADDING}
              x2={toStageX(0)}
              y2={STAGE_SIZE - STAGE_PADDING}
              stroke="#d7d0c3"
              strokeWidth="1.2"
            />

            {showAnswer && dataset.initialCentroids && dataset.targetCentroids
              ? dataset.initialCentroids.map((centroid, index) => {
                  const target = dataset.targetCentroids?.[index]
                  if (!target) {
                    return null
                  }

                  return (
                    <line
                      key={`${question.id}-trail-${index}`}
                      x1={toStageX(centroid[0])}
                      y1={toStageY(centroid[1])}
                      x2={toStageX(target[0])}
                      y2={toStageY(target[1])}
                      stroke={CLUSTER_COLORS[index % CLUSTER_COLORS.length]}
                      strokeWidth="2.5"
                      opacity="0.32"
                    />
                  )
                })
              : null}

            {showAnswer && dataset.comparisonCentroids
              ? dataset.comparisonCentroids.map((centroid, index) => (
                  <g key={`${question.id}-comparison-centroid-${index}`}>
                    <circle
                      cx={toStageX(centroid[0])}
                      cy={toStageY(centroid[1])}
                      r="12"
                      fill="none"
                      stroke={CLUSTER_COLORS[index % CLUSTER_COLORS.length]}
                      strokeWidth="3"
                      strokeDasharray="6 5"
                      opacity="0.65"
                    />
                  </g>
                ))
              : null}

            {dataset.points.map((point, index) => {
              const assignment = shownAssignments[index]
              const fill = assignment >= 0 ? CLUSTER_COLORS[assignment % CLUSTER_COLORS.length] : '#23323b'

              return (
                <circle
                  key={`${question.id}-point-${index}`}
                  cx={toStageX(point[0])}
                  cy={toStageY(point[1])}
                  r="9"
                  fill={fill}
                  opacity={assignment >= 0 ? 0.92 : 0.86}
                  stroke="#102027"
                  strokeWidth="1.5"
                  role={clickablePoints ? 'button' : undefined}
                  aria-label={`Point ${index + 1}`}
                  tabIndex={clickablePoints ? 0 : -1}
                  className={clickablePoints ? 'interactive-point' : undefined}
                  onClick={(event) => {
                    event.stopPropagation()
                    handlePointClick(index)
                  }}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      handlePointClick(index)
                    }
                  }}
                />
              )
            })}

            {shownCentroids.map((centroid, index) => (
              <g key={`${question.id}-centroid-${index}`}>
                <circle
                  cx={toStageX(centroid[0])}
                  cy={toStageY(centroid[1])}
                  r="14"
                  fill="#fff8ef"
                  stroke={CLUSTER_COLORS[index % CLUSTER_COLORS.length]}
                  strokeWidth="4"
                  role={draggableCentroids ? 'button' : undefined}
                  aria-label={`Centroid ${index + 1}`}
                  className={draggableCentroids ? 'interactive-centroid' : undefined}
                  onClick={(event) => event.stopPropagation()}
                  onPointerDown={(event) => beginCentroidDrag(event, index)}
                />
                <text
                  x={toStageX(centroid[0])}
                  y={toStageY(centroid[1]) + 5}
                  textAnchor="middle"
                  fontSize="12"
                  fontWeight="700"
                  fill="#21313b"
                  pointerEvents="none"
                >
                  {String.fromCharCode(65 + index)}
                </text>
              </g>
            ))}
          </svg>

          <div className="legend-row">
            {Array.from({ length: dataset.k }, (_, index) => (
              <span key={`${question.id}-legend-${index}`} className="legend-chip">
                <span
                  className="legend-swatch"
                  style={{ background: CLUSTER_COLORS[index % CLUSTER_COLORS.length] }}
                />
                <span className="legend-label">Cluster {String.fromCharCode(65 + index)}</span>
              </span>
            ))}
            <span className="legend-chip">
              <span className="legend-swatch" style={{ background: '#fff8ef', border: '2px solid #20313a' }} />
              <span className="legend-label">Centroids</span>
            </span>
          </div>
        </section>

        <section className="projection-panel">
          <h3 className="panel-title">What to watch</h3>
          <div className="card-list">
            <div className="status-panel">
              <h4>Current phase</h4>
              <p className="reveal-copy">{phaseSummary}</p>
            </div>

            {dataset.objectiveBefore !== undefined ? (
              <div className="status-panel">
                <h4>Objective</h4>
                <p className="reveal-copy">
                  {dataset.objectiveSequence?.length
                    ? dataset.objectiveSequence
                        .map((value, index) => `after assignment ${index + 1}: ${formatObjective(value)}`)
                        .join(' | ')
                    : `Before update: ${formatObjective(dataset.objectiveBefore)}${
                        dataset.objectiveAfter !== undefined &&
                        (showAnswer || currentPhase === 'centroids')
                          ? ` | After update: ${formatObjective(dataset.objectiveAfter)}`
                          : ''
                      }`}
                </p>
              </div>
            ) : null}

            {showAnswer && dataset.comparisonLabel ? (
              <div className="status-panel">
                <h4>Ghost comparison</h4>
                <p className="reveal-copy">{dataset.comparisonLabel}</p>
              </div>
            ) : null}

            <div className="status-panel">
              <h4>Interaction guide</h4>
              <ul className="status-copy">
                {clickablePoints ? <li>Click a point to cycle its cluster label.</li> : null}
                {question.interactionMode === 'placeCentroids' ? (
                  <li>Click the stage to place centroids, then drag them if you need to refine them.</li>
                ) : null}
                {draggableCentroids ? <li>Drag a centroid directly to reposition it.</li> : null}
                <li>The colors show the current cluster assignments used for checking.</li>
              </ul>
            </div>
          </div>
        </section>
      </div>
    </QuestionFrame>
  )
}
