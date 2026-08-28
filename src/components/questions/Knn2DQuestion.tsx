import { useRef, useState, type MouseEvent, type ReactNode } from 'react'
import { QuestionFrame } from '../QuestionFrame'
import type {
  AttemptOutcome,
  DistanceMetric,
  Knn2DQuestionSpec,
  LabeledPoint2D,
  QuestionState,
  Vec2,
} from '../../types'
import {
  knnInteractiveDatasets,
  type KnnAdversarialPlacementDataset,
  type KnnBestKDataset,
  type KnnDecisionBoundaryDataset,
  type KnnMetricComparisonDataset,
  type KnnPredictSequenceDataset,
  type KnnScalingTrapDataset,
} from '../../data/knnDatasets'
import { DEFAULT_HINT_SCHEDULE } from '../../lib/assignmentState'
import {
  classifyKnnPoint,
  computeAccuracy,
  findBestK,
  type PlotBounds,
} from '../../lib/knnMath'

interface Knn2DQuestionProps {
  question: Knn2DQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
}

interface StageCell {
  index: number
  center: Vec2
  x0: number
  x1: number
  y0: number
  y1: number
  row: number
  column: number
}

interface StageMargins {
  top: number
  right: number
  bottom: number
  left: number
}

const STAGE_SIZE = 430
const STAGE_PADDING = 28
const DEFAULT_BOUNDARY_RESOLUTION = 18
const NEUTRAL_CELL_FILL = '#ebe2d3'
const QUERY_FILL = '#1a252b'
const DEFAULT_STAGE_MARGINS: StageMargins = {
  top: STAGE_PADDING,
  right: STAGE_PADDING,
  bottom: STAGE_PADDING,
  left: STAGE_PADDING,
}
const AXIS_STAGE_MARGINS: StageMargins = {
  top: 24,
  right: 28,
  bottom: 56,
  left: 62,
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function metricLabel(metric: DistanceMetric) {
  return metric === 'manhattan' ? 'Manhattan' : 'Euclidean'
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`
}

function formatAxisTickLabel(value: number) {
  if (Math.abs(value - Math.round(value)) < 1e-9) {
    return String(Math.round(value))
  }

  return value.toFixed(Math.abs(value) < 2 ? 2 : 1).replace(/\.?0+$/, '')
}

function blankDecisionGrid(columns: number, rows: number) {
  return Array.from({ length: columns * rows }, () => null as number | null)
}

function buildStageCells(bounds: PlotBounds, columns: number, rows: number): StageCell[] {
  return Array.from({ length: columns * rows }, (_, cellIndex) => {
    const column = cellIndex % columns
    const row = Math.floor(cellIndex / columns)
    const x0 = bounds.minX + (column / columns) * (bounds.maxX - bounds.minX)
    const x1 = bounds.minX + ((column + 1) / columns) * (bounds.maxX - bounds.minX)
    const y0 = bounds.minY + (row / rows) * (bounds.maxY - bounds.minY)
    const y1 = bounds.minY + ((row + 1) / rows) * (bounds.maxY - bounds.minY)

    return {
      index: cellIndex,
      center: [(x0 + x1) / 2, (y0 + y1) / 2],
      x0,
      x1,
      y0,
      y1,
      row,
      column,
    }
  })
}

function diamondPoints(cx: number, cy: number, radius: number) {
  return `${cx},${cy - radius} ${cx + radius},${cy} ${cx},${cy + radius} ${cx - radius},${cy}`
}

function SummaryPanel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="status-panel">
      <h4>{title}</h4>
      {children}
    </div>
  )
}

export function Knn2DQuestion({
  question,
  state,
  questionNumber,
  totalQuestions,
  hints,
  onAttempt,
}: Knn2DQuestionProps) {
  const dataset = knnInteractiveDatasets[question.datasetId]
  const svgRef = useRef<SVGSVGElement | null>(null)
  const [feedback, setFeedback] = useState('')
  const [stepIndex, setStepIndex] = useState(0)
  const [selectedLabel, setSelectedLabel] = useState<number | null>(null)
  const [brushLabel, setBrushLabel] = useState(0)
  const [paintedCells, setPaintedCells] = useState<(number | null)[]>(() =>
    dataset.kind === 'decisionBoundary'
      ? blankDecisionGrid(dataset.gridColumns, dataset.gridRows)
      : [],
  )
  const [selectedK, setSelectedK] = useState(() => (dataset.kind === 'bestK' ? dataset.initialK : 1))
  const [candidatePoint, setCandidatePoint] = useState<Vec2 | null>(null)
  const showAnswer = state.status === 'correct' || state.status === 'gave_up'
  const boundariesUnlocked = state.status === 'correct'
  const currentSchedule = question.hintSchedule ?? DEFAULT_HINT_SCHEDULE

  const currentDraft = () => {
    switch (question.interactionMode) {
      case 'predictSequence':
      case 'decisionBoundary':
        return question.interactionMode === 'predictSequence'
          ? { step: stepIndex, predictedLabel: selectedLabel }
          : { step: stepIndex, cells: paintedCells }
      case 'bestK':
        return { k: selectedK }
      case 'scalingTrap':
        return { step: stepIndex > 0 ? 'normalized' : 'raw', predictedLabel: selectedLabel }
      case 'metricComparison':
        return { step: stepIndex > 0 ? 'manhattan' : 'euclidean', predictedLabel: selectedLabel }
      case 'adversarialPlacement':
        return { point: candidatePoint }
    }
  }

  const invalidAttempt = (message: string, answer: unknown = currentDraft()) => {
    const nextIncorrectAttempts = state.incorrectAttempts + 1
    setFeedback(
      currentSchedule.includes(nextIncorrectAttempts)
        ? 'Not yet. A new hint appeared below.'
        : message,
    )
    onAttempt('incorrect', answer)
  }

  const defaultBounds = dataset.bounds
  let stageBounds = defaultBounds
  let stageTrainingPoints = dataset.trainingPoints
  let stageTestPoints: LabeledPoint2D[] = []
  let stageQueryPoint: Vec2 | null = null
  let currentMetric: DistanceMetric = 'euclidean'
  let currentK = 1
  let currentViewLabel = ''
  let stageSummary = ''
  let checkButtonLabel = 'Check answer'
  let classPrompt = ''
  let showClassButtons = false
  let showSlider = false
  let showBoundary = false
  let showDecisionGrid = false
  let decisionGridCells: (number | null)[] = []
  let gridTemplate: StageCell[] = []
  let boundaryTemplate: StageCell[] = []
  let guideItems: string[] = []
  let infoPanels: Array<{ title: string; body: ReactNode }> = []
  let summaryLines: string[] = []
  let stageMargins = DEFAULT_STAGE_MARGINS
  let showQueryLegend = false
  let showTestLegend = false
  let showAddedPointLegend = false
  let addedPoint: LabeledPoint2D | null = null
  let currentTestAccuracy = 0
  let currentTrainingAccuracy = 0
  let bestChoice: { k: number; accuracy: number } | null = null
  let axisLabels: { x: string; y: string } | null = null
  let axisTicks: { x: number[]; y: number[] } | null = null

  switch (question.interactionMode) {
    case 'predictSequence': {
      const currentDataset = dataset as KnnPredictSequenceDataset
      const currentStep = Math.min(stepIndex, currentDataset.kSequence.length - 1)

      stageTrainingPoints = currentDataset.trainingPoints
      stageQueryPoint = currentDataset.queryPoint
      currentMetric = currentDataset.metric
      currentK = currentDataset.kSequence[currentStep]
      currentViewLabel = `k = ${currentK}`
      stageSummary = `Step ${currentStep + 1} of ${currentDataset.kSequence.length}: predict the query point's class for k = ${currentK}.`
      checkButtonLabel = `Check k = ${currentK}`
      classPrompt = `Choose the class for k = ${currentK}.`
      showClassButtons = true
      showQueryLegend = true
      summaryLines = currentDataset.kSequence.map((value) => {
        const labelIndex = classifyKnnPoint(
          currentDataset.trainingPoints,
          currentDataset.queryPoint,
          value,
          currentDataset.metric,
        ).label
        return `k = ${value}: ${currentDataset.classes[labelIndex].label}`
      })
      guideItems = [
        'The black diamond is the query point.',
        'Only the nearest k points vote. The rest are ignored.',
        'The dataset stays fixed while the neighborhood size changes.',
      ]
      infoPanels = [
        {
          title: 'Current phase',
          body: <p className="reveal-copy">{stageSummary}</p>,
        },
        {
          title: 'What changes with k',
          body: (
            <p className="reveal-copy">
              Small k listens to a tight local neighborhood. Large k smooths the prediction toward the wider class balance.
            </p>
          ),
        },
      ]
      if (showAnswer) {
        infoPanels.push({
          title: 'Correct answers',
          body: (
            <ul className="status-copy">
              {summaryLines.map((line) => (
                <li key={`${question.id}-${line}`}>{line}</li>
              ))}
            </ul>
          ),
        })
      } else if (selectedLabel !== null) {
        infoPanels.push({
          title: 'Your current choice',
          body: <p className="reveal-copy">{currentDataset.classes[selectedLabel].label}</p>,
        })
      }
      break
    }

    case 'decisionBoundary': {
      const currentDataset = dataset as KnnDecisionBoundaryDataset
      const currentStep = Math.min(stepIndex, currentDataset.kSequence.length - 1)

      currentMetric = currentDataset.metric
      currentK = currentDataset.kSequence[currentStep]
      currentViewLabel = `k = ${currentK}`
      stageSummary = `Step ${currentStep + 1} of ${currentDataset.kSequence.length}: paint the ${currentDataset.gridColumns} by ${currentDataset.gridRows} grid for k = ${currentK}.`
      checkButtonLabel = `Check k = ${currentK} boundary`
      showDecisionGrid = true
      decisionGridCells = paintedCells
      gridTemplate = buildStageCells(
        currentDataset.bounds,
        currentDataset.gridColumns,
        currentDataset.gridRows,
      )
      guideItems = [
        'Choose a class brush, then click cells to paint them.',
        'At k = 1, isolated points can create tiny islands.',
        'At larger k, the cells average over a wider neighborhood.',
      ]
      summaryLines = currentDataset.kSequence.map((value) => `Boundary checked for k = ${value}`)
      infoPanels = [
        {
          title: 'Current phase',
          body: <p className="reveal-copy">{stageSummary}</p>,
        },
        {
          title: 'Active brush',
          body: <p className="reveal-copy">{currentDataset.classes[brushLabel].label}</p>,
        },
      ]
      if (showAnswer) {
        infoPanels.push({
          title: 'Why the boundary changed',
          body: (
            <ul className="status-copy">
              <li>k = 1 follows individual points closely.</li>
              <li>k = 5 smooths away the outlier island by averaging over more neighbors.</li>
            </ul>
          ),
        })
      }
      break
    }

    case 'bestK': {
      const currentDataset = dataset as KnnBestKDataset
      stageTrainingPoints = currentDataset.trainingPoints
      stageTestPoints = currentDataset.testPoints
      currentMetric = currentDataset.metric
      currentK = selectedK
      currentViewLabel = `k = ${selectedK}`
      stageSummary = 'Move the slider and watch how the boundary, the training accuracy, and the test accuracy change together.'
      checkButtonLabel = 'Lock in this k'
      showSlider = true
      showBoundary = true
      showTestLegend = true
      currentTrainingAccuracy = computeAccuracy(
        currentDataset.trainingPoints,
        currentDataset.trainingPoints,
        selectedK,
        currentDataset.metric,
      )
      currentTestAccuracy = computeAccuracy(
        currentDataset.trainingPoints,
        currentDataset.testPoints,
        selectedK,
        currentDataset.metric,
      )
      bestChoice = findBestK(
        currentDataset.trainingPoints,
        currentDataset.testPoints,
        currentDataset.kValues,
        currentDataset.metric,
      )
      guideItems = [
        'Filled circles are training points.',
        'Ringed circles are held-out test points.',
        'Pick the k that maximizes test accuracy, not training accuracy.',
      ]
      infoPanels = [
        {
          title: 'Accuracy readout',
          body: (
            <div className="accuracy-grid">
              <div className="metric-card">
                <span className="metric-value">{formatPercent(currentTrainingAccuracy)}</span>
                <span className="metric-copy">Training accuracy</span>
              </div>
              <div className="metric-card">
                <span className="metric-value">{formatPercent(currentTestAccuracy)}</span>
                <span className="metric-copy">Test accuracy</span>
              </div>
            </div>
          ),
        },
      ]
      if (showAnswer && bestChoice) {
        infoPanels.push({
          title: 'Best held-out choice',
          body: (
            <p className="reveal-copy">
              k = {bestChoice.k} reaches the highest test accuracy: {formatPercent(bestChoice.accuracy)}.
            </p>
          ),
        })
      }
      break
    }

    case 'scalingTrap': {
      const currentDataset = dataset as KnnScalingTrapDataset
      const normalizedPhase = stepIndex > 0
      const feature1Values = currentDataset.trainingPoints.map((entry) => entry.point[0])
      const feature2Values = currentDataset.trainingPoints.map((entry) => entry.point[1])
      const feature1Min = Math.min(...feature1Values)
      const feature1Max = Math.max(...feature1Values)
      const feature2Min = Math.min(...feature2Values)
      const feature2Max = Math.max(...feature2Values)
      const rawLabel = classifyKnnPoint(
        currentDataset.trainingPoints,
        currentDataset.queryPoint,
        currentDataset.k,
        currentDataset.metric,
      ).label
      const normalizedLabel = classifyKnnPoint(
        currentDataset.normalizedTrainingPoints,
        currentDataset.normalizedQueryPoint,
        currentDataset.k,
        currentDataset.metric,
      ).label

      stageTrainingPoints = normalizedPhase
        ? currentDataset.normalizedTrainingPoints
        : currentDataset.trainingPoints
      stageQueryPoint = normalizedPhase
        ? currentDataset.normalizedQueryPoint
        : currentDataset.queryPoint
      stageBounds = normalizedPhase ? currentDataset.normalizedBounds : currentDataset.bounds
      stageMargins = AXIS_STAGE_MARGINS
      currentMetric = currentDataset.metric
      currentK = currentDataset.k
      currentViewLabel = normalizedPhase ? 'Normalized view' : 'Raw feature scales'
      stageSummary = normalizedPhase
        ? 'Step 2 of 2: predict the query point again after min-max normalization.'
        : 'Step 1 of 2: predict the query point before any normalization.'
      checkButtonLabel = normalizedPhase ? 'Check normalized prediction' : 'Check unscaled prediction'
      classPrompt = normalizedPhase
        ? 'Choose the class after normalization.'
        : 'Choose the class before normalization.'
      showClassButtons = true
      showBoundary = boundariesUnlocked
      showQueryLegend = true
      summaryLines = [
        `Before normalization: ${currentDataset.classes[rawLabel].label}`,
        `After normalization: ${currentDataset.classes[normalizedLabel].label}`,
      ]
      axisLabels = normalizedPhase
        ? {
            x: `${currentDataset.featureLabels[0]} (normalized)`,
            y: `${currentDataset.featureLabels[1]} (normalized)`,
          }
        : {
            x: currentDataset.featureLabels[0],
            y: currentDataset.featureLabels[1],
          }
      axisTicks = normalizedPhase
        ? {
            x: [0, 0.25, 0.5, 0.75, 1],
            y: [0, 0.25, 0.5, 0.75, 1],
          }
        : {
            x: [0, 25, 50, 75, 100],
            y: [0, 0.25, 0.5, 0.75, 1],
          }
      guideItems = [
        'The black diamond is the query point.',
        'Feature 1 has a much larger raw range than Feature 2.',
        'After normalization, both features live on comparable 0 to 1 scales.',
      ]
      infoPanels = [
        {
          title: 'Current phase',
          body: <p className="reveal-copy">{stageSummary}</p>,
        },
        {
          title: 'Feature ranges',
          body: (
            <ul className="status-copy">
              <li>
                {currentDataset.featureLabels[0]} spans roughly {formatAxisTickLabel(feature1Min)} to{' '}
                {formatAxisTickLabel(feature1Max)} before scaling.
              </li>
              <li>
                {currentDataset.featureLabels[1]} spans roughly {formatAxisTickLabel(feature2Min)} to{' '}
                {formatAxisTickLabel(feature2Max)} before scaling.
              </li>
            </ul>
          ),
        },
      ]
      if (showAnswer) {
        infoPanels.push({
          title: 'Before and after',
          body: (
            <ul className="status-copy">
              {summaryLines.map((line) => (
                <li key={`${question.id}-${line}`}>{line}</li>
              ))}
            </ul>
          ),
        })
      }
      break
    }

    case 'metricComparison': {
      const currentDataset = dataset as KnnMetricComparisonDataset
      const manhattanPhase = stepIndex > 0
      const euclideanLabel = classifyKnnPoint(
        currentDataset.trainingPoints,
        currentDataset.queryPoint,
        currentDataset.k,
        currentDataset.metric,
      ).label
      const manhattanLabel = classifyKnnPoint(
        currentDataset.trainingPoints,
        currentDataset.queryPoint,
        currentDataset.k,
        currentDataset.comparisonMetric,
      ).label

      stageTrainingPoints = currentDataset.trainingPoints
      stageQueryPoint = currentDataset.queryPoint
      currentMetric = manhattanPhase ? currentDataset.comparisonMetric : currentDataset.metric
      currentK = currentDataset.k
      currentViewLabel = manhattanPhase ? 'Manhattan distance' : 'Euclidean distance'
      stageSummary = manhattanPhase
        ? `Step 2 of 2: still using k = ${currentDataset.k}, switch the metric to Manhattan distance.`
        : `Step 1 of 2: using k = ${currentDataset.k}, predict the query point under Euclidean distance.`
      checkButtonLabel = manhattanPhase ? 'Check Manhattan prediction' : 'Check Euclidean prediction'
      classPrompt = manhattanPhase
        ? `Choose the class under Manhattan distance with k = ${currentDataset.k}.`
        : `Choose the class under Euclidean distance with k = ${currentDataset.k}.`
      showClassButtons = true
      showBoundary = boundariesUnlocked
      showQueryLegend = true
      summaryLines = [
        `Euclidean: ${currentDataset.classes[euclideanLabel].label}`,
        `Manhattan: ${currentDataset.classes[manhattanLabel].label}`,
      ]
      guideItems = [
        'The points do not move between the two steps.',
        'Euclidean distance rewards diagonal closeness more directly.',
        'Manhattan distance measures horizontal-plus-vertical travel instead.',
      ]
      infoPanels = [
        {
          title: 'Current phase',
          body: <p className="reveal-copy">{stageSummary}</p>,
        },
        {
          title: 'What stays fixed',
          body: (
            <p className="reveal-copy">
              The data points and k = {currentDataset.k} stay fixed. Only the distance metric changes.
            </p>
          ),
        },
      ]
      if (showAnswer) {
        infoPanels.push({
          title: 'Answer comparison',
          body: (
            <ul className="status-copy">
              {summaryLines.map((line) => (
                <li key={`${question.id}-${line}`}>{line}</li>
              ))}
            </ul>
          ),
        })
      }
      break
    }

    case 'adversarialPlacement': {
      const currentDataset = dataset as KnnAdversarialPlacementDataset
      const originalLabel = classifyKnnPoint(
        currentDataset.trainingPoints,
        currentDataset.queryPoint,
        currentDataset.k,
        currentDataset.metric,
      ).label
      const effectivePoint = candidatePoint

      stageQueryPoint = currentDataset.queryPoint
      currentMetric = currentDataset.metric
      currentK = currentDataset.k
      currentViewLabel = `Add one ${currentDataset.classes[currentDataset.examplePoint.label].label} point`
      stageSummary = `The query point is currently ${currentDataset.classes[originalLabel].label}. Click to place one new ${currentDataset.classes[currentDataset.examplePoint.label].label} training point that flips the vote.`
      checkButtonLabel = 'Check new point'
      showBoundary = true
      showQueryLegend = true
      showAddedPointLegend = true
      addedPoint = effectivePoint
        ? {
            point: effectivePoint,
            label: currentDataset.examplePoint.label,
          }
        : null
      stageTrainingPoints = addedPoint
        ? [...currentDataset.trainingPoints, addedPoint]
        : currentDataset.trainingPoints
      summaryLines = [
        `Original prediction: ${currentDataset.classes[originalLabel].label}`,
        `Target after the new point: ${currentDataset.classes[currentDataset.examplePoint.label].label}`,
      ]
      guideItems = [
        'Click anywhere on the stage to place or move the new point.',
        'To matter, the new point must enter the set of the three nearest neighbors.',
        'You only need to flip a 2-to-1 vote.',
      ]
      infoPanels = [
        {
          title: 'Current vote',
          body: (
            <p className="reveal-copy">
              Without your added point, the query is classified as {currentDataset.classes[originalLabel].label}.
            </p>
          ),
        },
      ]
      if (showAnswer) {
        infoPanels.push({
          title: 'What changed',
          body: (
            <ul className="status-copy">
              {summaryLines.map((line) => (
                <li key={`${question.id}-${line}`}>{line}</li>
              ))}
            </ul>
          ),
        })
      }
      break
    }
  }

  const toStageX = (value: number) =>
    stageMargins.left +
    ((value - stageBounds.minX) / (stageBounds.maxX - stageBounds.minX)) *
      (STAGE_SIZE - stageMargins.left - stageMargins.right)
  const toStageY = (value: number) =>
    STAGE_SIZE -
    stageMargins.bottom -
    ((value - stageBounds.minY) / (stageBounds.maxY - stageBounds.minY)) *
      (STAGE_SIZE - stageMargins.top - stageMargins.bottom)

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
      stageBounds.minX + normalizedX * (stageBounds.maxX - stageBounds.minX),
      stageBounds.maxY - normalizedY * (stageBounds.maxY - stageBounds.minY),
    ]
  }

  if (showBoundary) {
    const boundaryResolution = dataset.boundaryResolution ?? DEFAULT_BOUNDARY_RESOLUTION
    boundaryTemplate = buildStageCells(stageBounds, boundaryResolution, boundaryResolution)
  }

  const handleStageClick = (event: MouseEvent<SVGSVGElement>) => {
    if (question.interactionMode !== 'adversarialPlacement') {
      return
    }

    const point = fromEventToWorld(event.clientX, event.clientY)
    if (point) {
      setCandidatePoint(point)
    }
  }

  const checkWork = () => {
    switch (question.interactionMode) {
      case 'predictSequence': {
        const currentDataset = dataset as KnnPredictSequenceDataset
        const currentStep = Math.min(stepIndex, currentDataset.kSequence.length - 1)
        if (selectedLabel === null) {
          invalidAttempt('Choose a class before checking your answer.')
          return
        }

        const submission = {
          step: currentStep,
          predictedLabel: selectedLabel,
        }
        const result = question.validator(submission)

        if (!result.correct) {
          invalidAttempt(result.message ?? 'That prediction is not correct yet.')
          return
        }

        if (currentStep < currentDataset.kSequence.length - 1) {
          const nextK = currentDataset.kSequence[currentStep + 1]
          setFeedback(
            `k = ${currentDataset.kSequence[currentStep]} is correct. Now repeat the prediction for k = ${nextK}.`,
          )
          setSelectedLabel(null)
          setStepIndex((current) => current + 1)
          onAttempt('progress', submission)
          return
        }

        setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
        onAttempt('correct', submission)
        return
      }

      case 'decisionBoundary': {
        const currentDataset = dataset as KnnDecisionBoundaryDataset
        const currentStep = Math.min(stepIndex, currentDataset.kSequence.length - 1)
        if (paintedCells.some((cell) => cell === null)) {
          invalidAttempt('Paint every grid cell before checking the boundary.')
          return
        }

        const submission = {
          step: currentStep,
          cells: paintedCells,
        }
        const result = question.validator(submission)

        if (!result.correct) {
          invalidAttempt(result.message ?? 'At least one grid cell is still wrong.')
          return
        }

        if (currentStep < currentDataset.kSequence.length - 1) {
          setFeedback(`The k = ${currentDataset.kSequence[currentStep]} boundary is correct. Now repaint the grid for k = ${currentDataset.kSequence[currentStep + 1]}.`)
          setStepIndex((current) => current + 1)
          setPaintedCells(blankDecisionGrid(currentDataset.gridColumns, currentDataset.gridRows))
          onAttempt('progress', submission)
          return
        }

        setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
        onAttempt('correct', submission)
        return
      }

      case 'bestK': {
        const submission = { k: selectedK }
        const result = question.validator(submission)
        if (!result.correct) {
          invalidAttempt(result.message ?? 'That is not the best value of k.')
          return
        }

        setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
        onAttempt('correct', submission)
        return
      }

      case 'scalingTrap': {
        if (selectedLabel === null) {
          invalidAttempt('Choose a class before checking your answer.')
          return
        }

        const normalizedPhase = stepIndex > 0
        const submission = {
          step: normalizedPhase ? 'normalized' : 'raw',
          predictedLabel: selectedLabel,
        }
        const result = question.validator(submission)

        if (!result.correct) {
          invalidAttempt(result.message ?? 'That prediction is not correct yet.')
          return
        }

        if (!normalizedPhase) {
          setFeedback('The unscaled prediction is correct. The plot has now switched to normalized coordinates; predict the class again.')
          setSelectedLabel(null)
          setStepIndex(1)
          onAttempt('progress', submission)
          return
        }

        setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
        onAttempt('correct', submission)
        return
      }

      case 'metricComparison': {
        if (selectedLabel === null) {
          invalidAttempt('Choose a class before checking your answer.')
          return
        }

        const manhattanPhase = stepIndex > 0
        const submission = {
          step: manhattanPhase ? 'manhattan' : 'euclidean',
          predictedLabel: selectedLabel,
        }
        const result = question.validator(submission)

        if (!result.correct) {
          invalidAttempt(result.message ?? 'That prediction is not correct yet.')
          return
        }

        if (!manhattanPhase) {
          setFeedback('The Euclidean prediction is correct. Keep the same k and switch to Manhattan distance.')
          setSelectedLabel(null)
          setStepIndex(1)
          onAttempt('progress', submission)
          return
        }

        setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
        onAttempt('correct', submission)
        return
      }

      case 'adversarialPlacement': {
        if (!candidatePoint) {
          invalidAttempt('Click on the plot to place the new training point first.')
          return
        }

        const submission = {
          point: candidatePoint,
        }
        const result = question.validator(submission)

        if (!result.correct) {
          invalidAttempt(result.message ?? 'That new point does not flip the prediction yet.')
          return
        }

        setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
        onAttempt('correct', submission)
      }
    }
  }

  const zeroYVisible = stageBounds.minY <= 0 && stageBounds.maxY >= 0
  const zeroXVisible = stageBounds.minX <= 0 && stageBounds.maxX >= 0
  const xAxisY = zeroYVisible ? toStageY(0) : STAGE_SIZE - stageMargins.bottom
  const yAxisX = zeroXVisible ? toStageX(0) : stageMargins.left

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
          <button type="button" className="button" onClick={checkWork}>
            {checkButtonLabel}
          </button>
          <span className="pill">{currentViewLabel}</span>
          <span className="pill">Metric: {metricLabel(currentMetric)}</span>
          <span className="pill">k = {currentK}</span>
          {question.interactionMode === 'bestK' ? (
            <span className="pill">Test accuracy: {formatPercent(currentTestAccuracy)}</span>
          ) : null}
        </>
      }
    >
      <div className="visual-grid">
        <section className="visual-panel">
          <h3 className="panel-title">{dataset.label}</h3>
          <p className="panel-note">{stageSummary}</p>

          {showSlider ? (
            <label className="slider-shell" htmlFor={`${question.id}-k-slider`}>
              <span>Current k</span>
              <input
                id={`${question.id}-k-slider`}
                className="range-input"
                type="range"
                min={(dataset as KnnBestKDataset).kValues[0]}
                max={(dataset as KnnBestKDataset).kValues[(dataset as KnnBestKDataset).kValues.length - 1]}
                step="1"
                value={selectedK}
                onChange={(event) => setSelectedK(Number(event.target.value))}
              />
              <span className="question-index-chip">k = {selectedK}</span>
            </label>
          ) : null}

          {showClassButtons ? (
            <div className="knn-toolbar">
              <p className="panel-note">{classPrompt}</p>
              <div className="class-picker" role="radiogroup" aria-label={classPrompt}>
                {dataset.classes.map((classSpec, classIndex) => (
                  <button
                    key={`${question.id}-class-${classSpec.label}`}
                    type="button"
                    className={`class-choice ${selectedLabel === classIndex ? 'selected' : ''}`}
                    style={{
                      borderColor: selectedLabel === classIndex ? classSpec.color : 'rgba(23, 34, 40, 0.08)',
                      background:
                        selectedLabel === classIndex
                          ? `color-mix(in srgb, ${classSpec.color} 18%, white)`
                          : undefined,
                    }}
                    onClick={() => setSelectedLabel(classIndex)}
                  >
                    <span className="legend-swatch" style={{ background: classSpec.color }} />
                    <span>{classSpec.label}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          {showDecisionGrid ? (
            <div className="knn-toolbar">
              <p className="panel-note">Choose a paint brush, then color each cell in the grid.</p>
              <div className="class-picker" role="radiogroup" aria-label="Choose a class brush">
                {dataset.classes.map((classSpec, classIndex) => (
                  <button
                    key={`${question.id}-brush-${classSpec.label}`}
                    type="button"
                    className={`class-choice ${brushLabel === classIndex ? 'selected' : ''}`}
                    style={{
                      borderColor: brushLabel === classIndex ? classSpec.color : 'rgba(23, 34, 40, 0.08)',
                      background:
                        brushLabel === classIndex
                          ? `color-mix(in srgb, ${classSpec.color} 18%, white)`
                          : undefined,
                    }}
                    onClick={() => setBrushLabel(classIndex)}
                  >
                    <span className="legend-swatch" style={{ background: classSpec.color }} />
                    <span>{classSpec.label}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          <svg
            ref={svgRef}
            className="svg-stage knn-stage"
            viewBox={`0 0 ${STAGE_SIZE} ${STAGE_SIZE}`}
            role="img"
            aria-label={dataset.label}
            onClick={handleStageClick}
          >
            <rect x="0" y="0" width={STAGE_SIZE} height={STAGE_SIZE} rx="18" fill="transparent" />

            {showBoundary
              ? boundaryTemplate.map((cell) => {
                  const label = classifyKnnPoint(
                    stageTrainingPoints,
                    cell.center,
                    currentK,
                    currentMetric,
                  ).label
                  const color = dataset.classes[label]?.color ?? QUERY_FILL

                  return (
                    <rect
                      key={`${question.id}-boundary-${cell.index}`}
                      x={toStageX(cell.x0)}
                      y={toStageY(cell.y1)}
                      width={toStageX(cell.x1) - toStageX(cell.x0)}
                      height={toStageY(cell.y0) - toStageY(cell.y1)}
                      fill={color}
                      opacity="0.12"
                    />
                  )
                })
              : null}

            {showDecisionGrid
              ? gridTemplate.map((cell) => {
                  const cellLabel = decisionGridCells[cell.index]
                  return (
                    <rect
                      key={`${question.id}-grid-${cell.index}`}
                      x={toStageX(cell.x0)}
                      y={toStageY(cell.y1)}
                      width={toStageX(cell.x1) - toStageX(cell.x0)}
                      height={toStageY(cell.y0) - toStageY(cell.y1)}
                      fill={
                        cellLabel === null
                          ? NEUTRAL_CELL_FILL
                          : dataset.classes[cellLabel]?.color ?? QUERY_FILL
                      }
                      opacity={cellLabel === null ? 0.38 : 0.28}
                      stroke="rgba(72, 86, 93, 0.36)"
                      strokeWidth="1"
                      role="button"
                      aria-label={`Cell row ${cell.row + 1} column ${cell.column + 1}`}
                      tabIndex={0}
                      className="interactive-cell"
                      onClick={(event) => {
                        event.stopPropagation()
                        setPaintedCells((current) =>
                          current.map((value, index) => (index === cell.index ? brushLabel : value)),
                        )
                      }}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault()
                          setPaintedCells((current) =>
                            current.map((value, index) => (index === cell.index ? brushLabel : value)),
                          )
                        }
                      }}
                    />
                  )
                })
              : null}

            {zeroYVisible ? (
              <line
                x1={stageMargins.left}
                y1={xAxisY}
                x2={STAGE_SIZE - stageMargins.right}
                y2={xAxisY}
                stroke="#d7d0c3"
                strokeWidth="1.2"
              />
            ) : null}
            {zeroXVisible ? (
              <line
                x1={yAxisX}
                y1={stageMargins.top}
                x2={yAxisX}
                y2={STAGE_SIZE - stageMargins.bottom}
                stroke="#d7d0c3"
                strokeWidth="1.2"
              />
            ) : null}

            {axisTicks && axisLabels ? (
              <g aria-hidden="true">
                {axisTicks.x.map((tickValue) => {
                  const x = toStageX(tickValue)
                  return (
                    <g key={`${question.id}-x-axis-${tickValue}`}>
                      <line
                        x1={x}
                        y1={xAxisY}
                        x2={x}
                        y2={xAxisY + 7}
                        stroke="#89969c"
                        strokeWidth="1.2"
                      />
                      <text
                        x={x}
                        y={xAxisY + 22}
                        textAnchor="middle"
                        fontSize="11.5"
                        fill="#55646b"
                      >
                        {formatAxisTickLabel(tickValue)}
                      </text>
                    </g>
                  )
                })}

                {axisTicks.y.map((tickValue) => {
                  const y = toStageY(tickValue)
                  return (
                    <g key={`${question.id}-y-axis-${tickValue}`}>
                      <line
                        x1={yAxisX - 7}
                        y1={y}
                        x2={yAxisX}
                        y2={y}
                        stroke="#89969c"
                        strokeWidth="1.2"
                      />
                      <text
                        x={yAxisX - 12}
                        y={y + 4}
                        textAnchor="end"
                        fontSize="11.5"
                        fill="#55646b"
                      >
                        {formatAxisTickLabel(tickValue)}
                      </text>
                    </g>
                  )
                })}

                <text
                  x={(stageMargins.left + (STAGE_SIZE - stageMargins.right)) / 2}
                  y={STAGE_SIZE - 14}
                  textAnchor="middle"
                  fontSize="12.5"
                  fontWeight="700"
                  fill="#304048"
                >
                  {axisLabels.x}
                </text>

                <text
                  x={18}
                  y={(stageMargins.top + (STAGE_SIZE - stageMargins.bottom)) / 2}
                  textAnchor="middle"
                  transform={`rotate(-90 18 ${(stageMargins.top + (STAGE_SIZE - stageMargins.bottom)) / 2})`}
                  fontSize="12.5"
                  fontWeight="700"
                  fill="#304048"
                >
                  {axisLabels.y}
                </text>
              </g>
            ) : null}

            {stageTrainingPoints.map((point, index) => {
              const color = dataset.classes[point.label]?.color ?? QUERY_FILL
              const isAddedPoint =
                question.interactionMode === 'adversarialPlacement' &&
                addedPoint !== null &&
                index === stageTrainingPoints.length - 1 &&
                point.point[0] === addedPoint.point[0] &&
                point.point[1] === addedPoint.point[1] &&
                point.label === addedPoint.label

              return (
                <g key={`${question.id}-train-${index}`}>
                  {isAddedPoint ? (
                    <circle
                      cx={toStageX(point.point[0])}
                      cy={toStageY(point.point[1])}
                      r="12"
                      fill="none"
                      stroke={QUERY_FILL}
                      strokeWidth="2.5"
                      opacity="0.7"
                    />
                  ) : null}
                  <circle
                    cx={toStageX(point.point[0])}
                    cy={toStageY(point.point[1])}
                    r={isAddedPoint ? 9.5 : 8.5}
                    fill={color}
                    stroke={isAddedPoint ? QUERY_FILL : '#102027'}
                    strokeWidth={isAddedPoint ? 2.4 : 1.5}
                    opacity="0.96"
                  />
                </g>
              )
            })}

            {stageTestPoints.map((point, index) => {
              const color = dataset.classes[point.label]?.color ?? QUERY_FILL
              return (
                <g key={`${question.id}-test-${index}`}>
                  <circle
                    cx={toStageX(point.point[0])}
                    cy={toStageY(point.point[1])}
                    r="10"
                    fill="#fff8ef"
                    stroke={color}
                    strokeWidth="3"
                    opacity="0.96"
                  />
                  <circle
                    cx={toStageX(point.point[0])}
                    cy={toStageY(point.point[1])}
                    r="3.8"
                    fill={color}
                  />
                </g>
              )
            })}

            {stageQueryPoint ? (
              <g>
                <polygon
                  points={diamondPoints(toStageX(stageQueryPoint[0]), toStageY(stageQueryPoint[1]), 12)}
                  fill={QUERY_FILL}
                  stroke="#fff8ef"
                  strokeWidth="2.2"
                />
                <text
                  x={toStageX(stageQueryPoint[0])}
                  y={toStageY(stageQueryPoint[1]) + 4.5}
                  textAnchor="middle"
                  fontSize="12"
                  fontWeight="700"
                  fill="#fff8ef"
                  pointerEvents="none"
                >
                  Q
                </text>
              </g>
            ) : null}
          </svg>

          <div className="legend-row">
            {dataset.classes.map((classSpec) => (
              <span key={`${question.id}-${classSpec.label}`} className="legend-chip">
                <span className="legend-swatch" style={{ background: classSpec.color }} />
                <span className="legend-label">{classSpec.label}</span>
              </span>
            ))}
            {showQueryLegend ? (
              <span className="legend-chip">
                <span className="legend-swatch query-swatch" />
                <span className="legend-label">Query point</span>
              </span>
            ) : null}
            {showTestLegend ? (
              <span className="legend-chip">
                <span className="legend-swatch test-swatch" />
                <span className="legend-label">Held-out test point</span>
              </span>
            ) : null}
            {showAddedPointLegend ? (
              <span className="legend-chip">
                <span
                  className="legend-swatch"
                  style={{
                    background:
                      dataset.classes[(dataset as KnnAdversarialPlacementDataset).examplePoint.label]?.color,
                    boxShadow: '0 0 0 2px #1a252b inset',
                  }}
                />
                <span className="legend-label">Your new point</span>
              </span>
            ) : null}
          </div>
        </section>

        <section className="projection-panel">
          <h3 className="panel-title">What to watch</h3>
          <div className="card-list">
            {infoPanels.map((panel) => (
              <SummaryPanel key={`${question.id}-${panel.title}`} title={panel.title}>
                {panel.body}
              </SummaryPanel>
            ))}

            <SummaryPanel title="Interaction guide">
              <ul className="status-copy">
                {guideItems.map((item) => (
                  <li key={`${question.id}-${item}`}>{item}</li>
                ))}
              </ul>
            </SummaryPanel>

            {question.interactionMode === 'bestK' && bestChoice ? (
              <SummaryPanel title="Model-selection clue">
                <p className="reveal-copy">
                  Training accuracy can stay deceptively high for very small k. The held-out test accuracy is the quantity you should optimize.
                </p>
              </SummaryPanel>
            ) : null}
          </div>
        </section>
      </div>
    </QuestionFrame>
  )
}
