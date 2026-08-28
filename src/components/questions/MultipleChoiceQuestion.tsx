import { useState } from 'react'
import { QuestionFrame } from '../QuestionFrame'
import type {
  AttemptOutcome,
  MultipleChoiceQuestionSpec,
  QuestionState,
  Vec2,
} from '../../types'
import {
  multipleChoiceDatasets,
  type MultipleChoiceElbowDataset,
  type MultipleChoiceMetricComparisonDataset,
  type MultipleChoicePointCloudScenariosDataset,
  type MultipleChoiceTableDataset,
} from '../../data/multipleChoiceDatasets'
import { DEFAULT_HINT_SCHEDULE } from '../../lib/assignmentState'

interface MultipleChoiceQuestionProps {
  question: MultipleChoiceQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
}

const VIEW_SIZE = 300
const VIEW_PADDING = 26
const CLUSTER_COLORS = ['#1c7b8a', '#cc7c31', '#708c44', '#7e5da9']

function computeScatterDomain(points: Vec2[], centroids: Vec2[]) {
  const values = [...points, ...centroids]
  const xs = values.map((value) => value[0])
  const ys = values.map((value) => value[1])
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const span = Math.max(maxX - minX, maxY - minY, 1)
  const padding = span * 0.16

  return {
    minX: minX - padding,
    maxX: maxX + padding,
    minY: minY - padding,
    maxY: maxY + padding,
  }
}

function PointCloudPanel({
  title,
  points,
  centroids,
  assignments,
  highlightPointIndices = [],
}: {
  title: string
  points: Vec2[]
  centroids: Vec2[]
  assignments?: number[]
  highlightPointIndices?: number[]
}) {
  const domain = computeScatterDomain(points, centroids)
  const toX = (value: number) =>
    VIEW_PADDING + ((value - domain.minX) / (domain.maxX - domain.minX)) * (VIEW_SIZE - VIEW_PADDING * 2)
  const toY = (value: number) =>
    VIEW_SIZE -
    VIEW_PADDING -
    ((value - domain.minY) / (domain.maxY - domain.minY)) * (VIEW_SIZE - VIEW_PADDING * 2)

  return (
    <article className="scenario-card">
      <h4 className="panel-title">{title}</h4>
      <svg className="projection-stage scenario-stage" viewBox={`0 0 ${VIEW_SIZE} ${VIEW_SIZE}`} role="img" aria-label={title}>
        <line x1={VIEW_PADDING} y1={toY(0)} x2={VIEW_SIZE - VIEW_PADDING} y2={toY(0)} stroke="#d7d0c3" strokeWidth="1.2" />
        <line x1={toX(0)} y1={VIEW_PADDING} x2={toX(0)} y2={VIEW_SIZE - VIEW_PADDING} stroke="#d7d0c3" strokeWidth="1.2" />

        {points.map((point, index) => {
          const assignment = assignments?.[index]
          const fill = assignment === undefined ? '#20313a' : CLUSTER_COLORS[assignment % CLUSTER_COLORS.length]
          const highlighted = highlightPointIndices.includes(index)

          return (
            <g key={`${title}-point-${index}`}>
              {highlighted ? (
                <circle
                  cx={toX(point[0])}
                  cy={toY(point[1])}
                  r="11"
                  fill="none"
                  stroke="#8a5725"
                  strokeWidth="3"
                  opacity="0.85"
                />
              ) : null}
              <circle
                cx={toX(point[0])}
                cy={toY(point[1])}
                r="7.5"
                fill={fill}
                stroke="#122026"
                strokeWidth="1.4"
                opacity="0.95"
              />
            </g>
          )
        })}

        {centroids.map((centroid, index) => (
          <g key={`${title}-centroid-${index}`}>
            <circle
              cx={toX(centroid[0])}
              cy={toY(centroid[1])}
              r="12"
              fill="#fff8ef"
              stroke={CLUSTER_COLORS[index % CLUSTER_COLORS.length]}
              strokeWidth="3.5"
            />
            <text
              x={toX(centroid[0])}
              y={toY(centroid[1]) + 4}
              textAnchor="middle"
              fontSize="12"
              fontWeight="700"
              fill="#20313a"
            >
              {String.fromCharCode(65 + index)}
            </text>
          </g>
        ))}
      </svg>
    </article>
  )
}

function ElbowChart({ dataset }: { dataset: MultipleChoiceElbowDataset }) {
  const width = 340
  const height = 260
  const padding = 34
  const maxObjective = Math.max(...dataset.values.map((value) => value.objective))
  const minObjective = Math.min(...dataset.values.map((value) => value.objective))
  const x = (index: number) => padding + (index / (dataset.values.length - 1)) * (width - padding * 2)
  const y = (objective: number) =>
    height - padding - ((objective - minObjective) / (maxObjective - minObjective)) * (height - padding * 2)

  return (
    <section className="data-table">
      <h3 className="panel-title">Elbow curve</h3>
      <p className="table-caption">{dataset.caption}</p>
      <svg className="elbow-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="K-means elbow curve">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#cfc6b5" strokeWidth="1.5" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#cfc6b5" strokeWidth="1.5" />

        <polyline
          fill="none"
          stroke="#1c7b8a"
          strokeWidth="3.5"
          points={dataset.values.map((value, index) => `${x(index)},${y(value.objective)}`).join(' ')}
        />

        {dataset.values.map((value, index) => (
          <g key={`elbow-${value.k}`}>
            <circle cx={x(index)} cy={y(value.objective)} r="6.5" fill="#cc7c31" />
            <text x={x(index)} y={height - 10} textAnchor="middle" fontSize="12" fontWeight="700" fill="#54636a">
              k={value.k}
            </text>
            <text x={x(index)} y={y(value.objective) - 12} textAnchor="middle" fontSize="12" fill="#54636a">
              {value.objective.toFixed(1)}
            </text>
          </g>
        ))}
      </svg>
    </section>
  )
}

function DatasetPanel({
  dataset,
  resolved,
}: {
  dataset: (typeof multipleChoiceDatasets)[string]
  resolved: boolean
}) {
  if (dataset.kind === 'table') {
    const tableDataset = dataset as MultipleChoiceTableDataset
    return (
      <section className="data-table">
        <h3 className="panel-title">{tableDataset.title ?? 'Feature table'}</h3>
        <p className="table-caption">{tableDataset.caption}</p>
        <table className="table-grid">
          <thead>
            <tr>
              {tableDataset.headers.map((header) => (
                <th key={`${tableDataset.id}-${header}`}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tableDataset.rows.map((row, rowIndex) => (
              <tr key={`${tableDataset.id}-row-${rowIndex}`}>
                {row.map((value, valueIndex) => (
                  <td key={`${tableDataset.id}-cell-${rowIndex}-${valueIndex}`}>{value}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    )
  }

  if (dataset.kind === 'pointCloudScenarios') {
    const scenarioDataset = dataset as MultipleChoicePointCloudScenariosDataset
    return (
      <section className="data-table">
        <h3 className="panel-title">Candidate initializations</h3>
        <p className="table-caption">{scenarioDataset.caption}</p>
        <div className="scenario-grid">
          {scenarioDataset.scenarios.map((scenario) => (
            <PointCloudPanel
              key={scenario.id}
              title={scenario.title}
              points={scenarioDataset.points}
              centroids={scenario.centroids}
            />
          ))}
        </div>
      </section>
    )
  }

  if (dataset.kind === 'metricComparison') {
    const metricDataset = dataset as MultipleChoiceMetricComparisonDataset
    const changedIndices = metricDataset.euclideanAssignments.reduce<number[]>(
      (indices, assignment, index) =>
        assignment !== metricDataset.manhattanAssignments[index] ? [...indices, index] : indices,
      [],
    )

    return (
      <section className="data-table">
        <h3 className="panel-title">Metric comparison</h3>
        <p className="table-caption">{metricDataset.caption}</p>
        <div className="scenario-grid">
          <PointCloudPanel
            title="Euclidean distance"
            points={metricDataset.points}
            centroids={metricDataset.centroids}
            assignments={metricDataset.euclideanAssignments}
          />
          {resolved ? (
            <PointCloudPanel
              title="Manhattan distance"
              points={metricDataset.points}
              centroids={metricDataset.centroids}
              assignments={metricDataset.manhattanAssignments}
              highlightPointIndices={changedIndices}
            />
          ) : null}
        </div>
      </section>
    )
  }

  return <ElbowChart dataset={dataset as MultipleChoiceElbowDataset} />
}

export function MultipleChoiceQuestion({
  question,
  state,
  questionNumber,
  totalQuestions,
  hints,
  onAttempt,
}: MultipleChoiceQuestionProps) {
  const dataset = multipleChoiceDatasets[question.datasetId]
  const [selectedIds, setSelectedIds] = useState<Record<string, string | undefined>>(() => {
    if (
      state.latestAnswer &&
      typeof state.latestAnswer === 'object' &&
      !Array.isArray(state.latestAnswer) &&
      state.latestAnswer.selectedIds &&
      typeof state.latestAnswer.selectedIds === 'object' &&
      !Array.isArray(state.latestAnswer.selectedIds)
    ) {
      return state.latestAnswer.selectedIds as Record<string, string | undefined>
    }
    return {}
  })
  const [feedback, setFeedback] = useState('')
  const showComparison = state.status === 'correct' || state.status === 'gave_up'
  const schedule = question.hintSchedule ?? DEFAULT_HINT_SCHEDULE
  const activeSelections = selectedIds

  const checkChoice = () => {
    const result = question.validator({ selectedIds: activeSelections })
    if (result.correct) {
      setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
      onAttempt('correct', { selectedIds: activeSelections })
      return
    }

    const nextIncorrectAttempts = state.incorrectAttempts + 1
    setFeedback(
      schedule.includes(nextIncorrectAttempts)
        ? 'Not yet. A new hint appeared below.'
        : result.message ?? 'That choice is not right yet.',
    )
    onAttempt('incorrect', { selectedIds: activeSelections })
  }

  return (
    <QuestionFrame
      question={question}
      questionNumber={questionNumber}
      totalQuestions={totalQuestions}
      state={state}
      feedback={feedback}
      hints={hints}
      controls={
        <button
          type="button"
          className="button"
          onClick={checkChoice}
          disabled={question.parts.some((part) => !activeSelections[part.id])}
        >
          Check answer
        </button>
      }
    >
      <div className="table-shell">
        <DatasetPanel dataset={dataset} resolved={showComparison} />

        <section className="answer-panel">
          <h3 className="panel-title">
            {question.parts.length === 1
              ? 'Choose one answer'
              : question.parts.length === 2
                ? 'Answer both parts'
                : 'Answer all parts'}
          </h3>
          {question.parts.map((part) => (
            <div key={part.id} className="substep-shell">
              <div className="substep-header">
                <h3 className="substep-title">{part.prompt}</h3>
              </div>
              <div className="choice-list">
                {part.options.map((option) => (
                  <label
                    key={option.id}
                    className={`choice-option ${activeSelections[part.id] === option.id ? 'selected' : ''}`}
                  >
                    <input
                      type="radio"
                      name={`${question.id}-${part.id}`}
                      checked={activeSelections[part.id] === option.id}
                      onChange={() =>
                        setSelectedIds((current) => ({
                          ...current,
                          [part.id]: option.id,
                        }))
                      }
                    />
                    <span className="choice-copy">
                      <strong>{option.title}</strong>
                      <span>{option.description}</span>
                    </span>
                  </label>
                ))}
              </div>
            </div>
          ))}
        </section>
      </div>
    </QuestionFrame>
  )
}
