import { useState, type ReactNode } from 'react'
import { QuestionFrame } from '../QuestionFrame'
import type {
  AttemptOutcome,
  DecisionTreeQuestionSpec,
  QuestionState,
} from '../../types'
import {
  decisionTreeDatasets,
  type DecisionTreeBestSplitDataset,
  type DecisionTreeDataset,
  type DecisionTreeGiniWarmupDataset,
  type DecisionTreeRow,
  type DecisionTreeSplit,
  type DecisionTreeSplitScoreDataset,
  type DecisionTreeTableDataset,
  type DecisionTreeVisualDepthTwoDataset,
  type DecisionTreeVisualSplitDataset,
} from '../../data/decisionTreeDatasets'
import { DEFAULT_HINT_SCHEDULE } from '../../lib/assignmentState'
import {
  evaluateSplit,
  formatGini,
  formatThreshold,
  splitLabel,
} from '../../lib/decisionTreeMath'

interface DecisionTreeQuestionProps {
  question: DecisionTreeQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
}

const STAGE_SIZE = 420
const STAGE_PADDING = 34

function parseNumber(value: string) {
  const parsed = Number(value)
  return value.trim() === '' || Number.isNaN(parsed) ? null : parsed
}

function getFeatureLabel(dataset: Pick<DecisionTreeDataset, 'features'>, featureId: string) {
  return dataset.features.find((feature) => feature.id === featureId)?.label ?? featureId
}

function getClassLabel(dataset: Pick<DecisionTreeDataset, 'classes'>, label: 0 | 1) {
  return dataset.classes[label]?.label ?? String(label)
}

function DataTable({ dataset, rows }: { dataset: DecisionTreeTableDataset; rows: DecisionTreeRow[] }) {
  return (
    <section className="data-table">
      <h3 className="panel-title">Training data</h3>
      <table className="table-grid">
        <thead>
          <tr>
            <th>Row</th>
            {dataset.features.map((feature) => (
              <th key={`${dataset.id}-${feature.id}`}>{feature.label}</th>
            ))}
            <th>Outcome</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${dataset.id}-row-${row.id}`}>
              <td>{row.id}</td>
              {dataset.features.map((feature) => (
                <td key={`${dataset.id}-${row.id}-${feature.id}`}>
                  {formatThreshold(row.features[feature.id])}
                </td>
              ))}
              <td>{getClassLabel(dataset, row.label)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

function SplitButton({
  dataset,
  split,
  selected,
  disabled,
  onSelect,
}: {
  dataset: DecisionTreeDataset
  split: DecisionTreeSplit
  selected: boolean
  disabled: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      className={`class-choice ${selected ? 'selected' : ''}`}
      onClick={onSelect}
      disabled={disabled}
      aria-pressed={selected}
      style={{
        borderColor: selected ? '#1c7b8a' : 'rgba(23, 34, 40, 0.08)',
        background: selected ? 'rgba(229, 243, 245, 0.92)' : undefined,
      }}
    >
      {splitLabel(split, getFeatureLabel(dataset, split.featureId))}
    </button>
  )
}

function CandidateSplitPanel({
  title,
  dataset,
  rows,
  splits,
  selectedId,
  disabled,
  showScores,
  onSelect,
}: {
  title: string
  dataset: DecisionTreeDataset
  rows: DecisionTreeRow[]
  splits: DecisionTreeSplit[]
  selectedId: string | undefined
  disabled: boolean
  showScores: boolean
  onSelect: (splitId: string | undefined) => void
}) {
  return (
    <section className="answer-panel">
      <h3 className="panel-title">{title}</h3>
      <div className="class-picker decision-tree-candidates">
        {splits.map((split) => {
          const score = evaluateSplit(rows, split)
          return (
            <div key={`${title}-${split.id}`} className="decision-tree-candidate">
              <SplitButton
                dataset={dataset}
                split={split}
                selected={selectedId === split.id}
                disabled={disabled}
                onSelect={() => onSelect(split.id)}
              />
              {showScores ? (
                <p className="panel-note">weighted Gini: {formatGini(score.weightedGini)}</p>
              ) : null}
            </div>
          )
        })}
      </div>
    </section>
  )
}

function NodeGiniWarmup({
  dataset,
  inputs,
  disabled,
  onChange,
}: {
  dataset: DecisionTreeGiniWarmupDataset
  inputs: Record<string, string>
  disabled: boolean
  onChange: (nodeId: string, value: string) => void
}) {
  return (
    <div className="table-shell">
      <section className="data-table">
        <h3 className="panel-title">Node counts</h3>
        <table className="table-grid">
          <thead>
            <tr>
              <th>Node</th>
              <th>Passed</th>
              <th>Did not pass</th>
            </tr>
          </thead>
          <tbody>
            {dataset.nodes.map((node) => (
              <tr key={`${dataset.id}-${node.id}`}>
                <td>{node.label}</td>
                <td>{node.counts.positive}</td>
                <td>{node.counts.negative}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="answer-panel">
        <h3 className="panel-title">Gini values</h3>
        <div className="decision-tree-input-grid">
          {dataset.nodes.map((node) => (
            <label key={`${dataset.id}-answer-${node.id}`} className="decision-tree-input-label">
              <span>{node.label}</span>
              <input
                className="numeric-input"
                aria-label={`${node.label} Gini`}
                value={inputs[node.id] ?? ''}
                onChange={(event) => onChange(node.id, event.target.value)}
                disabled={disabled}
              />
            </label>
          ))}
        </div>
      </section>
    </div>
  )
}

interface ChildCountInputs {
  leftNegative: string
  leftPositive: string
  rightNegative: string
  rightPositive: string
}

function SplitScoreInputs({
  dataset,
  stepIndex,
  selectedLeftRowIds,
  countInputs,
  leftInput,
  rightInput,
  weightedInput,
  disabled,
  onToggleLeftRow,
  onCountChange,
  onLeftChange,
  onRightChange,
  onWeightedChange,
}: {
  dataset: DecisionTreeSplitScoreDataset
  stepIndex: number
  selectedLeftRowIds: string[]
  countInputs: ChildCountInputs
  leftInput: string
  rightInput: string
  weightedInput: string
  disabled: boolean
  onToggleLeftRow: (rowId: string) => void
  onCountChange: (field: keyof ChildCountInputs, value: string) => void
  onLeftChange: (value: string) => void
  onRightChange: (value: string) => void
  onWeightedChange: (value: string) => void
}) {
  return (
    <div className="table-shell">
      <DataTable dataset={dataset} rows={dataset.rows} />
      <section className="answer-panel">
        <h3 className="panel-title">
          Proposed split: {splitLabel(dataset.split, getFeatureLabel(dataset, dataset.split.featureId))}
        </h3>

        {stepIndex === 0 ? (
          <>
            <p className="panel-note">
              Select every row sent to the left child. Unselected rows go right.
            </p>
            <div className="class-picker" role="group" aria-label="Rows sent to the left child">
              {dataset.rows.map((row) => {
                const selected = selectedLeftRowIds.includes(row.id)
                return (
                  <button
                    key={`${dataset.id}-partition-${row.id}`}
                    type="button"
                    className={`class-choice ${selected ? 'selected' : ''}`}
                    onClick={() => onToggleLeftRow(row.id)}
                    disabled={disabled}
                    aria-pressed={selected}
                    style={{
                      borderColor: selected ? '#1c7b8a' : 'rgba(23, 34, 40, 0.08)',
                      background: selected ? 'rgba(229, 243, 245, 0.92)' : undefined,
                    }}
                  >
                    Row {row.id}
                  </button>
                )
              })}
            </div>
          </>
        ) : null}

        {stepIndex === 1 ? (
          <>
            <p className="panel-note">
              Count each outcome in your two children, then compute their Gini impurities.
            </p>
            <div className="decision-tree-input-grid two-column">
              <label className="decision-tree-input-label">
                <span>Left: did not pass</span>
                <input
                  className="numeric-input"
                  aria-label="Left child did not pass count"
                  value={countInputs.leftNegative}
                  onChange={(event) => onCountChange('leftNegative', event.target.value)}
                  disabled={disabled}
                />
              </label>
              <label className="decision-tree-input-label">
                <span>Left: passed</span>
                <input
                  className="numeric-input"
                  aria-label="Left child passed count"
                  value={countInputs.leftPositive}
                  onChange={(event) => onCountChange('leftPositive', event.target.value)}
                  disabled={disabled}
                />
              </label>
              <label className="decision-tree-input-label">
                <span>Right: did not pass</span>
                <input
                  className="numeric-input"
                  aria-label="Right child did not pass count"
                  value={countInputs.rightNegative}
                  onChange={(event) => onCountChange('rightNegative', event.target.value)}
                  disabled={disabled}
                />
              </label>
              <label className="decision-tree-input-label">
                <span>Right: passed</span>
                <input
                  className="numeric-input"
                  aria-label="Right child passed count"
                  value={countInputs.rightPositive}
                  onChange={(event) => onCountChange('rightPositive', event.target.value)}
                  disabled={disabled}
                />
              </label>
              <label className="decision-tree-input-label">
                <span>Left child Gini</span>
                <input
                  className="numeric-input"
                  aria-label="Left child Gini"
                  value={leftInput}
                  onChange={(event) => onLeftChange(event.target.value)}
                  disabled={disabled}
                />
              </label>
              <label className="decision-tree-input-label">
                <span>Right child Gini</span>
                <input
                  className="numeric-input"
                  aria-label="Right child Gini"
                  value={rightInput}
                  onChange={(event) => onRightChange(event.target.value)}
                  disabled={disabled}
                />
              </label>
            </div>
          </>
        ) : null}

        {stepIndex === 2 ? (
          <label className="decision-tree-input-label">
            <span>Weighted split Gini</span>
            <input
              className="numeric-input"
              aria-label="Weighted split Gini"
              value={weightedInput}
              onChange={(event) => onWeightedChange(event.target.value)}
              disabled={disabled}
            />
          </label>
        ) : null}
      </section>
    </div>
  )
}

function VisualSplitPanel({
  dataset,
  featureId,
  threshold,
  disabled,
  showScore,
  onFeatureChange,
  onThresholdChange,
}: {
  dataset: DecisionTreeVisualSplitDataset
  featureId: string
  threshold: number
  disabled: boolean
  showScore: boolean
  onFeatureChange: (featureId: string) => void
  onThresholdChange: (threshold: number) => void
}) {
  const toX = (value: number) =>
    STAGE_PADDING +
    ((value - dataset.bounds.minX) / (dataset.bounds.maxX - dataset.bounds.minX)) *
      (STAGE_SIZE - STAGE_PADDING * 2)
  const toY = (value: number) =>
    STAGE_SIZE -
    STAGE_PADDING -
    ((value - dataset.bounds.minY) / (dataset.bounds.maxY - dataset.bounds.minY)) *
      (STAGE_SIZE - STAGE_PADDING * 2)
  const split = evaluateSplit(dataset.rows, { featureId, threshold })
  const min = featureId === 'x' ? dataset.bounds.minX : dataset.bounds.minY
  const max = featureId === 'x' ? dataset.bounds.maxX : dataset.bounds.maxY
  const featureLabel = getFeatureLabel(dataset, featureId)

  return (
    <div className="visual-grid">
      <section className="visual-panel">
        <h3 className="panel-title">{dataset.label}</h3>
        <div className="knn-toolbar">
          <div className="class-picker" role="radiogroup" aria-label="Choose split direction">
            {dataset.features.map((feature) => (
              <button
                key={`${dataset.id}-axis-${feature.id}`}
                type="button"
                className={`class-choice ${featureId === feature.id ? 'selected' : ''}`}
                onClick={() => onFeatureChange(feature.id)}
                disabled={disabled}
                style={{
                  borderColor: featureId === feature.id ? '#1c7b8a' : 'rgba(23, 34, 40, 0.08)',
                  background: featureId === feature.id ? 'rgba(229, 243, 245, 0.92)' : undefined,
                }}
              >
                {feature.id === 'x' ? 'Vertical split' : 'Horizontal split'}
              </button>
            ))}
          </div>
          <label className="slider-shell" htmlFor={`${dataset.id}-threshold`}>
            <span>{featureLabel}</span>
            <input
              id={`${dataset.id}-threshold`}
              className="range-input"
              type="range"
              min={min}
              max={max}
              step="0.5"
              value={threshold}
              onChange={(event) => onThresholdChange(Number(event.target.value))}
              disabled={disabled}
            />
            <span className="question-index-chip">{formatThreshold(threshold)}</span>
          </label>
        </div>

        <svg className="svg-stage knn-stage" viewBox={`0 0 ${STAGE_SIZE} ${STAGE_SIZE}`} role="img" aria-label={dataset.label}>
          <rect x="0" y="0" width={STAGE_SIZE} height={STAGE_SIZE} rx="18" fill="transparent" />
          <line x1={STAGE_PADDING} y1={toY(0)} x2={STAGE_SIZE - STAGE_PADDING} y2={toY(0)} stroke="#d7d0c3" strokeWidth="1.2" />
          <line x1={toX(0)} y1={STAGE_PADDING} x2={toX(0)} y2={STAGE_SIZE - STAGE_PADDING} stroke="#d7d0c3" strokeWidth="1.2" />

          {featureId === 'x' ? (
            <line
              x1={toX(threshold)}
              y1={STAGE_PADDING}
              x2={toX(threshold)}
              y2={STAGE_SIZE - STAGE_PADDING}
              stroke="#142026"
              strokeWidth="3"
              strokeDasharray="8 7"
            />
          ) : (
            <line
              x1={STAGE_PADDING}
              y1={toY(threshold)}
              x2={STAGE_SIZE - STAGE_PADDING}
              y2={toY(threshold)}
              stroke="#142026"
              strokeWidth="3"
              strokeDasharray="8 7"
            />
          )}

          {dataset.rows.map((row) => (
            <g key={`${dataset.id}-point-${row.id}`}>
              <title>
                {`Row ${row.id}: x ${formatThreshold(row.features.x)}, y ${formatThreshold(row.features.y)}, ${getClassLabel(dataset, row.label)}`}
              </title>
              <circle
                cx={toX(row.features.x)}
                cy={toY(row.features.y)}
                r="8"
                fill={dataset.classes[row.label].color}
                stroke="#122026"
                strokeWidth="1.4"
              />
            </g>
          ))}
        </svg>
      </section>

      <section className="answer-panel">
        {showScore ? (
          <>
            <h3 className="panel-title">Resolved split score</h3>
            <div className="accuracy-grid">
              <div className="metric-card">
                <span className="metric-value">{formatGini(split.left.gini)}</span>
                <span className="metric-copy">Left Gini</span>
              </div>
              <div className="metric-card">
                <span className="metric-value">{formatGini(split.right.gini)}</span>
                <span className="metric-copy">Right Gini</span>
              </div>
              <div className="metric-card">
                <span className="metric-value">{formatGini(split.weightedGini)}</span>
                <span className="metric-copy">Weighted Gini</span>
              </div>
            </div>
          </>
        ) : (
          <>
            <h3 className="panel-title">Score hidden</h3>
            <p className="panel-note">
              Count the two colors on each side of your line. The child and weighted Gini values
              appear after you submit.
            </p>
            <div className="class-picker" aria-label="Class color legend">
              {dataset.classes.map((classSpec) => (
                <span key={`${dataset.id}-${classSpec.label}`} className="pill">
                  <span
                    aria-hidden="true"
                    style={{
                      display: 'inline-block',
                      width: '0.7rem',
                      height: '0.7rem',
                      marginRight: '0.35rem',
                      borderRadius: '50%',
                      background: classSpec.color,
                    }}
                  />
                  {classSpec.label}
                </span>
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  )
}

function VisualDepthTwoPanel({
  dataset,
  selectedRootSplitId,
  selectedChildSplitId,
  disabled,
  onRootSelect,
  onChildSelect,
}: {
  dataset: DecisionTreeVisualDepthTwoDataset
  selectedRootSplitId: string | undefined
  selectedChildSplitId: string | undefined
  disabled: boolean
  onRootSelect: (splitId: string | undefined) => void
  onChildSelect: (splitId: string | undefined) => void
}) {
  const toX = (value: number) =>
    STAGE_PADDING +
    ((value - dataset.bounds.minX) / (dataset.bounds.maxX - dataset.bounds.minX)) *
      (STAGE_SIZE - STAGE_PADDING * 2)
  const toY = (value: number) =>
    STAGE_SIZE -
    STAGE_PADDING -
    ((value - dataset.bounds.minY) / (dataset.bounds.maxY - dataset.bounds.minY)) *
      (STAGE_SIZE - STAGE_PADDING * 2)
  const rootSplit = dataset.rootCandidates.find((split) => split.id === selectedRootSplitId)
  const childSplit = dataset.rightChildCandidates.find(
    (split) => split.id === selectedChildSplitId,
  )
  const rootEvaluation = rootSplit ? evaluateSplit(dataset.rows, rootSplit) : undefined
  const childRows = rootEvaluation?.right.rows ?? []
  const childEvaluation = childSplit ? evaluateSplit(childRows, childSplit) : undefined

  const childLine = (() => {
    if (!rootSplit || !childSplit) {
      return null
    }

    if (rootSplit.featureId === 'x') {
      return childSplit.featureId === 'x' ? (
        <line
          x1={toX(childSplit.threshold)}
          y1={STAGE_PADDING}
          x2={toX(childSplit.threshold)}
          y2={STAGE_SIZE - STAGE_PADDING}
          stroke="#b36a28"
          strokeWidth="3"
          strokeDasharray="5 6"
        />
      ) : (
        <line
          x1={toX(rootSplit.threshold)}
          y1={toY(childSplit.threshold)}
          x2={STAGE_SIZE - STAGE_PADDING}
          y2={toY(childSplit.threshold)}
          stroke="#b36a28"
          strokeWidth="3"
          strokeDasharray="5 6"
        />
      )
    }

    return childSplit.featureId === 'x' ? (
      <line
        x1={toX(childSplit.threshold)}
        y1={STAGE_PADDING}
        x2={toX(childSplit.threshold)}
        y2={toY(rootSplit.threshold)}
        stroke="#b36a28"
        strokeWidth="3"
        strokeDasharray="5 6"
      />
    ) : (
      <line
        x1={STAGE_PADDING}
        y1={toY(childSplit.threshold)}
        x2={STAGE_SIZE - STAGE_PADDING}
        y2={toY(childSplit.threshold)}
        stroke="#b36a28"
        strokeWidth="3"
        strokeDasharray="5 6"
      />
    )
  })()

  return (
    <div className="visual-grid">
      <section className="visual-panel">
        <h3 className="panel-title">{dataset.label}</h3>
        <svg
          className="svg-stage knn-stage"
          viewBox={`0 0 ${STAGE_SIZE} ${STAGE_SIZE}`}
          role="img"
          aria-label={dataset.label}
        >
          <rect x="0" y="0" width={STAGE_SIZE} height={STAGE_SIZE} rx="18" fill="transparent" />
          <line
            x1={STAGE_PADDING}
            y1={toY(0)}
            x2={STAGE_SIZE - STAGE_PADDING}
            y2={toY(0)}
            stroke="#d7d0c3"
            strokeWidth="1.2"
          />
          <line
            x1={toX(0)}
            y1={STAGE_PADDING}
            x2={toX(0)}
            y2={STAGE_SIZE - STAGE_PADDING}
            stroke="#d7d0c3"
            strokeWidth="1.2"
          />

          {rootSplit ? (
            rootSplit.featureId === 'x' ? (
              <line
                x1={toX(rootSplit.threshold)}
                y1={STAGE_PADDING}
                x2={toX(rootSplit.threshold)}
                y2={STAGE_SIZE - STAGE_PADDING}
                stroke="#142026"
                strokeWidth="3"
                strokeDasharray="8 7"
              />
            ) : (
              <line
                x1={STAGE_PADDING}
                y1={toY(rootSplit.threshold)}
                x2={STAGE_SIZE - STAGE_PADDING}
                y2={toY(rootSplit.threshold)}
                stroke="#142026"
                strokeWidth="3"
                strokeDasharray="8 7"
              />
            )
          ) : null}
          {childLine}

          {dataset.rows.map((row) => (
            <g key={`${dataset.id}-point-${row.id}`}>
              <title>
                {`Row ${row.id}: x ${formatThreshold(row.features.x)}, y ${formatThreshold(row.features.y)}, ${getClassLabel(dataset, row.label)}`}
              </title>
              <circle
                cx={toX(row.features.x)}
                cy={toY(row.features.y)}
                r="8"
                fill={dataset.classes[row.label].color}
                stroke="#122026"
                strokeWidth="1.4"
              />
            </g>
          ))}
        </svg>

        <div className="class-picker" aria-label="Split line legend">
          <span className="pill">Dark line: root</span>
          <span className="pill">Gold line: right child only</span>
        </div>
        {disabled && rootEvaluation && childEvaluation ? (
          <div className="accuracy-grid">
            <div className="metric-card">
              <span className="metric-value">{formatGini(rootEvaluation.weightedGini)}</span>
              <span className="metric-copy">Root weighted Gini</span>
            </div>
            <div className="metric-card">
              <span className="metric-value">{formatGini(childEvaluation.weightedGini)}</span>
              <span className="metric-copy">Right-child weighted Gini</span>
            </div>
          </div>
        ) : (
          <p className="panel-note">Impurity scores appear after you submit.</p>
        )}
      </section>

      <div className="decision-tree-stack">
        <CandidateSplitPanel
          title="1. Root split"
          dataset={dataset}
          rows={dataset.rows}
          splits={dataset.rootCandidates}
          selectedId={selectedRootSplitId}
          disabled={disabled}
          showScores={disabled}
          onSelect={onRootSelect}
        />
        <CandidateSplitPanel
          title="2. Split the right child"
          dataset={dataset}
          rows={childRows}
          splits={dataset.rightChildCandidates}
          selectedId={selectedChildSplitId}
          disabled={disabled || !rootSplit}
          showScores={disabled}
          onSelect={onChildSelect}
        />
      </div>
    </div>
  )
}

export function DecisionTreeQuestion({
  question,
  state,
  questionNumber,
  totalQuestions,
  hints,
  onAttempt,
}: DecisionTreeQuestionProps) {
  const dataset = decisionTreeDatasets[question.datasetId]
  const [feedback, setFeedback] = useState('')
  const [stepIndex, setStepIndex] = useState(0)
  const [giniInputs, setGiniInputs] = useState<Record<string, string>>({})
  const [selectedLeftRowIds, setSelectedLeftRowIds] = useState<string[]>([])
  const [childCountInputs, setChildCountInputs] = useState<ChildCountInputs>({
    leftNegative: '',
    leftPositive: '',
    rightNegative: '',
    rightPositive: '',
  })
  const [leftGiniInput, setLeftGiniInput] = useState('')
  const [rightGiniInput, setRightGiniInput] = useState('')
  const [weightedGiniInput, setWeightedGiniInput] = useState('')
  const [selectedSplitId, setSelectedSplitId] = useState<string | undefined>()
  const [selectedRootSplitId, setSelectedRootSplitId] = useState<string | undefined>()
  const [selectedChildSplitId, setSelectedChildSplitId] = useState<string | undefined>()
  const [visualFeatureId, setVisualFeatureId] = useState('x')
  const [visualThreshold, setVisualThreshold] = useState(4.5)
  const showAnswer = state.status === 'correct' || state.status === 'gave_up'
  const schedule = question.hintSchedule ?? DEFAULT_HINT_SCHEDULE

  const splitScoreSubmission = (step: 'partition' | 'childStats' | 'weightedGini') => ({
    step,
    leftRowIds: [...selectedLeftRowIds].sort(),
    counts: {
      left: {
        negative: parseNumber(childCountInputs.leftNegative),
        positive: parseNumber(childCountInputs.leftPositive),
      },
      right: {
        negative: parseNumber(childCountInputs.rightNegative),
        positive: parseNumber(childCountInputs.rightPositive),
      },
    },
    values: {
      left: parseNumber(leftGiniInput),
      right: parseNumber(rightGiniInput),
    },
    value: parseNumber(weightedGiniInput),
  })

  const invalidAttempt = (message: string, answer: unknown) => {
    const nextIncorrectAttempts = state.incorrectAttempts + 1
    setFeedback(schedule.includes(nextIncorrectAttempts) ? 'Not yet. A new hint appeared below.' : message)
    onAttempt('incorrect', answer)
  }

  const checkWork = () => {
    if (question.interactionMode === 'giniWarmup') {
      const values = Object.fromEntries(
        dataset.kind === 'giniWarmup'
          ? dataset.nodes.map((node) => [node.id, parseNumber(giniInputs[node.id] ?? '')])
          : [],
      )
      const submission = { values }
      const result = question.validator(submission)
      if (!result.correct) {
        invalidAttempt(result.message ?? 'At least one Gini value is not correct yet.', submission)
        return
      }
      setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
      onAttempt('correct', submission)
      return
    }

    if (question.interactionMode === 'splitScore') {
      if (stepIndex === 0) {
        const submission = splitScoreSubmission('partition')
        const result = question.validator(submission)
        if (!result.correct) {
          invalidAttempt(result.message ?? 'The row partition is not correct yet.', submission)
          return
        }
        setFeedback('Row routing confirmed. Now derive the class counts and child Gini values.')
        setStepIndex(1)
        onAttempt('progress', submission)
        return
      }

      if (stepIndex === 1) {
        const submission = splitScoreSubmission('childStats')
        const result = question.validator(submission)
        if (!result.correct) {
          invalidAttempt(result.message ?? 'The child counts or Gini values are not correct yet.', submission)
          return
        }
        setFeedback('Child statistics confirmed. Now combine them using the child sizes.')
        setStepIndex(2)
        onAttempt('progress', submission)
        return
      }

      const submission = splitScoreSubmission('weightedGini')
      const result = question.validator(submission)
      if (!result.correct) {
        invalidAttempt(result.message ?? 'The weighted Gini is not correct yet.', submission)
        return
      }
      setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
      onAttempt('correct', submission)
      return
    }

    if (question.interactionMode === 'visualSplit') {
      const submission = {
        featureId: visualFeatureId,
        threshold: visualThreshold,
      }
      const result = question.validator(submission)
      if (!result.correct) {
        invalidAttempt(result.message ?? 'That visual split is not correct yet.', submission)
        return
      }
      setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
      onAttempt('correct', submission)
      return
    }

    if (question.interactionMode === 'depthTwo') {
      const submission = {
        rootSplitId: selectedRootSplitId ?? null,
        rightChildSplitId: selectedChildSplitId ?? null,
      }
      const result = question.validator(submission)
      if (!result.correct) {
        invalidAttempt(result.message ?? 'That tree is not complete yet.', submission)
        return
      }
      setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
      onAttempt('correct', submission)
      return
    }

    const submission = { splitId: selectedSplitId ?? null }
    const result = question.validator(submission)
    if (!result.correct) {
      invalidAttempt(result.message ?? 'That split is not correct yet.', submission)
      return
    }
    setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
    onAttempt('correct', submission)
  }

  let checkButtonLabel = 'Check answer'
  let body = null as ReactNode
  let statusPills: ReactNode[] = []

  if (dataset.kind === 'giniWarmup') {
    checkButtonLabel = 'Check Gini values'
    body = (
      <NodeGiniWarmup
        dataset={dataset}
        inputs={giniInputs}
        disabled={false}
        onChange={(nodeId, value) =>
          setGiniInputs((current) => ({ ...current, [nodeId]: value }))
        }
      />
    )
  }

  if (dataset.kind === 'splitScore') {
    checkButtonLabel =
      stepIndex === 0
        ? 'Check row routing'
        : stepIndex === 1
          ? 'Check child statistics'
          : 'Check weighted Gini'
    body = (
      <SplitScoreInputs
        dataset={dataset}
        stepIndex={stepIndex}
        selectedLeftRowIds={selectedLeftRowIds}
        countInputs={childCountInputs}
        leftInput={leftGiniInput}
        rightInput={rightGiniInput}
        weightedInput={weightedGiniInput}
        disabled={false}
        onToggleLeftRow={(rowId) =>
          setSelectedLeftRowIds((current) =>
            current.includes(rowId)
              ? current.filter((candidateId) => candidateId !== rowId)
              : [...current, rowId],
          )
        }
        onCountChange={(field, value) =>
          setChildCountInputs((current) => ({ ...current, [field]: value }))
        }
        onLeftChange={setLeftGiniInput}
        onRightChange={setRightGiniInput}
        onWeightedChange={setWeightedGiniInput}
      />
    )
    statusPills = [<span key="step" className="pill">Step {stepIndex + 1} / 3</span>]
  }

  if (dataset.kind === 'bestThreshold' || dataset.kind === 'bestRootSplit') {
    const bestDataset = dataset as DecisionTreeBestSplitDataset
    body = (
      <div className="table-shell">
        <DataTable dataset={bestDataset} rows={bestDataset.rows} />
        <CandidateSplitPanel
          title="Candidate splits"
          dataset={bestDataset}
          rows={bestDataset.rows}
          splits={bestDataset.candidateSplits}
          selectedId={selectedSplitId}
          disabled={false}
          showScores={showAnswer}
          onSelect={setSelectedSplitId}
        />
      </div>
    )
  }

  if (dataset.kind === 'visualSplit') {
    body = (
      <VisualSplitPanel
        dataset={dataset}
        featureId={visualFeatureId}
        threshold={visualThreshold}
        disabled={false}
        showScore={showAnswer}
        onFeatureChange={(nextFeatureId) => {
          setVisualFeatureId(nextFeatureId)
          setVisualThreshold(nextFeatureId === 'x' ? 4.5 : 2.5)
        }}
        onThresholdChange={setVisualThreshold}
      />
    )
  }

  if (dataset.kind === 'visualDepthTwo') {
    body = (
      <VisualDepthTwoPanel
        dataset={dataset}
        selectedRootSplitId={selectedRootSplitId}
        selectedChildSplitId={selectedChildSplitId}
        disabled={false}
        onRootSelect={(splitId) => {
          setSelectedRootSplitId(splitId)
          setSelectedChildSplitId(undefined)
        }}
        onChildSelect={setSelectedChildSplitId}
      />
    )
  }

  if (dataset.kind === 'depthTwo') {
    const selectedRoot = dataset.rootCandidates.find((split) => split.id === selectedRootSplitId)
    const rightChildRows = selectedRoot
      ? evaluateSplit(dataset.rows, selectedRoot).right.rows
      : []
    body = (
      <div className="table-shell">
        <DataTable dataset={dataset} rows={dataset.rows} />
        <div className="decision-tree-stack">
          <CandidateSplitPanel
            title="Root split"
            dataset={dataset}
            rows={dataset.rows}
            splits={dataset.rootCandidates}
            selectedId={selectedRootSplitId}
            disabled={false}
            showScores={showAnswer}
            onSelect={(splitId) => {
              setSelectedRootSplitId(splitId)
              setSelectedChildSplitId(undefined)
            }}
          />
          <CandidateSplitPanel
            title="Right child split"
            dataset={dataset}
            rows={rightChildRows}
            splits={dataset.rightChildCandidates}
            selectedId={selectedChildSplitId}
            disabled={!selectedRoot}
            showScores={showAnswer}
            onSelect={setSelectedChildSplitId}
          />
        </div>
      </div>
    )
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
        <>
          <button type="button" className="button" onClick={checkWork}>
            {checkButtonLabel}
          </button>
          {statusPills}
        </>
      }
    >
      {body}
    </QuestionFrame>
  )
}
