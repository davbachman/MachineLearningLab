import { useState, type CSSProperties, type KeyboardEvent } from 'react'
import { QuestionFrame } from '../QuestionFrame'
import type {
  AttemptOutcome,
  QuestionState,
  RandomForestQuestionSpec,
} from '../../types'
import {
  randomForestDatasets,
  type RandomForestBootstrapDataset,
  type RandomForestClassSpec,
  type RandomForestCodeTraceDataset,
  type RandomForestFeatureGeometryDataset,
  type RandomForestLabel,
  type RandomForestOobDataset,
  type RandomForestPlotBounds,
  type RandomForestPoint,
  type RandomForestSplitCandidate,
  type RandomForestVarianceDataset,
  type RandomForestVoteGeometryDataset,
} from '../../data/randomForestDatasets'
import {
  aggregateOobPredictions,
  bootstrapMultiplicities,
  correlatedEnsembleVariance,
  findBestStump,
  majorityVote,
  oobAccuracy,
  outOfBagIds,
  predictForest,
  predictStump,
  trainedStumpFromEvaluation,
} from '../../lib/randomForestMath'
import { DEFAULT_HINT_SCHEDULE } from '../../lib/assignmentState'

interface RandomForestQuestionProps {
  question: RandomForestQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
  onGiveUp: (answer: unknown) => void
}

interface PlotLine extends RandomForestSplitCandidate {
  color: string
  label: string
}

const STAGE_SIZE = 430
const STAGE_PADDING = 38
const FOREST_LINE_COLORS = ['#8f5f32', '#725fa3', '#3f7f6f', '#b65e5e', '#3c668d']

const compactGridStyle: CSSProperties = {
  display: 'grid',
  gap: '0.8rem',
}

const splitGridStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(245px, 1fr))',
  gap: '1rem',
}

function parseNumber(value: string) {
  const parsed = Number(value)
  return value.trim() === '' || Number.isNaN(parsed) ? undefined : parsed
}

function parseNumberList(value: string): Array<number | string> {
  if (!value.trim()) {
    return []
  }

  return value
    .split(/[\s,]+/)
    .filter(Boolean)
    .map((token) => {
      const parsed = Number(token)
      return Number.isFinite(parsed) ? parsed : token
    })
}

function formatRule(split: RandomForestSplitCandidate) {
  return `${split.feature} <= ${split.threshold}`
}

function toggleId(ids: string[], id: string) {
  return ids.includes(id) ? ids.filter((value) => value !== id) : [...ids, id]
}

function ClassButtons({
  classes,
  value,
  label,
  disabled,
  onChange,
}: {
  classes: RandomForestClassSpec[]
  value: RandomForestLabel | undefined
  label: string
  disabled: boolean
  onChange: (label: RandomForestLabel) => void
}) {
  return (
    <div className="class-picker" role="radiogroup" aria-label={label}>
      {classes.map((classSpec, classIndex) => (
        <button
          key={`${label}-${classIndex}`}
          type="button"
          className={`class-choice ${value === classIndex ? 'selected' : ''}`}
          aria-pressed={value === classIndex}
          disabled={disabled}
          onClick={() => onChange(classIndex as RandomForestLabel)}
          style={{
            borderColor: value === classIndex ? classSpec.color : 'rgba(23, 34, 40, 0.08)',
            background: value === classIndex ? `${classSpec.color}22` : undefined,
          }}
        >
          {classSpec.label}
        </button>
      ))}
    </div>
  )
}

function ForestPlot({
  label,
  rows,
  probes,
  bounds,
  classes,
  lines,
  predictions = {},
  disabled,
  onProbeClick,
}: {
  label: string
  rows: RandomForestPoint[]
  probes: Array<Pick<RandomForestPoint, 'id' | 'x' | 'y'>>
  bounds: RandomForestPlotBounds
  classes: RandomForestClassSpec[]
  lines: PlotLine[]
  predictions?: Record<string, RandomForestLabel | undefined>
  disabled: boolean
  onProbeClick?: (probeId: string) => void
}) {
  const toX = (value: number) =>
    STAGE_PADDING +
    ((value - bounds.minX) / (bounds.maxX - bounds.minX)) *
      (STAGE_SIZE - STAGE_PADDING * 2)
  const toY = (value: number) =>
    STAGE_SIZE -
    STAGE_PADDING -
    ((value - bounds.minY) / (bounds.maxY - bounds.minY)) *
      (STAGE_SIZE - STAGE_PADDING * 2)
  const handleProbeKey = (event: KeyboardEvent<SVGGElement>, probeId: string) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      onProbeClick?.(probeId)
    }
  }

  return (
    <svg
      className="svg-stage knn-stage"
      viewBox={`0 0 ${STAGE_SIZE} ${STAGE_SIZE}`}
      role="img"
      aria-label={label}
    >
      <rect width={STAGE_SIZE} height={STAGE_SIZE} rx="18" fill="transparent" />
      <line
        x1={STAGE_PADDING}
        y1={STAGE_SIZE - STAGE_PADDING}
        x2={STAGE_SIZE - STAGE_PADDING}
        y2={STAGE_SIZE - STAGE_PADDING}
        stroke="#d7d0c3"
        strokeWidth="1.2"
      />
      <line
        x1={STAGE_PADDING}
        y1={STAGE_PADDING}
        x2={STAGE_PADDING}
        y2={STAGE_SIZE - STAGE_PADDING}
        stroke="#d7d0c3"
        strokeWidth="1.2"
      />

      {lines.map((line, index) =>
        line.feature === 'x' ? (
          <g key={`${line.id}-plot-line`}>
            <line
              x1={toX(line.threshold)}
              y1={STAGE_PADDING}
              x2={toX(line.threshold)}
              y2={STAGE_SIZE - STAGE_PADDING}
              stroke={line.color}
              strokeWidth="2.5"
              strokeDasharray={index % 2 ? '5 5' : '9 6'}
              opacity="0.82"
            />
            <text
              x={toX(line.threshold) + 4}
              y={STAGE_PADDING + 13 + (index % 3) * 13}
              fontSize="10"
              fontWeight="700"
              fill={line.color}
            >
              {line.label}
            </text>
          </g>
        ) : (
          <g key={`${line.id}-plot-line`}>
            <line
              x1={STAGE_PADDING}
              y1={toY(line.threshold)}
              x2={STAGE_SIZE - STAGE_PADDING}
              y2={toY(line.threshold)}
              stroke={line.color}
              strokeWidth="2.5"
              strokeDasharray={index % 2 ? '5 5' : '9 6'}
              opacity="0.82"
            />
            <text
              x={STAGE_PADDING + 5 + (index % 2) * 34}
              y={toY(line.threshold) - 5}
              fontSize="10"
              fontWeight="700"
              fill={line.color}
            >
              {line.label}
            </text>
          </g>
        ),
      )}

      {rows.map((row) => (
        <g key={`${label}-row-${row.id}`}>
          <circle
            cx={toX(row.x)}
            cy={toY(row.y)}
            r="7.5"
            fill={classes[row.label].color}
            stroke="#122026"
            strokeWidth="1.3"
          />
          <text
            x={toX(row.x) + 10}
            y={toY(row.y) - 8}
            fontSize="11"
            fontWeight="700"
            fill="#39474d"
          >
            {row.id}
          </text>
        </g>
      ))}

      {probes.map((probe) => {
        const prediction = predictions[probe.id]
        const interactive = Boolean(onProbeClick) && !disabled
        return (
          <g
            key={`${label}-probe-${probe.id}`}
            role={interactive ? 'button' : undefined}
            tabIndex={interactive ? 0 : undefined}
            aria-label={interactive ? `Set prediction for probe ${probe.id}` : undefined}
            onClick={interactive ? () => onProbeClick?.(probe.id) : undefined}
            onKeyDown={interactive ? (event) => handleProbeKey(event, probe.id) : undefined}
            style={{ cursor: interactive ? 'pointer' : 'default' }}
          >
            <rect
              x={toX(probe.x) - 10}
              y={toY(probe.y) - 10}
              width="20"
              height="20"
              rx="3"
              fill={prediction === undefined ? '#fffaf3' : classes[prediction].color}
              stroke="#142026"
              strokeWidth="2.5"
              transform={`rotate(45 ${toX(probe.x)} ${toY(probe.y)})`}
            />
            <text
              x={toX(probe.x)}
              y={toY(probe.y) + 4}
              textAnchor="middle"
              fontSize="11"
              fontWeight="800"
              fill={prediction === undefined ? '#142026' : '#fffaf3'}
            >
              {probe.id}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

function BootstrapAuditPanel({
  dataset,
  counts,
  oobIds,
  disabled,
  onCountChange,
  onToggleOob,
}: {
  dataset: RandomForestBootstrapDataset
  counts: Record<string, string>
  oobIds: string[]
  disabled: boolean
  onCountChange: (rowId: string, value: string) => void
  onToggleOob: (rowId: string) => void
}) {
  return (
    <div className="table-shell">
      <section className="data-table">
        <h3 className="panel-title">Bootstrap draw, in order</h3>
        <div className="inline-actions">
          {dataset.draws.map((rowId, index) => (
            <span className="pill" key={`${dataset.id}-draw-${index}`}>
              {index + 1}: {rowId}
            </span>
          ))}
        </div>
        <p className="table-caption">The sample has eight draws, not eight distinct rows.</p>
      </section>

      <section className="answer-panel">
        <h3 className="panel-title">Multiplicity and OOB status</h3>
        <table className="table-grid">
          <thead>
            <tr>
              <th>Row</th>
              <th>Class</th>
              <th>Multiplicity</th>
              <th>OOB?</th>
            </tr>
          </thead>
          <tbody>
            {dataset.rows.map((row) => (
              <tr key={`${dataset.id}-answer-${row.id}`}>
                <td>{row.id}</td>
                <td>{dataset.classes[row.label].label}</td>
                <td>
                  <input
                    className="numeric-input"
                    type="number"
                    min="0"
                    step="1"
                    value={counts[row.id] ?? ''}
                    aria-label={`Multiplicity for row ${row.id}`}
                    disabled={disabled}
                    onChange={(event) => onCountChange(row.id, event.target.value)}
                    style={{ width: '5.5rem' }}
                  />
                </td>
                <td>
                  <input
                    type="checkbox"
                    checked={oobIds.includes(row.id)}
                    aria-label={`Row ${row.id} is out of bag`}
                    disabled={disabled}
                    onChange={() => onToggleOob(row.id)}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}

function FeatureGeometryPanel({
  dataset,
  selectedSplits,
  predictions,
  disabled,
  onSplitChange,
  onPredictionChange,
}: {
  dataset: RandomForestFeatureGeometryDataset
  selectedSplits: Record<string, string | undefined>
  predictions: Record<string, Record<string, RandomForestLabel | undefined>>
  disabled: boolean
  onSplitChange: (treeId: string, splitId: string) => void
  onPredictionChange: (probeId: string, treeId: string, label: RandomForestLabel) => void
}) {
  const lines = dataset.trees.flatMap<PlotLine>((tree) => {
    const candidate = tree.candidates.find((value) => value.id === selectedSplits[tree.id])
    return candidate ? [{ ...candidate, color: tree.color, label: tree.label }] : []
  })

  return (
    <div className="visual-grid">
      <section className="visual-panel">
        <h3 className="panel-title">Training geometry and probe points</h3>
        <p className="panel-note">Circles are training rows; diamonds P and Q are probes.</p>
        <ForestPlot
          label={dataset.label}
          rows={dataset.rows}
          probes={dataset.probes}
          bounds={dataset.bounds}
          classes={dataset.classes}
          lines={lines}
          disabled={disabled}
        />
      </section>

      <div style={compactGridStyle}>
        {dataset.trees.map((tree) => (
          <section className="answer-panel" key={`${dataset.id}-${tree.id}`}>
            <h3 className="panel-title">{tree.label}</h3>
            <p className="table-caption">Draws: {tree.draws.join(', ')}</p>
            <div className="class-picker" role="radiogroup" aria-label={`${tree.label} threshold`}>
              {tree.candidates.map((candidate) => (
                <button
                  type="button"
                  className={`class-choice ${selectedSplits[tree.id] === candidate.id ? 'selected' : ''}`}
                  key={candidate.id}
                  disabled={disabled}
                  aria-pressed={selectedSplits[tree.id] === candidate.id}
                  onClick={() => onSplitChange(tree.id, candidate.id)}
                  style={{
                    borderColor:
                      selectedSplits[tree.id] === candidate.id
                        ? tree.color
                        : 'rgba(23, 34, 40, 0.08)',
                    background:
                      selectedSplits[tree.id] === candidate.id ? `${tree.color}22` : undefined,
                  }}
                >
                  {formatRule(candidate)}
                </button>
              ))}
            </div>
          </section>
        ))}

        <section className="answer-panel">
          <h3 className="panel-title">Trace the fitted stumps</h3>
          <div style={compactGridStyle}>
            {dataset.probes.flatMap((probe) =>
              dataset.trees.map((tree) => (
                <div key={`${probe.id}-${tree.id}`}>
                  <p className="panel-note" style={{ marginBottom: '0.35rem' }}>
                    Probe {probe.id} through {tree.label}
                  </p>
                  <ClassButtons
                    classes={dataset.classes}
                    value={predictions[probe.id]?.[tree.id]}
                    label={`Prediction for probe ${probe.id} from ${tree.label}`}
                    disabled={disabled}
                    onChange={(label) => onPredictionChange(probe.id, tree.id, label)}
                  />
                </div>
              )),
            )}
          </div>
        </section>
      </div>
    </div>
  )
}

function ForestVotePanel({
  dataset,
  predictions,
  disabled,
  onPredictionChange,
}: {
  dataset: RandomForestVoteGeometryDataset
  predictions: Record<string, RandomForestLabel | undefined>
  disabled: boolean
  onPredictionChange: (probeId: string, label: RandomForestLabel) => void
}) {
  const lines = dataset.trees.map<PlotLine>((tree, index) => ({
    id: tree.id,
    feature: tree.feature,
    threshold: tree.threshold,
    label: tree.id,
    color: FOREST_LINE_COLORS[index % FOREST_LINE_COLORS.length],
  }))

  return (
    <div className="visual-grid">
      <section className="visual-panel">
        <h3 className="panel-title">Five-tree forest</h3>
        <p className="panel-note">Click a diamond to toggle its proposed forest class.</p>
        <ForestPlot
          label={dataset.label}
          rows={dataset.rows}
          probes={dataset.probes}
          bounds={dataset.bounds}
          classes={dataset.classes}
          lines={lines}
          predictions={predictions}
          disabled={disabled}
          onProbeClick={(probeId) =>
            onPredictionChange(probeId, predictions[probeId] === 0 ? 1 : 0)
          }
        />
      </section>

      <div style={compactGridStyle}>
        <section className="data-table">
          <h3 className="panel-title">Tree rules</h3>
          <div style={compactGridStyle}>
            {dataset.trees.map((tree, index) => (
              <div className="metric-card" key={`${dataset.id}-${tree.id}`}>
                <span
                  className="metric-value"
                  style={{ color: FOREST_LINE_COLORS[index % FOREST_LINE_COLORS.length] }}
                >
                  {tree.id}: {formatRule(tree)}
                </span>
                <span className="metric-copy">
                  left → {dataset.classes[tree.leftLabel].label}; right → {dataset.classes[tree.rightLabel].label}
                </span>
              </div>
            ))}
          </div>
        </section>

        <section className="answer-panel">
          <h3 className="panel-title">Forest predictions</h3>
          <div style={compactGridStyle}>
            {dataset.probes.map((probe) => (
              <div key={`${dataset.id}-answer-${probe.id}`}>
                <p className="panel-note" style={{ marginBottom: '0.35rem' }}>
                  Probe {probe.id} = ({probe.x}, {probe.y})
                </p>
                <ClassButtons
                  classes={dataset.classes}
                  value={predictions[probe.id]}
                  label={`Forest prediction for probe ${probe.id}`}
                  disabled={disabled}
                  onChange={(label) => onPredictionChange(probe.id, label)}
                />
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}

function OobEstimatePanel({
  dataset,
  predictions,
  accuracy,
  disabled,
  onPredictionChange,
  onAccuracyChange,
}: {
  dataset: RandomForestOobDataset
  predictions: Record<string, RandomForestLabel | undefined>
  accuracy: string
  disabled: boolean
  onPredictionChange: (rowId: string, label: RandomForestLabel) => void
  onAccuracyChange: (value: string) => void
}) {
  return (
    <div style={compactGridStyle}>
      <section className="data-table" style={{ overflowX: 'auto' }}>
        <h3 className="panel-title">OOB prediction matrix</h3>
        <table className="table-grid">
          <thead>
            <tr>
              <th>Row</th>
              <th>True class</th>
              {dataset.trees.map((tree) => (
                <th key={`${dataset.id}-${tree.id}`}>{tree.id}</th>
              ))}
              <th>OOB majority</th>
            </tr>
          </thead>
          <tbody>
            {dataset.rows.map((row) => (
              <tr key={`${dataset.id}-row-${row.id}`}>
                <td>{row.id}</td>
                <td>{row.label}</td>
                {dataset.trees.map((tree) => (
                  <td key={`${tree.id}-${row.id}`}>
                    {tree.predictions[row.id] === undefined ? '—' : tree.predictions[row.id]}
                  </td>
                ))}
                <td>
                  <ClassButtons
                    classes={dataset.classes}
                    value={predictions[row.id]}
                    label={`OOB majority for row ${row.id}`}
                    disabled={disabled}
                    onChange={(label) => onPredictionChange(row.id, label)}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="answer-panel">
        <label className="decision-tree-input-label">
          <span>OOB accuracy as a decimal</span>
          <input
            className="numeric-input"
            aria-label="OOB accuracy"
            value={accuracy}
            disabled={disabled}
            onChange={(event) => onAccuracyChange(event.target.value)}
          />
        </label>
      </section>
    </div>
  )
}

function VariancePanel({
  dataset,
  variances,
  bestScenarioId,
  interventionId,
  disabled,
  onVarianceChange,
  onBestScenarioChange,
  onInterventionChange,
}: {
  dataset: RandomForestVarianceDataset
  variances: Record<string, string>
  bestScenarioId: string | undefined
  interventionId: string | undefined
  disabled: boolean
  onVarianceChange: (scenarioId: string, value: string) => void
  onBestScenarioChange: (scenarioId: string) => void
  onInterventionChange: (interventionId: string) => void
}) {
  return (
    <div style={compactGridStyle}>
      <section className="data-table">
        <h3 className="panel-title">Forest scenarios</h3>
        <table className="table-grid">
          <thead>
            <tr>
              <th>Scenario</th>
              <th>Trees (T)</th>
              <th>Correlation (rho)</th>
              <th>Var(average)</th>
            </tr>
          </thead>
          <tbody>
            {dataset.scenarios.map((scenario) => (
              <tr key={`${dataset.id}-${scenario.id}`}>
                <td>{scenario.label}</td>
                <td>{scenario.treeCount}</td>
                <td>{scenario.correlation}</td>
                <td>
                  <input
                    className="numeric-input"
                    aria-label={`${scenario.label} variance`}
                    value={variances[scenario.id] ?? ''}
                    disabled={disabled}
                    onChange={(event) => onVarianceChange(scenario.id, event.target.value)}
                    style={{ width: '7rem' }}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <div style={splitGridStyle}>
        <section className="answer-panel">
          <h3 className="panel-title">Lowest approximate variance</h3>
          <div className="class-picker" role="radiogroup" aria-label="Lowest variance forest">
            {dataset.scenarios.map((scenario) => (
              <button
                type="button"
                className={`class-choice ${bestScenarioId === scenario.id ? 'selected' : ''}`}
                key={`${dataset.id}-best-${scenario.id}`}
                disabled={disabled}
                onClick={() => onBestScenarioChange(scenario.id)}
              >
                {scenario.label}
              </button>
            ))}
          </div>
        </section>

        <section className="answer-panel">
          <h3 className="panel-title">Which change directly lowers correlation?</h3>
          <div style={compactGridStyle}>
            {dataset.interventions.map((intervention) => (
              <button
                type="button"
                className={`class-choice ${interventionId === intervention.id ? 'selected' : ''}`}
                key={`${dataset.id}-intervention-${intervention.id}`}
                disabled={disabled}
                onClick={() => onInterventionChange(intervention.id)}
                style={{ textAlign: 'left' }}
              >
                <strong>{intervention.label}</strong>
                <span className="metric-copy">{intervention.description}</span>
              </button>
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}

function CodeTracePanel({
  dataset,
  oobIds,
  trainingFeatureValues,
  queryProjection,
  forestLabel,
  featureStatementId,
  predictionStatementId,
  aggregationStatementId,
  disabled,
  onToggleOob,
  onTrainingFeatureValuesChange,
  onQueryProjectionChange,
  onForestLabelChange,
  onFeatureStatementChange,
  onPredictionStatementChange,
  onAggregationStatementChange,
}: {
  dataset: RandomForestCodeTraceDataset
  oobIds: string[]
  trainingFeatureValues: string
  queryProjection: string
  forestLabel: RandomForestLabel | undefined
  featureStatementId: string | undefined
  predictionStatementId: string | undefined
  aggregationStatementId: string | undefined
  disabled: boolean
  onToggleOob: (rowId: string) => void
  onTrainingFeatureValuesChange: (value: string) => void
  onQueryProjectionChange: (value: string) => void
  onForestLabelChange: (label: RandomForestLabel) => void
  onFeatureStatementChange: (statementId: string) => void
  onPredictionStatementChange: (statementId: string) => void
  onAggregationStatementChange: (statementId: string) => void
}) {
  return (
    <div className="visual-grid">
      <section className="data-table">
        <h3 className="panel-title">Reference implementation</h3>
        <pre
          style={{
            margin: 0,
            padding: '1rem',
            borderRadius: '16px',
            overflowX: 'auto',
            background: '#172228',
            color: '#f7efe3',
            lineHeight: 1.65,
          }}
        >
          {dataset.statements.map((statement) => (
            <code key={statement.id} style={{ display: 'block' }}>
              <span style={{ color: '#edb86b', fontWeight: 800 }}>{statement.id}</span>{'  '}
              {statement.code}
            </code>
          ))}
        </pre>

        <h3 className="panel-title" style={{ marginTop: '1rem' }}>Concrete trace</h3>
        <table className="table-grid">
          <thead>
            <tr>
              <th>Row</th>
              <th>X[row]</th>
              <th>y</th>
            </tr>
          </thead>
          <tbody>
            {dataset.rows.map((row) => (
              <tr key={`${dataset.id}-${row.id}`}>
                <td>{row.id}</td>
                <td>[{row.features.join(', ')}]</td>
                <td>{row.label}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="table-caption">
          sample = [{dataset.bootstrapDraws.join(', ')}], features = [{dataset.selectedFeatureIndices.join(', ')}], X_query = [{dataset.query.join(', ')}]
        </p>
        <p className="table-caption">Five tree votes: [{dataset.treeVotes.join(', ')}]</p>
      </section>

      <section className="answer-panel">
        <h3 className="panel-title">Structured trace</h3>
        <div style={compactGridStyle}>
          <fieldset style={{ border: 0, padding: 0, margin: 0 }}>
            <legend className="panel-note">OOB row IDs for this sample</legend>
            <div className="inline-actions">
              {dataset.rows.map((row) => (
                <label className="pill" key={`${dataset.id}-oob-${row.id}`}>
                  <input
                    type="checkbox"
                    checked={oobIds.includes(row.id)}
                    disabled={disabled}
                    onChange={() => onToggleOob(row.id)}
                  />
                  {row.id}
                </label>
              ))}
            </div>
          </fieldset>

          <label className="decision-tree-input-label">
            <span>X[sample][:, features], flattened in draw order</span>
            <input
              className="numeric-input"
              aria-label="Projected bootstrap training values"
              placeholder="e.g. 6, 9, 6, 2"
              value={trainingFeatureValues}
              disabled={disabled}
              onChange={(event) => onTrainingFeatureValuesChange(event.target.value)}
            />
          </label>

          <label className="decision-tree-input-label">
            <span>X_query[:, features]</span>
            <input
              className="numeric-input"
              aria-label="Projected query values"
              value={queryProjection}
              disabled={disabled}
              onChange={(event) => onQueryProjectionChange(event.target.value)}
            />
          </label>

          <div>
            <p className="panel-note" style={{ marginBottom: '0.35rem' }}>Forest output for the five votes</p>
            <ClassButtons
              classes={dataset.classes}
              value={forestLabel}
              label="Code trace forest output"
              disabled={disabled}
              onChange={onForestLabelChange}
            />
          </div>

          {[
            {
              label: 'Statement that samples a feature subset',
              value: featureStatementId,
              onChange: onFeatureStatementChange,
            },
            {
              label: 'Statement that must reuse the feature subset at prediction time',
              value: predictionStatementId,
              onChange: onPredictionStatementChange,
            },
            {
              label: 'Statement that aggregates across trees',
              value: aggregationStatementId,
              onChange: onAggregationStatementChange,
            },
          ].map((item) => (
            <label className="decision-tree-input-label" key={item.label}>
              <span>{item.label}</span>
              <select
                className="numeric-input"
                aria-label={item.label}
                value={item.value ?? ''}
                disabled={disabled}
                onChange={(event) => item.onChange(event.target.value)}
              >
                <option value="">Choose a statement</option>
                {dataset.statementOptions.map((option) => (
                  <option key={`${item.label}-${option.id}`} value={option.id}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
      </section>
    </div>
  )
}

export function RandomForestQuestion({
  question,
  state,
  questionNumber,
  totalQuestions,
  hints,
  onAttempt,
  onGiveUp,
}: RandomForestQuestionProps) {
  const dataset = randomForestDatasets[question.datasetId]
  const [feedback, setFeedback] = useState('')
  const [multiplicityInputs, setMultiplicityInputs] = useState<Record<string, string>>({})
  const [bootstrapOobIds, setBootstrapOobIds] = useState<string[]>([])
  const [selectedSplits, setSelectedSplits] = useState<Record<string, string | undefined>>({})
  const [featurePredictions, setFeaturePredictions] = useState<
    Record<string, Record<string, RandomForestLabel | undefined>>
  >({})
  const [forestPredictions, setForestPredictions] = useState<
    Record<string, RandomForestLabel | undefined>
  >({})
  const [oobPredictions, setOobPredictions] = useState<
    Record<string, RandomForestLabel | undefined>
  >({})
  const [oobAccuracyInput, setOobAccuracyInput] = useState('')
  const [varianceInputs, setVarianceInputs] = useState<Record<string, string>>({})
  const [bestScenarioId, setBestScenarioId] = useState<string | undefined>()
  const [interventionId, setInterventionId] = useState<string | undefined>()
  const [codeOobIds, setCodeOobIds] = useState<string[]>([])
  const [trainingFeatureValues, setTrainingFeatureValues] = useState('')
  const [queryProjection, setQueryProjection] = useState('')
  const [codeForestLabel, setCodeForestLabel] = useState<RandomForestLabel | undefined>()
  const [featureStatementId, setFeatureStatementId] = useState<string | undefined>()
  const [predictionStatementId, setPredictionStatementId] = useState<string | undefined>()
  const [aggregationStatementId, setAggregationStatementId] = useState<string | undefined>()
  const resolved = state.status === 'correct' || state.status === 'gave_up'
  const schedule = question.hintSchedule ?? DEFAULT_HINT_SCHEDULE

  const buildSubmission = (): unknown => {
    if (dataset.kind === 'bootstrapAudit') {
      return {
        multiplicities: Object.fromEntries(
          dataset.rows.map((row) => [row.id, parseNumber(multiplicityInputs[row.id] ?? '')]),
        ),
        oobIds: bootstrapOobIds,
      }
    }

    if (dataset.kind === 'featureSubsamplingGeometry') {
      return { selectedSplits, predictions: featurePredictions }
    }

    if (dataset.kind === 'forestVoteGeometry') {
      return { predictions: forestPredictions }
    }

    if (dataset.kind === 'oobEstimate') {
      return { predictions: oobPredictions, accuracy: parseNumber(oobAccuracyInput) }
    }

    if (dataset.kind === 'varianceReduction') {
      return {
        variances: Object.fromEntries(
          dataset.scenarios.map((scenario) => [
            scenario.id,
            parseNumber(varianceInputs[scenario.id] ?? ''),
          ]),
        ),
        bestScenarioId,
        interventionId,
      }
    }

    return {
      oobIds: codeOobIds,
      trainingFeatureValues: parseNumberList(trainingFeatureValues),
      queryProjection: parseNumberList(queryProjection),
      forestLabel: codeForestLabel,
      featureStatementId,
      predictionStatementId,
      aggregationStatementId,
    }
  }

  const checkWork = () => {
    const submission = buildSubmission()
    const result = question.validator(submission)
    if (!result.correct) {
      const nextIncorrectAttempts = state.incorrectAttempts + 1
      setFeedback(
        schedule.includes(nextIncorrectAttempts)
          ? 'Not yet. A new hint appeared below.'
          : (result.message ?? 'That answer is not correct yet.'),
      )
      onAttempt('incorrect', submission)
      return
    }

    setFeedback(result.message ?? question.successCopy ?? 'Correct! Good Job!')
    onAttempt('correct', submission)
  }

  const revealAnswer = () => {
    const draft = buildSubmission()

    if (dataset.kind === 'bootstrapAudit') {
      const rowIds = dataset.rows.map((row) => row.id)
      setMultiplicityInputs(
        Object.fromEntries(
          Object.entries(bootstrapMultiplicities(rowIds, dataset.draws)).map(([id, count]) => [
            id,
            String(count),
          ]),
        ),
      )
      setBootstrapOobIds(outOfBagIds(rowIds, dataset.draws))
    }

    if (dataset.kind === 'featureSubsamplingGeometry') {
      const nextSplits: Record<string, string> = {}
      const trained = new Map(
        dataset.trees.map((tree) => {
          const best = findBestStump(dataset.rows, tree.candidates, tree.draws)
          nextSplits[tree.id] = best.candidate.id
          return [tree.id, trainedStumpFromEvaluation(best)]
        }),
      )
      setSelectedSplits(nextSplits)
      setFeaturePredictions(
        Object.fromEntries(
          dataset.probes.map((probe) => [
            probe.id,
            Object.fromEntries(
              dataset.trees.map((tree) => [
                tree.id,
                predictStump(trained.get(tree.id)!, probe),
              ]),
            ),
          ]),
        ),
      )
    }

    if (dataset.kind === 'forestVoteGeometry') {
      setForestPredictions(
        Object.fromEntries(
          dataset.probes.map((probe) => [probe.id, predictForest(dataset.trees, probe)]),
        ),
      )
    }

    if (dataset.kind === 'oobEstimate') {
      setOobPredictions(
        Object.fromEntries(
          aggregateOobPredictions(
            dataset.rows.map((row) => row.id),
            dataset.trees,
          ).map((entry) => [entry.rowId, entry.prediction ?? 0]),
        ),
      )
      setOobAccuracyInput(String(oobAccuracy(dataset.rows, dataset.trees)))
    }

    if (dataset.kind === 'varianceReduction') {
      const values = Object.fromEntries(
        dataset.scenarios.map((scenario) => [
          scenario.id,
          correlatedEnsembleVariance(
            dataset.singleTreeVariance,
            scenario.correlation,
            scenario.treeCount,
          ),
        ]),
      )
      setVarianceInputs(
        Object.fromEntries(
          Object.entries(values).map(([scenarioId, value]) => [scenarioId, value.toFixed(3)]),
        ),
      )
      setBestScenarioId(
        dataset.scenarios.reduce((best, scenario) =>
          values[scenario.id] < values[best.id] ? scenario : best,
        ).id,
      )
      setInterventionId('feature-subsampling')
    }

    if (dataset.kind === 'codeTrace') {
      const rowsById = new Map(dataset.rows.map((row) => [row.id, row]))
      setCodeOobIds(
        outOfBagIds(
          dataset.rows.map((row) => row.id),
          dataset.bootstrapDraws,
        ),
      )
      setTrainingFeatureValues(
        dataset.bootstrapDraws
          .flatMap((rowId) =>
            dataset.selectedFeatureIndices.map(
              (featureIndex) => rowsById.get(rowId)!.features[featureIndex],
            ),
          )
          .join(', '),
      )
      setQueryProjection(
        dataset.selectedFeatureIndices.map((featureIndex) => dataset.query[featureIndex]).join(', '),
      )
      setCodeForestLabel(majorityVote(dataset.treeVotes))
      setFeatureStatementId('S3')
      setPredictionStatementId('S5')
      setAggregationStatementId('S7')
    }

    setFeedback('Answer revealed. Compare each structured field with your draft.')
    onGiveUp(draft)
  }

  let body = null

  if (dataset.kind === 'bootstrapAudit') {
    body = (
      <BootstrapAuditPanel
        dataset={dataset}
        counts={multiplicityInputs}
        oobIds={bootstrapOobIds}
        disabled={resolved}
        onCountChange={(rowId, value) =>
          setMultiplicityInputs((current) => ({ ...current, [rowId]: value }))
        }
        onToggleOob={(rowId) => setBootstrapOobIds((current) => toggleId(current, rowId))}
      />
    )
  }

  if (dataset.kind === 'featureSubsamplingGeometry') {
    body = (
      <FeatureGeometryPanel
        dataset={dataset}
        selectedSplits={selectedSplits}
        predictions={featurePredictions}
        disabled={resolved}
        onSplitChange={(treeId, splitId) =>
          setSelectedSplits((current) => ({ ...current, [treeId]: splitId }))
        }
        onPredictionChange={(probeId, treeId, label) =>
          setFeaturePredictions((current) => ({
            ...current,
            [probeId]: { ...current[probeId], [treeId]: label },
          }))
        }
      />
    )
  }

  if (dataset.kind === 'forestVoteGeometry') {
    body = (
      <ForestVotePanel
        dataset={dataset}
        predictions={forestPredictions}
        disabled={resolved}
        onPredictionChange={(probeId, label) =>
          setForestPredictions((current) => ({ ...current, [probeId]: label }))
        }
      />
    )
  }

  if (dataset.kind === 'oobEstimate') {
    body = (
      <OobEstimatePanel
        dataset={dataset}
        predictions={oobPredictions}
        accuracy={oobAccuracyInput}
        disabled={resolved}
        onPredictionChange={(rowId, label) =>
          setOobPredictions((current) => ({ ...current, [rowId]: label }))
        }
        onAccuracyChange={setOobAccuracyInput}
      />
    )
  }

  if (dataset.kind === 'varianceReduction') {
    body = (
      <VariancePanel
        dataset={dataset}
        variances={varianceInputs}
        bestScenarioId={bestScenarioId}
        interventionId={interventionId}
        disabled={resolved}
        onVarianceChange={(scenarioId, value) =>
          setVarianceInputs((current) => ({ ...current, [scenarioId]: value }))
        }
        onBestScenarioChange={setBestScenarioId}
        onInterventionChange={setInterventionId}
      />
    )
  }

  if (dataset.kind === 'codeTrace') {
    body = (
      <CodeTracePanel
        dataset={dataset}
        oobIds={codeOobIds}
        trainingFeatureValues={trainingFeatureValues}
        queryProjection={queryProjection}
        forestLabel={codeForestLabel}
        featureStatementId={featureStatementId}
        predictionStatementId={predictionStatementId}
        aggregationStatementId={aggregationStatementId}
        disabled={resolved}
        onToggleOob={(rowId) => setCodeOobIds((current) => toggleId(current, rowId))}
        onTrainingFeatureValuesChange={setTrainingFeatureValues}
        onQueryProjectionChange={setQueryProjection}
        onForestLabelChange={setCodeForestLabel}
        onFeatureStatementChange={setFeatureStatementId}
        onPredictionStatementChange={setPredictionStatementId}
        onAggregationStatementChange={setAggregationStatementId}
      />
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
          <button type="button" className="button" onClick={checkWork} disabled={resolved}>
            Check answer
          </button>
          <button type="button" className="button-secondary" onClick={revealAnswer} disabled={resolved}>
            Give up
          </button>
        </>
      }
    >
      {body}
    </QuestionFrame>
  )
}
