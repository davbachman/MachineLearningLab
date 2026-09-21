import { useState } from 'react'
import type { BatchGradientVisualDataset } from '../../data/batchGradientAssignment'
import { batchEpochs, batchLearningRate, batchLoss, batchOptimum, batchSizes, batchStart, batchTargets, batchTrajectory, shuffleSchedules } from '../../lib/batchGradientConcepts'
import type { TeachingBatchSize } from '../../lib/batchGradientConcepts'

const teal = '#1c7b8a', orange = '#b9601b', purple = '#77326b'
const fixed = (value: number) => value.toFixed(4)

function TargetPlot() {
  const [w, setW] = useState(3)
  const x = (row: number) => 75 + row * 67
  const y = (value: number) => 290 - value * 28
  return <>
    <label className="slider-shell"><span>Explore current prediction w = {w.toFixed(2)}</span><input aria-label="Current constant prediction" className="range-input" type="range" min="0" max="8" step="0.25" value={w} onChange={event => setW(Number(event.target.value))} /></label>
    <button type="button" className="button button-secondary" onClick={() => setW(3)}>Return to w = 3 for the questions</button>
    <svg viewBox="0 0 620 348" role="img" aria-label={`Targets of rows R1 through R8, with current constant prediction w = ${w}; first four rows form batch A`} style={{ width: '100%' }}>
    <rect x="45" y="42" width="265" height="248" fill="#e8f1f1" />
    <rect x="310" y="42" width="265" height="248" fill="#f8edde" />
    {[0, 2, 4, 6, 8].map(value => <g key={value}>
      <line x1="45" x2="575" y1={y(value)} y2={y(value)} stroke="#d8d4ca" />
      <text x="33" y={y(value) + 5} textAnchor="end" fontSize="15">{value}</text>
    </g>)}
    <text x="10" y="24" fontSize="16">Target / prediction</text>
    <text x="177" y="60" textAnchor="middle" fontSize="14">First four rows</text>
    <text x="443" y="60" textAnchor="middle" fontSize="14">Last four rows</text>
    <line x1="45" x2="575" y1={y(w)} y2={y(w)} stroke={purple} strokeWidth="2" strokeDasharray="7 4" />
    <text x="581" y={y(w) - 8} textAnchor="end" fontSize="15" fill={purple}>w = {w.toFixed(2)}</text>
    {batchTargets.map((target, row) => <g key={row}>
      <line x1={x(row)} x2={x(row)} y1={y(target)} y2={y(w)} stroke={row < 4 ? teal : orange} opacity="0.55" />
      <circle cx={x(row)} cy={y(target)} r="6" fill={row < 4 ? teal : orange} />
      <text x={x(row) + 11} y={y(target) - 7} fontSize="15">{target}</text>
      <text x={x(row)} y="315" textAnchor="middle" fontSize="15">R{row + 1}</text>
    </g>)}
    <text x="310" y="342" textAnchor="middle" fontSize="16">Observation row (not a feature)</text>
    </svg>
    <p className="panel-note">Moving the line changes the residuals but does not change any targets. For both questions below, use w = 3 regardless of what you explored.</p>
  </>
}

function BatchExplorer() {
  const [size, setSize] = useState<TeachingBatchSize>(1)
  const [step, setStep] = useState(5)
  const run = batchTrajectory(size)
  const fullRun = batchTrajectory(8)
  const current = run[step - 1]
  const x = (processed: number) => 58 + processed / 32 * 522
  const y = (loss: number) => 292 - (loss - 3) / 9 * 246
  const selectedPoints = [{ rowsProcessed: 0, fullLoss: batchLoss(batchStart) }, ...run]
  const fullPoints = [{ rowsProcessed: 0, fullLoss: batchLoss(batchStart) }, ...fullRun]
  return <>
    <div style={{ display: 'flex', gap: '1.25rem', alignItems: 'end', flexWrap: 'wrap', marginBottom: '1rem' }}>
      <label className="slider-shell" style={{ flex: '1 1 230px' }}>Batch size
        <select aria-label="Batch size" value={size} onChange={event => {
          const next = Number(event.target.value) as TeachingBatchSize
          setSize(next)
          setStep(Math.min(step, batchTrajectory(next).length))
        }}>{batchSizes.map(value => <option key={value} value={value}>{value}{value === 1 ? ' (stochastic)' : value === 8 ? ' (full dataset)' : ' (mini-batch)'}</option>)}</select>
      </label>
      <button type="button" className="button button-secondary" onClick={() => { setSize(1); setStep(5) }}>Show batch size 1, update 5</button>
    </div>
    <div className="visual-grid" style={{ gridTemplateColumns: 'minmax(0, 1fr)' }}>
      <section className="visual-panel">
        <h4 className="panel-title">Full-data loss across four epochs</h4>
        <svg viewBox="0 0 620 354" role="img" aria-label={`Full-data loss over 32 observations processed, batch size ${size}; dashed reference uses all eight rows per update`} style={{ width: '100%', maxWidth: 700, display: 'block', margin: 'auto' }}>
          {[4, 6, 8, 10, 12].map(value => <g key={value}>
            <line x1="58" x2="580" y1={y(value)} y2={y(value)} stroke="#ded9cf" />
            <text x="46" y={y(value) + 5} textAnchor="end" fontSize="15">{value}</text>
          </g>)}
          {[0, 8, 16, 24, 32].map(value => <g key={value}>
            <line x1={x(value)} x2={x(value)} y1="46" y2="292" stroke="#ece7de" />
            <text x={x(value)} y="316" textAnchor="middle" fontSize="15">{value}</text>
          </g>)}
          <text x="12" y="24" fontSize="16">Loss (½ MSE)</text>
          <text x="310" y="346" textAnchor="middle" fontSize="16">Observations processed (8 = one epoch)</text>
          <line x1="58" x2="580" y1={y(batchLoss(batchOptimum))} y2={y(batchLoss(batchOptimum))} stroke="#888" strokeDasharray="2 4" />
          <polyline points={fullPoints.map(point => `${x(point.rowsProcessed)},${y(point.fullLoss)}`).join(' ')} fill="none" stroke={orange} strokeWidth="3" strokeDasharray="8 5" />
          <polyline points={selectedPoints.map(point => `${x(point.rowsProcessed)},${y(point.fullLoss)}`).join(' ')} fill="none" stroke={teal} strokeWidth="2.5" />
          <line x1={x(current.rowsProcessed)} x2={x(current.rowsProcessed)} y1="46" y2="292" stroke={purple} strokeDasharray="3 3" opacity="0.6" />
          <circle cx={x(current.rowsProcessed)} cy={y(current.fullLoss)} r="6" fill={purple} stroke="white" strokeWidth="2" />
        </svg>
        <p className="panel-note">Solid teal: selected batch size. Dashed orange: full-data updates. Dotted gray: minimum loss, 3.75. Purple dot: inspected update. Lines connect computed checkpoints; they are not extra updates.</p>
      </section>
      <section className="projection-panel">
        <h4 className="panel-title">Inspect a single update</h4>
        <label className="slider-shell"><span>Update {step} of {run.length} (epoch {current.epoch})</span>
          <input className="range-input" aria-label="Inspect update" type="range" min="1" max={run.length} step="1" value={step} onChange={event => setStep(Number(event.target.value))} />
        </label>
        <div aria-live="polite" aria-atomic="true">
          <p>Selected rows: <output aria-label="Selected batch rows">{current.rows.map(row => `R${row + 1} (target ${batchTargets[row]})`).join(', ')}</output></p>
          <p><output aria-label="Parameter update">w: {fixed(current.before)} → {fixed(current.w)}</output></p>
          <table className="table-grid"><caption>Loss for this update (½ MSE)</caption>
            <thead><tr><th>Data used to evaluate loss</th><th>Before</th><th>After</th></tr></thead>
            <tbody><tr><th>Selected batch</th><td>{fixed(current.batchLossBefore)}</td><td>{fixed(current.batchLossAfter)}</td></tr>
              <tr><th>All eight rows</th><td>{fixed(current.fullLossBefore)}</td><td>{fixed(current.fullLoss)}</td></tr></tbody>
          </table>
          <p className="panel-note">After all {batchEpochs} epochs: {run.length} updates; 32 observations processed; full-data loss = {fixed(run[run.length - 1].fullLoss)}.</p>
        </div>
        <p className="panel-note">Learning rate = {batchLearningRate}. Only the batch size changes between runs. The viewer does not alter the run or your selected answers below.</p>
      </section>
    </div>
  </>
}

function BudgetPlot() {
  const sizes = [1, 20, 120]
  return <svg viewBox="0 0 730 262" style={{ width: '100%' }} role="img" aria-label="One epoch over 120 observations: batch sizes 1, 20, and 120 divide the same width into 120, 6, and 1 updates">
    <text x="170" y="28" fontSize="16">One pass through 120 observations</text>
    {sizes.map((size, index) => <g key={size}>
      <text x="152" y={67 + index * 67} textAnchor="end" fontSize="16">Batch size {size}</text>
      {Array.from({ length: 120 / size }, (_, block) => <rect key={block} x={170 + block * 4.2 * size} y={45 + index * 67} width={size * 4.2} height="32" fill={index === 1 ? '#d4e8e8' : '#e9e3d6'} stroke={teal} strokeWidth="1" />)}
      <text x="422" y={99 + index * 67} textAnchor="middle" fontSize="14">{120 / size} {size === 120 ? 'update' : 'updates'} in this epoch</text>
    </g>)}
  </svg>
}

function ShufflePlot() {
  return <div style={{ display: 'grid', gap: '1.5rem' }}>
    {shuffleSchedules.map(schedule => <section key={schedule.name}>
      <h4 className="panel-title">Schedule {schedule.name}</h4>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(100px, 1fr))', gap: '0.6rem' }}>
        {[0, 1, 2, 3].map(batch => <div key={batch} style={{ border: '1px solid #8aa3a4', borderBottom: '3px solid #1c7b8a', borderRadius: 8, padding: '0.75rem', background: '#edf3f1' }}>
          <p style={{ margin: '0 0 0.6rem', fontWeight: 600 }}>Batch {batch + 1}</p>
          {schedule.rows.slice(2 * batch, 2 * batch + 2).map(row => <p key={row} style={{ margin: '0.25rem 0' }}>R{row + 1}: target {batchTargets[row]}</p>)}
        </div>)}
      </div>
    </section>)}
  </div>
}

function NearMinimumPlot() {
  const [rate, setRate] = useState(0.4)
  const next = 4 + 3 * rate
  const x = (w: number) => 55 + w * 64
  const y = (loss: number) => 297 - loss * 9
  const path = (rows?: number[]) => Array.from({ length: 161 }, (_, i) => `${x(i / 20)},${y(batchLoss(i / 20, rows))}`).join(' ')
  return <>
    <label className="slider-shell"><span>Learning rate: {rate.toFixed(2)}</span><input className="range-input" aria-label="Single-row learning rate" type="range" min="0.05" max="0.80" step="0.05" value={rate} onChange={event => setRate(Number(event.target.value))} /></label>
    <p className="panel-note">Each change replays exactly one R7-only update from w = 4. It never starts from the previous slider result. All allowed rates are positive, so the questions apply to every slider setting.</p>
    <svg viewBox="0 0 620 354" style={{ width: '100%' }} role="img" aria-label="Full-data and R7-only loss curves: at parameter w = 4 the full-data curve is flat but the R7 curve slopes downward">
    {[0, 5, 10, 15, 20, 25].map(value => <g key={value}>
      <line x1="55" x2="567" y1={y(value)} y2={y(value)} stroke="#ded9cf" />
      <text x="43" y={y(value) + 5} textAnchor="end" fontSize="15">{value}</text>
    </g>)}
    {[0, 2, 4, 6, 8].map(value => <text key={value} x={x(value)} y="320" textAnchor="middle" fontSize="15">{value}</text>)}
    <text x="10" y="24" fontSize="16">Loss (½ MSE)</text><text x="310" y="347" textAnchor="middle" fontSize="16">Parameter w</text>
    <polyline points={path()} fill="none" stroke={teal} strokeWidth="3" />
    <polyline points={path([6])} fill="none" stroke={orange} strokeWidth="3" strokeDasharray="7 5" />
    <line x1={x(4)} x2={x(4)} y1="55" y2="297" stroke={purple} strokeDasharray="3 4" />
    <text x={x(4) + 10} y="65" fill={purple} fontSize="16">Current w = 4</text>
    <circle cx={x(4)} cy={y(batchLoss(4))} r="5" fill={teal} />
    <rect x={x(4) - 4} y={y(batchLoss(4, [6])) - 4} width="8" height="8" fill={orange} />
    <line x1={x(next)} x2={x(next)} y1="88" y2="297" stroke="#576871" strokeDasharray="2 4" />
    <circle cx={x(next)} cy={y(batchLoss(next))} r="5" fill="white" stroke={teal} strokeWidth="2.5" />
    <rect x={x(next) - 4} y={y(batchLoss(next, [6])) - 4} width="8" height="8" fill="white" stroke={orange} strokeWidth="2.5" />
    <text x={x(next) + 8} y="99" fontSize="14">After update</text>
    <text x={x(6.6)} y={y(batchLoss(6.6)) - 12} fontSize="15" fill={teal}>All rows</text>
    <text x={x(1.3)} y={y(batchLoss(1.3, [6])) - 12} fontSize="15" fill={orange}>R7 only</text>
    </svg>
    <div className="status-panel" aria-live="polite" aria-atomic="true">
      <p><output aria-label="Single-row updated prediction">R7-only update: w = 4 → {next.toFixed(2)}</output></p>
      <p>Full-data loss: {fixed(batchLoss(4))} → {fixed(batchLoss(next))}. R7 loss: {fixed(batchLoss(4, [6]))} → {fixed(batchLoss(next, [6]))}.</p>
    </div>
    <p className="panel-note">Filled markers: before the update. Hollow markers: after it. Circles use all rows; squares use R7 only.</p>
  </>
}

const titles = { directions: 'The data and current constant prediction', explorer: 'Replay mini-batch updates', budget: 'An epoch is not an update', shuffle: 'Compare batch composition', nearMinimum: 'One optimum, different batch slopes' }

export function BatchGradientVisual({ dataset }: { dataset: BatchGradientVisualDataset }) {
  return <section className="data-table">
    <h3 className="panel-title">{titles[dataset.mode]}</h3>
    <p className="table-caption">{dataset.caption}</p>
    {dataset.mode === 'explorer' ? <BatchExplorer /> : dataset.mode === 'shuffle' ? <ShufflePlot />
      : <div style={{ maxWidth: 820, margin: 'auto' }}>{dataset.mode === 'directions' ? <TargetPlot /> : dataset.mode === 'budget' ? <BudgetPlot /> : <NearMinimumPlot />}</div>}
  </section>
}
