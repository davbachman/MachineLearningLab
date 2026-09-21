import { useId, useState } from 'react'
import type { NeuralFoundationsVisualDataset } from '../../data/neuralFoundationsAssignments'
import { allPairsMse, batchGroups, contrastImage, contrastKernel, denseParameterCount, dropoutValues, exponentialMoments, hingeValue, logRoot, logRootDerivative, maxPool, pairedMse, poolingImage, sharedBranchTrace, squaredErrorGrid, twoBranchValue, validCorrelation } from '../../lib/neuralFoundations'
import { softmaxProbabilities } from '../../lib/softmaxConcepts'

const teal = '#176d78', orange = '#af571b', purple = '#723a87'
const round = (x: number) => Number(x.toFixed(3)).toString()
const equalChartColumns = { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }

function Slider({ label, value, min, max, step = 1, onChange }: { label: string; value: number; min: number; max: number; step?: number; onChange: (value: number) => void }) {
  return <label className="slider-shell"><span>{label}: {round(value)}</span><input type="range" className="range-input" aria-label={label} min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))} /></label>
}

function Matrix({ values, title, highlight, buttons }: { values: number[][]; title: string; highlight?: (r: number, c: number) => boolean; buttons?: (r: number, c: number) => void }) {
  return <div><h4>{title}</h4><div style={{ overflowX: 'auto' }}><table aria-label={title} style={{ width: 'auto', margin: '0 auto', borderCollapse: 'separate', borderSpacing: 4 }}><thead><tr><th scope="col">row / col</th>{values[0].map((_, c) => <th scope="col" key={c}>{c}</th>)}</tr></thead><tbody>{values.map((row, r) => <tr key={r}><th scope="row">{r}</th>{row.map((value, c) => <td key={c} style={{ textAlign: 'center', minWidth: 44, height: 44, padding: 4, border: highlight?.(r, c) ? `3px solid ${teal}` : '1px solid #d4d0c8', background: highlight?.(r, c) ? '#e1f1f0' : '#f8f5ed' }}>{buttons ? <button type="button" aria-label={`Inspect patch row ${r} column ${c}`} className="button secondary" style={{ minWidth: 36, padding: 4 }} onClick={() => buttons(r, c)}>{round(value)}</button> : round(value)}</td>)}</tr>)}</tbody></table></div></div>
}

function FunctionPlot({ fn, probe, xRange = [-3, 3], yRange = [-1, 6], tangent }: { fn: (x: number) => number; probe: number; xRange?: [number, number]; yRange?: [number, number]; tangent?: number }) {
  const [xmin, xmax] = xRange, [ymin, ymax] = yRange
  const sx = (x: number) => 48 + (x - xmin) / (xmax - xmin) * 360
  const sy = (y: number) => 285 - (y - ymin) / (ymax - ymin) * 245
  const coords = Array.from({ length: 121 }, (_, i) => { const x = xmin + (xmax - xmin) * i / 120; return `${sx(x)},${sy(fn(x))}` }).join(' ')
  const y = fn(probe), radius = (xmax - xmin) / 15
  const xTicks = xmin === -3 && xmax === 3 ? [-3, -2, -1, 0, 1, 2, 3] : [0.5, 2, 4, 6, 8]
  const yTicks = Number.isInteger(ymin) && Number.isInteger(ymax) ? Array.from({ length: ymax - ymin + 1 }, (_, i) => ymin + i) : [-0.5, 0, 0.5, 1]
  return <svg viewBox="0 0 440 330" style={{ width: '100%' }} role="img" aria-label="Function curve with movable probe; horizontal input axis and vertical output axis">
    {xTicks.map(x => <g key={x}><line x1={sx(x)} x2={sx(x)} y1="40" y2="285" stroke={x === 0 ? '#aaa397' : '#e5dfd6'} /><text x={sx(x)} y="305" textAnchor="middle" fontSize="12">{round(x)}</text></g>)}
    {yTicks.map(v => <g key={v}><line x1="48" x2="408" y1={sy(v)} y2={sy(v)} stroke={v === 0 ? '#aaa397' : '#e5dfd6'} /><text x="40" y={sy(v) + 4} textAnchor="end" fontSize="12">{round(v)}</text></g>)}
    <polyline points={coords} stroke={teal} strokeWidth="3" fill="none" />
    {tangent !== undefined && <line x1={sx(probe - radius)} x2={sx(probe + radius)} y1={sy(y - radius * tangent)} y2={sy(y + radius * tangent)} stroke={orange} strokeWidth="3" />}
    <circle cx={sx(probe)} cy={sy(y)} r="6" fill="white" stroke={orange} strokeWidth="3" />
    <text x="225" y="327" textAnchor="middle" fontSize="14">Input value</text><text x="48" y="23" fontSize="14">Output value</text>
  </svg>
}

function Hinge() {
  const [threshold, setThreshold] = useState(1), [probe, setProbe] = useState(2)
  return <div className="visual-grid"><FunctionPlot fn={x => hingeValue(x, threshold)} probe={probe} /><div className="projection-panel">
    <Slider label="Activation threshold" value={threshold} min={-2} max={2} step={0.25} onChange={setThreshold} />
    <Slider label="Hinge probe input" value={probe} min={-3} max={3} step={0.25} onChange={setProbe} />
    <p><output aria-label="Hinge probe result">Preactivation: {round(probe - threshold)}; output: {round(hingeValue(probe, threshold))}</output></p>
    <p className="panel-note">Scored reference: threshold 1. The open circle is exploratory.</p>
    <button className="button secondary" type="button" onClick={() => { setThreshold(1); setProbe(2) }}>Reset hinge</button>
  </div></div>
}

function Branches() {
  const [activated, setActivated] = useState(true), [probe, setProbe] = useState(-2)
  return <div className="visual-grid"><FunctionPlot fn={x => twoBranchValue(x, activated)} probe={probe} yRange={[-1, 4]} /><div className="projection-panel">
    <label><input type="checkbox" checked={activated} onChange={e => setActivated(e.target.checked)} /> Apply ReLU to both hidden branches</label>
    <Slider label="Two-branch probe input" value={probe} min={-3} max={3} step={0.25} onChange={setProbe} />
    <p>First branch: {round(activated ? Math.max(0, probe) : probe)}</p><p>Second branch: {round(activated ? Math.max(0, -probe) : -probe)}</p>
    <p><output aria-label="Two-branch output">Sum: {round(twoBranchValue(probe, activated))}</output></p>
    <p className="panel-note">Same weights in both modes. Scored probe inputs: −2 and 2.</p>
  </div></div>
}

function Shapes() {
  const [batch, setBatch] = useState(3)
  const widths = [2, 2, 1]
  return <div>
    <Slider label="Observation count" value={batch} min={1} max={6} onChange={setBatch} />
    <svg viewBox="0 0 660 295" style={{ width: '100%' }} role="img" aria-label={`${batch} rows pass through input width 2, hidden width 2, and output width 1; model parameters are shared`}>
      {widths.map((width, layer) => <g key={layer}><text x={75 + layer * 220} y="28" fontSize="17" textAnchor="middle">{['Input', 'Hidden', 'Output'][layer]}</text>
        {Array.from({ length: batch }, (_, row) => Array.from({ length: width }, (_, col) => <rect key={`${row}-${col}`} x={40 + layer * 220 + col * 34} y={50 + row * 32} width="27" height="24" fill={layer === 2 ? '#f3dfcd' : '#d5ebe9'} stroke={teal} />))}
        <text x={75 + layer * 220} y="272" textAnchor="middle" fontSize="16">({batch}, {width})</text>
        {layer < 2 && <g><line x1={130 + layer * 220} x2={224 + layer * 220} y1="126" y2="126" stroke={teal} strokeWidth="3" /><text x={177 + layer * 220} y="111" textAnchor="middle" fontSize="13">dense{layer === 0 ? ' + ReLU' : ''}</text><text x={177 + layer * 220} y="150" textAnchor="middle" fontSize="12">same map per row</text></g>}
      </g>)}
    </svg>
    <p><output aria-label="Shared network dimensions">Current batch: {batch} rows. Network: 2 → 2 → 1. Trainable parameters: {denseParameterCount(widths)} for every batch size.</output></p>
    <p className="panel-note">Each rectangle is one scalar entry, not a separate neuron with separate parameters. For the shape question the scored batch is 5; for parameter counting the batch size is irrelevant.</p>
  </div>
}

function Graph() {
  const [a, setA] = useState(2), [b, setB] = useState(3)
  const arrowId = useId()
  const trace = sharedBranchTrace(a, b)
  return <div className="visual-grid"><div><svg viewBox="0 0 450 300" style={{ width: '100%' }} role="img" aria-label="Computation graph of f equals a times b plus a, with two paths from a">
    <defs><marker id={arrowId} markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 z" fill={teal} /></marker></defs>
    {['M 98 74 L 189 122', 'M 98 216 L 189 151', 'M 258 137 L 329 137', 'M 85 49 Q 224 -18 354 110'].map(d => <path key={d} d={d} fill="none" stroke={teal} strokeWidth="2" markerEnd={`url(#${arrowId})`} />)}
    {[{ x: 65, y: 74, label: `a = ${a}` }, { x: 65, y: 216, label: `b = ${b}` }, { x: 225, y: 137, label: '×' }, { x: 365, y: 137, label: '+' }].map(node => <g key={node.label}><rect x={node.x - 33} y={node.y - 24} width="66" height="48" rx="10" fill="#eff7f5" stroke={teal} /><text x={node.x} y={node.y + 6} textAnchor="middle" fontSize="18">{node.label}</text></g>)}
    <text x="227" y="210" textAnchor="middle" fontSize="17">ab = {trace.product}</text><text x="365" y="210" textAnchor="middle" fontSize="17">f = {trace.value}</text>
    <text x="225" y="278" textAnchor="middle" fontSize="16">Two paths from a; one path from b</text>
  </svg></div><div className="projection-panel">
    <Slider label="Graph input a" value={a} min={-3} max={4} onChange={setA} /><Slider label="Graph input b" value={b} min={-3} max={4} onChange={setB} />
    <p><output aria-label="Graph gradient contributions">To a: product path {trace.productPath} + direct path {trace.directPath} = {trace.da}. To b: {trace.db}.</output></p>
    <button className="button secondary" type="button" onClick={() => { setA(2); setB(3) }}>Reset graph reference</button>
    <p className="panel-note">Fixed scored reference: a = 2, b = 3. The diagram shows one fresh derivative, not an accumulated gradient buffer.</p>
  </div></div>
}

function Chain() {
  const [u, setU] = useState(4)
  return <div className="visual-grid"><FunctionPlot fn={logRoot} probe={u} tangent={logRootDerivative(u)} xRange={[0.5, 8]} yRange={[-0.5, 1.2]} /><div className="projection-panel">
    <Slider label="Chain-rule input u" value={u} min={1} max={7} step={0.25} onChange={setU} />
    <p>u → square root → logarithm</p><p>Intermediate √u: {round(Math.sqrt(u))}</p>
    <p><output aria-label="Chain local derivatives">Local slopes: {round(1 / (2 * Math.sqrt(u)))} × {round(1 / Math.sqrt(u))}; total: {round(logRootDerivative(u))}</output></p>
    <p className="panel-note">The short orange line is a tangent, not a finite jump. Fixed scored reference: u = 4.</p>
  </div></div>
}

function PairedLoss() {
  const [all, setAll] = useState(false)
  const p = [1, 3], t = [2, 0]
  return <div className="visual-grid"><Matrix values={squaredErrorGrid(p, t)} title="Squared error: prediction rows × target columns" highlight={(r, c) => all || r === c} /><div className="projection-panel">
    <div className="question-controls"><button type="button" className="button secondary" aria-pressed={!all} onClick={() => setAll(false)}>Matching observations</button><button type="button" className="button secondary" aria-pressed={all} onClick={() => setAll(true)}>All possible pairs</button></div>
    <p>Prediction rows: 0 → 1, 1 → 3. Target columns: 0 → 2, 1 → 0.</p>
    <p><output aria-label="Comparison loss">{all ? 'All-pairs' : 'Paired'} mean squared error: {all ? allPairsMse(p, t) : pairedMse(p, t)}</output></p>
    <p className="panel-note">Teal borders show exactly which terms enter the average. The denominator is the number of selected terms.</p>
  </div></div>
}

function Bars({ values, labels, title, domain }: { values: number[]; labels: string[]; title: string; domain: [number, number] }) {
  const [min, max] = domain, y = (v: number) => 215 - (v - min) / (max - min) * 175
  return <svg viewBox="0 0 360 265" style={{ width: '100%' }} role="img" aria-label={title}>
    <text x="25" y="22" fontSize="15">{title}</text><line x1="38" x2="345" y1={y(0)} y2={y(0)} stroke="#8f8982" />
    {[min, 0, max].filter((v, i, a) => a.indexOf(v) === i).map(v => <text key={v} x="31" y={y(v) + 4} textAnchor="end" fontSize="12">{round(v)}</text>)}
    {values.map((value, i) => { const x = 55 + i * (280 / values.length); return <g key={i}><rect x={x} y={Math.min(y(value), y(0))} width={Math.min(44, 210 / values.length)} height={Math.max(1, Math.abs(y(value) - y(0)))} fill={[teal, orange, purple][i % 3]} /><text x={x + 20} y={value >= 0 ? y(value) - 7 : y(value) + 18} textAnchor="middle" fontSize="14">{round(value)}</text><text x={x + 20} y="254" textAnchor="middle" fontSize="14">{labels[i]}</text></g> })}
  </svg>
}

function Logits() {
  const [scale, setScale] = useState(1)
  const scores = [2, -1, 0].map(s => s * scale), probabilities = softmaxProbabilities(scores)
  return <div><Slider label="Class score multiplier" value={scale} min={0.5} max={2} step={0.25} onChange={setScale} /><div style={equalChartColumns}><Bars values={scores} labels={['A', 'B', 'C']} title="Raw class scores" domain={[-2, 4]} /><Bars values={probabilities} labels={['A', 'B', 'C']} title="Softmax probabilities" domain={[0, 1]} /></div>
    <p><output aria-label="Score probability comparison">Probabilities A, B, C: ({probabilities.map(round).join(', ')}). Predicted class: A.</output></p><p className="panel-note">Fixed scored reference: multiplier 1. Changing the multiplier rescales fixed scores; it does not train a model.</p>
  </div>
}

function Moments() {
  const [beta, setBeta] = useState(0.5)
  const g = [2, -2, 2, -2], m = exponentialMoments(g, beta)
  return <div><Slider label="Gradient memory beta" value={beta} min={0} max={0.9} step={0.1} onChange={setBeta} /><div style={equalChartColumns}><Bars values={g} labels={['1', '2', '3', '4']} title="Raw gradient by step" domain={[-2.5, 2.5]} /><Bars values={m} labels={['1', '2', '3', '4']} title="Running first moment by step" domain={[-2.5, 2.5]} /></div>
    <p><output aria-label="Running gradient moments">m₁ through m₄: ({m.map(round).join(', ')})</output></p><p className="panel-note">Both plots share the same vertical scale. Each slider change replays from m₀ = 0. Scored reference: β = 0.5 for the first two steps.</p>
  </div>
}

function Batches() {
  const [size, setSize] = useState(5), [shuffled, setShuffled] = useState(false)
  const order = shuffled ? [8, 2, 11, 0, 5, 7, 1, 10, 4, 9, 3, 6] : Array.from({ length: 12 }, (_, i) => i)
  const groups = batchGroups(order, size)
  return <div><Slider label="Mini-batch size" value={size} min={2} max={7} onChange={setSize} /><button type="button" className="button secondary" onClick={() => setShuffled(!shuffled)}>{shuffled ? 'Restore original order' : 'Use a shuffled order'}</button>
    <div aria-label="Mini-batch membership" style={{ marginTop: 16 }}>{groups.map((group, i) => <div key={i} style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 7, marginBottom: 10 }}><strong style={{ minWidth: 86 }}>Batch {i + 1}</strong>{group.map(id => <span key={id} style={{ border: `2px solid ${teal}`, borderRadius: 4, padding: '8px 12px', background: '#e8f3f1' }}>{id}</span>)}</div>)}</div>
    <p><output aria-label="Batch size summary">Batch sizes: ({groups.map(g => g.length).join(', ')}). Observations used: {groups.flat().length}; distinct IDs: {new Set(groups.flat()).size}.</output></p>
    <p className="panel-note">Reference questions use batch size 5. IDs are zero-based observation labels, not feature values. The final short batch is kept.</p>
  </div>
}

function Dropout() {
  const [training, setTraining] = useState(true)
  const values = dropoutValues([1, 2, 3, 4], [true, false, true, true], 0.25, training)
  return <div><div className="question-controls"><button type="button" className="button secondary" aria-pressed={training} onClick={() => setTraining(true)}>Training mask</button><button type="button" className="button secondary" aria-pressed={!training} onClick={() => setTraining(false)}>Evaluation mode</button></div>
    <div style={equalChartColumns}><Bars values={[1, 2, 3, 4]} labels={['1', '2', '3', '4']} title="Input activations" domain={[0, 6]} /><Bars values={values} labels={['1', '2', '3', '4']} title={training ? 'Training output (fixed mask)' : 'Evaluation output'} domain={[0, 6]} /></div>
    <p><output aria-label="Dropout output">{training ? 'Training' : 'Evaluation'} activations: ({values.map(round).join(', ')})</output></p><p className="panel-note">These are layer modes; no optimization or gradient computation is performed in this demonstration.</p>
  </div>
}

function Correlation() {
  const [row, setRow] = useState(0), [col, setCol] = useState(0)
  const response = validCorrelation(contrastImage, contrastKernel)
  const patch = contrastImage.slice(row, row + 2).map(r => r.slice(col, col + 2))
  const terms = patch.flat().map((value, i) => `${contrastKernel.flat()[i]}×${value}`)
  return <div><div className="visual-grid"><Matrix values={contrastImage} title="Input image intensity" highlight={(r, c) => r >= row && r < row + 2 && c >= col && c < col + 2} /><Matrix values={contrastKernel} title="Shared filter weights" /></div>
    <div className="visual-grid"><Matrix values={response} title="Response map: select a location" highlight={(r, c) => r === row && c === col} buttons={(r, c) => { setRow(r); setCol(c) }} /><div className="projection-panel">
      <Slider label="Patch start row" value={row} min={0} max={2} onChange={setRow} /><Slider label="Patch start column" value={col} min={0} max={2} onChange={setCol} />
      <p><output aria-label="Selected filter calculation">({terms.join(') + (')}) = {response[row][col]}</output></p><p className="panel-note">The input outline marks the selected 2 × 2 patch. Fixed local-response reference: row 0, column 0. Every response-map cell uses the same four filter entries.</p>
    </div></div>
  </div>
}

function Pooling() {
  const [moved, setMoved] = useState(false)
  const values = poolingImage.map(row => [...row])
  if (moved) { values[0][0] = 5; values[0][1] = 1 }
  const pooled = maxPool(values, [2, 2])
  return <div><div className="visual-grid"><Matrix values={values} title="Image: complete blocks are outlined" highlight={(r, c) => r < 2 && c < 4} /><div>
    <Matrix values={pooled} title="Pooled maximum values" />
    <button type="button" className="button secondary" onClick={() => setMoved(!moved)}>{moved ? 'Restore peak position' : 'Move peak within first block'}</button>
    <p><output aria-label="Pooled output values">Output: {JSON.stringify(pooled)}. Shape: (1, 2).</output></p>
    <p className="panel-note">Block A: rows 0–1, columns 0–1. Block B: rows 0–1, columns 2–3. Unoutlined row 2 and column 4 are discarded, including their 99s. The button only swaps 1 and 5 inside block A.</p>
  </div></div></div>
}

const titles: Record<NeuralFoundationsVisualDataset['mode'], string> = {
  hinge: 'A movable rectified feature', branches: 'Combine two hidden features', shapes: 'Batch axis and feature axis', graph: 'A shared-variable computation graph', chain: 'A chain of local derivatives', pairedLoss: 'Which comparisons enter the loss?', logits: 'Scores and probabilities', moments: 'Replay a running gradient average', batches: 'Partition an epoch', dropout: 'Compare layer modes', correlation: 'A shared local contrast detector', pooling: 'Complete pooling blocks and discarded borders',
}
export function NeuralFoundationsVisual({ dataset }: { dataset: NeuralFoundationsVisualDataset }) {
  return <section className="data-table"><h3 className="panel-title">{titles[dataset.mode]}</h3><p className="table-caption">{dataset.caption}</p>
    {dataset.mode === 'hinge' ? <Hinge /> : dataset.mode === 'branches' ? <Branches /> : dataset.mode === 'shapes' ? <Shapes /> : dataset.mode === 'graph' ? <Graph /> : dataset.mode === 'chain' ? <Chain /> : dataset.mode === 'pairedLoss' ? <PairedLoss /> : dataset.mode === 'logits' ? <Logits /> : dataset.mode === 'moments' ? <Moments /> : dataset.mode === 'batches' ? <Batches /> : dataset.mode === 'dropout' ? <Dropout /> : dataset.mode === 'correlation' ? <Correlation /> : <Pooling />}
  </section>
}
