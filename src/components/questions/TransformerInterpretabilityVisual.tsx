import { useState } from 'react'
import type { TransformerInterpretabilityVisualDataset } from '../../data/transformerInterpretabilityAssignments'
import { addScaled, attentionPattern, attentionValues, dictionaryDirections, normalizeChannels, samplingProbabilities, sparseMetrics, steerResidual, steeringTrials, type Vector2 } from '../../lib/transformerInterpretability'

const fmt = (values: number[]) => `(${values.map(x => x.toFixed(2)).join(', ')})`
const colors = ['#167c86', '#bd682b', '#783d83', '#3c628d']

function Slider({ label, value, min = 0, max = 2, step = .1, onChange }: { label: string; value: number; min?: number; max?: number; step?: number; onChange: (value: number) => void }) {
  return <label className="slider-shell"><span>{label}: {value}</span><input className="range-input" aria-label={label} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} /></label>
}

function Bars({ values, labels, title, maximum = 1 }: { values: number[]; labels: string[]; title: string; maximum?: number }) {
  return <svg viewBox="0 0 420 230" style={{ width: '100%', maxWidth: 540 }} role="img" aria-label={title}>
    <text x="12" y="20" fontSize="14">{title}</text>
    {[0, .5, 1].map(r => <g key={r}><line x1="40" x2="410" y1={185 - r * 140} y2={185 - r * 140} stroke="#ddd5c9" /><text x="34" y={190 - r * 140} textAnchor="end" fontSize="12">{(maximum * r).toFixed(1)}</text></g>)}
    {values.map((v, i) => <g key={labels[i]}><rect x={65 + i * 86} y={185 - v / maximum * 140} width="43" height={v / maximum * 140} fill={colors[i]} /><text x={87 + i * 86} y={175 - v / maximum * 140} textAnchor="middle" fontSize="14">{v.toFixed(3)}</text><text x={87 + i * 86} y="209" textAnchor="middle" fontSize="13">{labels[i]}</text></g>)}
  </svg>
}

function VectorPlot({ points, max = 6 }: { points: { label: string; vector: Vector2 }[]; max?: number }) {
  const px = (x: number) => 65 + x * 280 / max
  const py = (y: number) => 315 - y * 280 / max
  return <svg viewBox="0 0 420 365" style={{ width: '100%', maxWidth: 510 }} role="img" aria-label="Two-channel activation space; labeled arrows share the origin">
    {[0, 1, 2, 3, 4, 5, 6].filter(x => x <= max).map(x => <g key={x}><line x1={px(x)} x2={px(x)} y1="35" y2="315" stroke="#e6dfd4" /><line x1="65" x2="345" y1={py(x)} y2={py(x)} stroke="#e6dfd4" /><text x={px(x)} y="337" textAnchor="middle" fontSize="13">{x}</text><text x="50" y={py(x) + 5} textAnchor="end" fontSize="13">{x}</text></g>)}
    <text x="205" y="360" fontSize="14">Channel 1</text><text x="10" y="20" fontSize="14">Channel 2</text>
    {points.map((p, i) => <g key={p.label}><line x1={px(0)} y1={py(0)} x2={px(p.vector[0])} y2={py(p.vector[1])} stroke={colors[i]} strokeWidth="3" strokeDasharray={i % 2 ? '7 4' : undefined} /><circle cx={px(p.vector[0])} cy={py(p.vector[1])} r={5 + i} fill="white" stroke={colors[i]} strokeWidth="2" /><text x={px(p.vector[0]) + 8} y={py(p.vector[1]) - 10 + i * 13} fontSize="14">{p.label}</text></g>)}
  </svg>
}

function ResidualExplorer() {
  const [attention, setAttention] = useState(1), [mlp, setMlp] = useState(1)
  const middle = addScaled([2, 1], [-1, 2], attention), final = addScaled(middle, [.5, -.5], mlp)
  return <div className="visual-grid"><VectorPlot points={[{ label: 'Input', vector: [2, 1] }, { label: 'After attention', vector: middle }, { label: 'After MLP', vector: final }]} /><div className="projection-panel"><Slider label="Attention update scale" value={attention} onChange={setAttention} /><Slider label="MLP update scale" value={mlp} onChange={setMlp} /><p>Input (2, 1); attention update (−1, 2); MLP update (0.5, −0.5).</p><output aria-label="Residual result">Final residual: {fmt(final)}</output><p className="panel-note">Toy updates are supplied constants, not outputs of a trained network. Both scales 1 is the scored reference; explore zero to see what a residual path preserves.</p></div></div>
}

function NormalizationExplorer() {
  const [coordinate, setCoordinate] = useState(10)
  const tokens = [[1, 2, 3], [coordinate, 10, 13]]
  return <div><Slider label="Second token, first channel" value={coordinate} min={4} max={16} step={1} onChange={setCoordinate} /><table><thead><tr><th>Token position</th><th>Original channels</th><th>Normalized channels</th></tr></thead><tbody>{tokens.map((t, i) => <tr key={i}><th>{i}</th><td>{fmt(t)}</td><td><output aria-label={`Normalized token ${i}`}>{fmt(normalizeChannels(t))}</output></td></tr>)}</tbody></table><p>Each row is normalized independently across its channels: subtract that row’s mean and divide by its population standard deviation. For this toy calculation, omit epsilon and learned scale/bias.</p><p className="panel-note">Scored comparison: coordinate 10 versus 16. Token 0 stays fixed in both cases.</p></div>
}

function WindowExplorer() {
  const [start, setStart] = useState(2)
  const stream = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
  return <div><Slider label="Window start position" value={start} max={3} step={1} onChange={setStart} /><table><thead><tr><th>Stream position</th>{stream.map((_, i) => <th key={i}>{i}</th>)}</tr></thead><tbody><tr><th>Token</th>{stream.map((token, i) => <td key={i} style={{ background: i >= start && i < start + 3 ? '#d3edf0' : undefined }}>{token}</td>)}</tr><tr><th>Input → next-token target</th>{stream.map((_, i) => <td key={i}>{i >= start && i < start + 3 ? `${stream[i]} → ${stream[i + 1]}` : '—'}</td>)}</tr></tbody></table><output aria-label="Window targets">Targets: {stream.slice(start + 1, start + 4).join(', ')}</output><p className="panel-note">Context length is 3. Each highlighted input has its own next-token target. Scored reference: start position 2.</p></div>
}

function SamplingExplorer() {
  const [temperature, setTemperature] = useState(1), [k, setK] = useState(2)
  const probabilities = samplingProbabilities([2, 1, 0, -1], temperature, k)
  return <div className="visual-grid"><Bars values={probabilities} labels={['A', 'B', 'C', 'D']} title="Next-token probability" /><div className="projection-panel"><Slider label="Sampling temperature" value={temperature} min={.5} max={2} step={.5} onChange={setTemperature} /><Slider label="Top-k cutoff" value={k} min={1} max={4} step={1} onChange={setK} /><p>Fixed token logits A, B, C, D = (2, 1, 0, −1). Divide retained logits by temperature, exponentiate, and normalize.</p><output aria-label="Sampling probabilities">{fmt(probabilities)}</output><p className="panel-note">Scored comparison: k = 2, temperatures 1 and 0.5. No training occurs; these are distribution calculations, not random samples.</p></div></div>
}

function AttentionExplorer() {
  const [query, setQuery] = useState(2)
  const result = attentionPattern[query].reduce((sum, w, i) => sum + w * attentionValues[i], 0)
  return <div><Slider label="Query position" value={query} max={3} step={1} onChange={setQuery} /><table><caption>One toy head: rows = query positions; columns = key positions</caption><thead><tr><th>Query ↓ / Key →</th>{attentionValues.map((_, i) => <th key={i}>{i}</th>)}</tr></thead><tbody>{attentionPattern.map((row, i) => <tr key={i}><th>{i}{i === query ? ' ← selected' : ''}</th>{row.map((w, j) => <td key={j} style={{ background: `rgba(22,124,134,${w * .6})`, fontWeight: i === query ? 700 : 400 }}>{w.toFixed(2)}{j > i ? ' (masked)' : ''}</td>)}</tr>)}</tbody></table><p>Scalar values at keys 0–3: (2, −1, 4, 0).</p><output aria-label="Attention weighted value">Weighted value for query {query}: {result.toFixed(2)}</output><p className="panel-note">Scored reference: query 2. Weights sum to 1 across keys in each row. This toy head has no dropout.</p></div>
}

function DirectionExplorer() {
  const [angle, setAngle] = useState(0)
  const points: { label: string; vector: Vector2 }[] = [{ label: 'P1', vector: [2, 3] }, { label: 'P2', vector: [4, 1] }, { label: 'O1', vector: [0, 3] }, { label: 'O2', vector: [2, 1] }]
  const direction: Vector2 = [Math.cos(angle * Math.PI / 180), Math.sin(angle * Math.PI / 180)]
  return <div className="visual-grid"><VectorPlot points={points} max={4} /><div className="projection-panel"><p>P1 and P2 have the positive label; O1 and O2 have the other label. Coordinates are measured activations in a toy two-channel space.</p><Slider label="Projection angle in degrees" value={angle} max={90} step={15} onChange={setAngle} /><p>Unit probe direction: {fmt(direction)}</p><table><thead><tr><th>Context</th><th>Projection</th></tr></thead><tbody>{points.map(p => <tr key={p.label}><th>{p.label}</th><td>{(p.vector[0] * direction[0] + p.vector[1] * direction[1]).toFixed(2)}</td></tr>)}</tbody></table><p className="panel-note">Scored reference: angle 0°. Construct the label-associated direction from mean(P) − mean(O), then normalize. Rotating the probe is exploration, not an intervention in a model.</p></div></div>
}

function DictionaryExplorer() {
  const [coefficients, setCoefficients] = useState([0, 0, 2])
  const metrics = sparseMetrics(coefficients)
  return <div className="visual-grid"><VectorPlot points={[{ label: 'Target', vector: [1.2, 1.6] }, { label: 'Reconstruction', vector: metrics.reconstruction }]} /><div className="projection-panel">{coefficients.map((value, i) => <Slider key={i} label={`Feature ${i + 1} coefficient`} value={value} onChange={v => setCoefficients(old => old.map((x, j) => i === j ? v : x))} />)}<p>Unit decoder directions: {dictionaryDirections.map(fmt).join('; ')}. Reconstruction = sum of coefficient × direction. Target = (1.2, 1.6).</p><output aria-label="Sparse reconstruction metrics">Reconstruction {fmt(metrics.reconstruction)}; MSE {metrics.mse.toFixed(3)}; L1 {metrics.l1.toFixed(2)}; active features {metrics.l0}.</output><p className="panel-note">Scored reference: coefficients (0, 0, 2), compared with (1.2, 1.6, 0). Controls are exploratory; no encoder is being trained.</p></div></div>
}

function SparsityTradeoff() {
  const [lambda, setLambda] = useState(.2)
  const candidates = [{ name: 'Exact', z: [1.2, 1.6, 0] }, { name: 'Sparse/shrunk', z: [0, 0, 1.5] }]
  return <div><Slider label="L1 penalty strength" value={lambda} max={.3} step={.025} onChange={setLambda} /><table><thead><tr><th>Candidate code</th><th>Reconstruction</th><th>MSE</th><th>L1</th><th>MSE + λL1</th></tr></thead><tbody>{candidates.map(c => { const m = sparseMetrics(c.z); return <tr key={c.name}><th>{c.name}: {fmt(c.z)}</th><td>{fmt(m.reconstruction)}</td><td>{m.mse.toFixed(3)}</td><td>{m.l1.toFixed(2)}</td><td>{(m.mse + lambda * m.l1).toFixed(3)}</td></tr> })}</tbody></table><p>The same target (1.2, 1.6) and unit dictionary directions (1, 0), (0, 1), (0.6, 0.8) are used for both candidates. MSE averages the two squared coordinate errors.</p><p className="panel-note">Scored reference: λ = 0.2. Changing λ reranks these fixed candidate codes; it does not refit them.</p></div>
}

function SteeringExplorer() {
  const [factor, setFactor] = useState(2)
  const original: Vector2 = [3, 2], direction: Vector2 = [.6, .8]
  const result = steerResidual(original, 2, direction, factor)
  return <div className="visual-grid"><VectorPlot points={[{ label: 'Baseline', vector: original }, { label: 'Edited', vector: result }]} /><div className="projection-panel"><Slider label="Feature contribution multiplier" value={factor} onChange={setFactor} /><p>Residual h = (3, 2), coefficient z = 2, unit direction d = (0.6, 0.8). Edit h′ = h + (factor − 1)zd, holding the remainder h − zd fixed.</p><output aria-label="Steered residual">Edited residual: {fmt(result)}</output><p className="panel-note">Scored reference: factor 0 (ablation). Factor 1 is baseline and factor 2 adds one extra contribution. Re-encoding an edited residual need not return exactly factor × z in a nonorthogonal learned dictionary.</p></div></div>
}

function SteeringQuality() {
  const [trial, setTrial] = useState(1)
  return <div className="visual-grid"><Bars values={steeringTrials.map(t => t.probability)} labels={['Base', 'Moderate', 'Strong', 'Random']} title="Toy held-out mean P(period)" /><div className="projection-panel"><label>Inspect condition <select aria-label="Steering condition" value={trial} onChange={e => setTrial(Number(e.target.value))}>{steeringTrials.map((t, i) => <option key={t.name} value={i}>{t.name}</option>)}</select></label><p><output aria-label="Steering quality cost">{steeringTrials[trial].name}: cross-entropy {steeringTrials[trial].loss.toFixed(2)}; increase above baseline {(steeringTrials[trial].loss - 1).toFixed(2)}.</output></p><table><thead><tr><th>Condition</th><th>Cross-entropy</th></tr></thead><tbody>{steeringTrials.map(t => <tr key={t.name}><th>{t.name}</th><td>{t.loss.toFixed(2)}</td></tr>)}</tbody></table><p className="panel-note">These are supplied illustrative measurements, not checkpoint results. Fixed scored rule: maximize mean P(period), subject to cross-entropy increase ≤ 0.10. All rows use the same held-out examples.</p></div></div>
}

function StaticPanel({ mode }: { mode: TransformerInterpretabilityVisualDataset['mode'] }) {
  if (mode === 'tying') return <table><thead><tr><th>Operation</th><th>Shape / role</th></tr></thead><tbody><tr><th>Token lookup</th><td>5 vocabulary rows × 3 channels</td></tr><tr><th>Final token vector</th><td>3 channels</td></tr><tr><th>Vocabulary scores</th><td>Dot product with each of the same 5 rows</td></tr></tbody></table>
  if (mode === 'split') return <table><thead><tr><th>Policy</th><th>Training</th><th>Validation</th></tr></thead><tbody><tr><th>X: split within a story</th><td>A complete; B beginning</td><td>B ending; C complete</td></tr><tr><th>Y: split whole stories</th><td>A complete; B complete</td><td>C complete</td></tr></tbody></table>
  if (mode === 'lens') return <table><thead><tr><th>Representation decoded with same normalization/head</th><th>Logits for A, B, C</th></tr></thead><tbody><tr><th>After embeddings</th><td>(1, 2, 0)</td></tr><tr><th>After block 0</th><td>(3, 1, 0)</td></tr><tr><th>After final block</th><td>(0, 2, 4)</td></tr></tbody></table>
  if (mode === 'scale') return <table><thead><tr><th>Description</th><th>Coefficient</th><th>Direction</th><th>Contribution</th></tr></thead><tbody><tr><th>Original</th><td>2</td><td>(0.6, 0.8)</td><td>(1.2, 1.6)</td></tr><tr><th>Rescaled</th><td>1</td><td>(1.2, 1.6)</td><td>(1.2, 1.6)</td></tr></tbody></table>
  return <table><thead><tr><th>Data split</th><th>Allowed use in this design</th></tr></thead><tbody><tr><th>Training</th><td>Fit the GPT and SAE</td></tr><tr><th>Calibration</th><td>Select the feature and candidate prompt</td></tr><tr><th>Untouched final test</th><td>Evaluate target effects, random controls, and quality cost</td></tr></tbody></table>
}

export function TransformerInterpretabilityVisual({ dataset }: { dataset: TransformerInterpretabilityVisualDataset }) {
  const panels = { residual: ResidualExplorer, normalization: NormalizationExplorer, windows: WindowExplorer, sampling: SamplingExplorer, attention: AttentionExplorer, direction: DirectionExplorer, dictionary: DictionaryExplorer, sparsity: SparsityTradeoff, steering: SteeringExplorer, quality: SteeringQuality }
  const Panel = dataset.mode in panels ? panels[dataset.mode as keyof typeof panels] : null
  return <section className="data-table transformer-visual"><style>{`.transformer-visual table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: .92rem; } .transformer-visual th, .transformer-visual td { padding: .65rem .45rem; text-align: left; border-bottom: 1px solid #ddd5c9; } .transformer-visual th { font-weight: 650; } .transformer-visual caption { text-align: left; margin-bottom: .6rem; }`}</style><h3 className="panel-title">{dataset.title}</h3><p className="table-caption">{dataset.caption}</p>{Panel ? <Panel /> : <StaticPanel mode={dataset.mode} />}</section>
}
