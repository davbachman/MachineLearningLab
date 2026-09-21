import { useState } from 'react'
import type { SequenceFoundationsVisualDataset } from '../../data/sequenceFoundationsAssignments'
import { attentionMatrix, causalRow, cleanPixels, edgeImage, edgeKernel, filterAt, headShapes, memoryValues, mse, noisyPixels, pairCompression, poolTwo, poolingImage, restoredPixels, retrieve, shiftedWindow, simpleTokens } from '../../lib/sequenceFoundations'

function Grid({ values, title, selected }: { values: number[][]; title: string; selected?: [number, number] }) {
  const width = values[0].length * 66 + 50, height = values.length * 60 + 48
  return <figure style={{ margin: 0 }}><figcaption>{title}</figcaption><svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', maxWidth: 390 }} role="img" aria-label={title}>
    {values[0].map((_, column) => <text key={column} x={column * 66 + 66} y="20" textAnchor="middle" fontSize="12">col {column}</text>)}
    {values.map((row, i) => <g key={i}><text x="2" y={i * 60 + 61} fontSize="12">{i}</text>{row.map((value, j) => <g key={j}><rect x={34 + j * 66} y={30 + i * 60} width="62" height="56" fill={selected && i >= selected[0] && i < selected[0] + 2 && j >= selected[1] && j < selected[1] + 2 ? '#d4edf0' : '#f5f1e8'} stroke="#827c71" /><text x={65 + j * 66} y={64 + i * 60} textAnchor="middle" fontSize="16">{Number(value.toFixed(3))}</text></g>)}</g>)}
  </svg></figure>
}
function Slider({ label, value, set, min = 0, max, step = 1 }: { label: string; value: number; set: (n: number) => void; min?: number; max: number; step?: number }) {
  return <label className="slider-shell"><span>{label}: {value}</span><input className="range-input" type="range" aria-label={label} min={min} max={max} step={step} value={value} onChange={e => set(Number(e.target.value))} /></label>
}
function FilterExplorer() {
  const [row, setRow] = useState(0), [column, setColumn] = useState(0)
  return <><div className="visual-grid"><Grid values={edgeImage} title="Input image; pixel values 0 or 1" selected={[row, column]} /><Grid values={edgeKernel} title="Shared filter; multiply matching entries, then sum" /></div><Slider label="Patch top row" value={row} set={setRow} max={2} /><Slider label="Patch left column" value={column} set={setColumn} max={2} /><output aria-label="Filter response">Response at ({row}, {column}) = {filterAt(edgeImage, edgeKernel, row, column)}</output><p className="panel-note">Stride 1, no padding, no bias; filter is not flipped. Scored reference: top row 0, compare left columns 0, 1, and 2. The same four weights are reused everywhere.</p></>
}
function PoolExplorer() {
  const [mode, setMode] = useState<'max' | 'mean'>('max')
  return <><label>Pooling rule <select aria-label="Pooling rule" value={mode} onChange={e => setMode(e.target.value as 'max' | 'mean')}><option value="max">Maximum</option><option value="mean">Mean</option></select></label><div className="visual-grid"><Grid values={poolingImage} title="4 × 4 input; nonoverlapping 2 × 2 windows" /><Grid values={poolTwo(poolingImage, mode)} title={`${mode === 'max' ? 'Maximum' : 'Mean'} output; stride 2`} /></div><p className="panel-note">Scored reference: maximum pooling. The mean option explores a different summary of each identical window.</p></>
}
function PixelStrip({ values, title }: { values: number[]; title: string }) {
  return <figure style={{ margin: '1rem 0' }}><figcaption>{title}</figcaption><svg viewBox="0 0 360 88" role="img" aria-label={`${title}: ${values.join(', ')}`} style={{ width: '100%', maxWidth: 420 }}>{values.map((v, i) => <g key={i}><rect x={i * 90 + 3} y="5" width="82" height="52" fill={`rgb(${v * 255},${v * 255},${v * 255})`} stroke="#545454" /><text x={i * 90 + 44} y="79" textAnchor="middle" fontSize="15">{v.toFixed(2)}</text></g>)}</svg></figure>
}
function DenoiseExplorer() {
  const [target, setTarget] = useState<'clean' | 'noisy'>('clean')
  const truth = target === 'clean' ? cleanPixels : noisyPixels
  return <><PixelStrip values={cleanPixels} title="Original clean image" /><div className="visual-grid"><PixelStrip values={noisyPixels} title="Candidate A: copy the corrupted input" /><PixelStrip values={restoredPixels} title="Candidate B: reconstruct missing detail" /></div><label>Loss target <select aria-label="Reconstruction target" value={target} onChange={e => setTarget(e.target.value as 'clean' | 'noisy')}><option value="clean">Clean original</option><option value="noisy">Corrupted input</option></select></label><p><output aria-label="Reconstruction losses">Mean squared errors: A = {mse(noisyPixels, truth).toFixed(3)}; B = {mse(restoredPixels, truth).toFixed(3)}</output></p><p className="panel-note">Loss is the mean of four squared pixel errors. Scored reference: clean target. No training is performed; these are two fixed candidate outputs.</p></>
}
function BottleneckExplorer() {
  const [variant, setVariant] = useState('A')
  const pixels = variant === 'A' ? [0, 1, .2, .8] : [1, 0, .8, .2]
  const compressed = pairCompression(pixels)
  return <><label>Input image <select aria-label="Compression input" value={variant} onChange={e => setVariant(e.target.value)}><option>A</option><option>B</option></select></label><PixelStrip values={pixels} title={`Image ${variant}`} /><p><output aria-label="Compressed representation">Two-number representation: [{compressed.code.join(', ')}]</output></p><PixelStrip values={compressed.reconstruction} title="Reconstruction: repeat each pair mean twice" /><p className="panel-note">Encoder averages pixels 0–1 and 2–3. Decoder repeats each average. Scored comparison: A = [0, 1, 0.2, 0.8], B = [1, 0, 0.8, 0.2]. This simple fixed bottleneck is illustrative, not a learned network.</p></>
}
function TokenExplorer() {
  const [mode, setMode] = useState('words'), [sentence, setSentence] = useState('blue kite!')
  const tokens = mode === 'words' ? simpleTokens(sentence) : [...sentence]
  return <><label>Text <select aria-label="Tokenization text" value={sentence} onChange={e => setSentence(e.target.value)}><option>blue kite!</option><option>blue kite.</option><option>Blue kite!</option></select></label>{' '}<label>Units <select aria-label="Tokenization units" value={mode} onChange={e => setMode(e.target.value)}><option value="words">Words + punctuation</option><option value="characters">Characters (including spaces)</option></select></label><ol aria-label="Tokens in sequence order" style={{ display: 'flex', flexWrap: 'wrap', gap: '.5rem', padding: '1rem 0', listStyle: 'none' }}>{tokens.map((token, i) => <li key={i} style={{ minWidth: 42, border: '1px solid #aebec0', borderRadius: 6, padding: '.4rem .7rem', textAlign: 'center', background: '#eef5f5' }}><span style={{ display: 'block', fontSize: '.7rem' }}>pos {i}</span><code style={{ fontSize: '1.1rem' }}>{token === ' ' ? '␠' : token}</code></li>)}</ol><output aria-label="Token count">Sequence length: {tokens.length}</output><p className="panel-note">Words are case-sensitive; punctuation is separate; spaces are omitted only in word mode. Scored reference: text “blue kite!”; compare both unit choices. The symbol ␠ marks one space character.</p></>
}
function WindowExplorer() {
  const [start, setStart] = useState(0), window = shiftedWindow(start)
  return <><p>Token stream: &lt;bos&gt;, Ava, found, a, blue, kite, ., &lt;eos&gt;</p><Slider label="Window start index" value={start} set={setStart} max={3} /><table><thead><tr><th>Window position</th><th>Input token</th><th>Next-token target</th></tr></thead><tbody>{window.inputs.map((token, i) => <tr key={i}><th>{i}</th><td>{token}</td><td>{window.targets[i]}</td></tr>)}</tbody></table><p className="panel-note">Four input positions; targets are shifted one stream position right. Scored reference: start index 0. Changing the start selects a new slice, not new token IDs.</p></>
}
function EmbeddingExplorer() {
  const [position, setPosition] = useState(0), positions = [[0, 1], [1, 0], [-1, 1]]
  return <><p>The token “kite” has ID 7 and vector [2, −1]. Position vectors are position 0: [0, 1], position 1: [1, 0], position 2: [−1, 1].</p><Slider label="Token position" value={position} set={setPosition} max={2} /><Grid values={[[2, -1], positions[position], [2 + positions[position][0], -1 + positions[position][1]]]} title="Rows: token vector, position vector, combined vector" /><p className="panel-note">Coordinates label embedding channels, not numeric token IDs. Scored reference: compare positions 0 and 1 for the same token.</p></>
}
function WeightBars({ weights }: { weights: number[] }) {
  return <svg viewBox="0 0 440 210" role="img" aria-label={`Key probabilities: ${weights.map(w => w.toFixed(3)).join(', ')}`} style={{ width: '100%', maxWidth: 500 }}>{[0, .5, 1].map(v => <g key={v}><line x1="34" x2="430" y1={170 - 110 * v} y2={170 - 110 * v} stroke="#ddd" /><text x="2" y={175 - 110 * v} fontSize="12">{v}</text></g>)}{weights.map((w, i) => <g key={i}><rect x={50 + i * 95} y={170 - w * 110} width="54" height={w * 110} fill="#287c87" /><text x={77 + i * 95} y={158 - w * 110} textAnchor="middle" fontSize="14">{w.toFixed(3)}</text><text x={77 + i * 95} y="193" textAnchor="middle" fontSize="13">Key {i}</text></g>)}<text x="36" y="22" fontSize="14">Attention probability (row sums to 1)</text></svg>
}
function RetrievalExplorer() {
  const [qx, setQx] = useState(1), [qy, setQy] = useState(0), [third, setThird] = useState(5)
  const result = retrieve([qx, qy], [2, 8, third])
  return <><p>Keys 0, 1, 2: [1, 0], [0, 1], [1, 1]. Scalar values: 2, 8, {third}. Scores are query·key / √2.</p><WeightBars weights={result.weights} /><Slider label="Query first coordinate" value={qx} set={setQx} min={-2} max={2} step={.5} /><Slider label="Query second coordinate" value={qy} set={setQy} min={-2} max={2} step={.5} /><Slider label="Value at key 2" value={third} set={setThird} max={10} /><p><output aria-label="Retrieved value">Weighted output: {result.result.toFixed(3)}</output></p><p className="panel-note">Scored reference: query [1, 0], values [2, 8, 5]; then change only value 2 from 5 to 9. Controls do not train or change the keys.</p></>
}
function MatrixExplorer() {
  const [row, setRow] = useState(0)
  return <><p>Queries 0, 1, 2: [1, 0], [0, 1], [1, 1]. Keys are the same three vectors; values are [2, 8, 5]. Rows are queries; columns are keys.</p><Grid values={attentionMatrix} title="Attention probability matrix; three decimals shown" /><Slider label="Inspect query row" value={row} set={setRow} max={2} /><WeightBars weights={attentionMatrix[row]} /><output aria-label="Selected attention output">Output for query {row}: {attentionMatrix[row].reduce((s, w, i) => s + w * memoryValues[i], 0).toFixed(3)}</output><p className="panel-note">Scored reference: query row 1. Normalization is across keys within a row, not down the query column.</p></>
}
function CausalExplorer() {
  const [row, setRow] = useState(1), [masked, setMasked] = useState(true)
  return <><Slider label="Query position" value={row} set={setRow} max={3} /><label><input type="checkbox" aria-label="Apply causal mask" checked={masked} onChange={e => setMasked(e.target.checked)} /> Apply causal mask before normalization</label><Grid values={[0, 1, 2, 3].map(i => [0, 1, 2, 3].map(j => Number(!masked || j <= i)))} title="Allowed keys: 1 = available, 0 = blocked; rows query positions" /><WeightBars weights={causalRow(row, masked)} /><p className="panel-note">Raw score rows: [2,1,0,−1], [0,2,1,0], [1,0,2,1], [0,1,0,2]. Scored reference: causal mask enabled; compare query positions 0 and 2. Position indices start at 0.</p></>
}
function HeadsExplorer() {
  const [heads, setHeads] = useState(2), shapes = headShapes(3, 6, 8, heads)
  return <><label>Number of heads <select aria-label="Attention head count" value={heads} onChange={e => setHeads(Number(e.target.value))}>{[1, 2, 4, 8].map(n => <option key={n}>{n}</option>)}</select></label><p>Input shape stays (batch 3, sequence 6, channels 8).</p><div style={{ display: 'flex', gap: '.3rem' }}>{Array.from({ length: heads }, (_, i) => <div key={i} style={{ flex: 1, background: '#d4edf0', border: '1px solid #357985', padding: '.5rem', textAlign: 'center' }}>H{i}<br />{shapes.headSize} channels</div>)}</div><p><output aria-label="Head shapes">Per-head Q/K/V tensor: ({shapes.split.join(', ')}). Weights: ({shapes.weights.join(', ')}). Recombined output: ({shapes.output.join(', ')}).</output></p><p className="panel-note">Scored reference: 2 heads. Splitting redistributes the fixed eight channels; it does not duplicate all eight channels into every head.</p></>
}
export function SequenceFoundationsVisual({ dataset }: { dataset: SequenceFoundationsVisualDataset }) {
  return <section className="data-table"><h3 className="panel-title">{dataset.title}</h3><p className="table-caption">{dataset.caption}</p>
    {dataset.mode === 'filter' ? <FilterExplorer /> : dataset.mode === 'pool' ? <PoolExplorer /> : dataset.mode === 'denoise' ? <DenoiseExplorer /> : dataset.mode === 'bottleneck' ? <BottleneckExplorer /> : dataset.mode === 'tokenize' ? <TokenExplorer /> : dataset.mode === 'window' ? <WindowExplorer /> : dataset.mode === 'embedding' ? <EmbeddingExplorer /> : dataset.mode === 'retrieve' ? <RetrievalExplorer /> : dataset.mode === 'matrix' ? <MatrixExplorer /> : dataset.mode === 'mask' ? <CausalExplorer /> : dataset.mode === 'heads' ? <HeadsExplorer /> : <div className="projection-panel">{dataset.notes?.map(note => <p key={note}>{note}</p>)}</div>}
  </section>
}
