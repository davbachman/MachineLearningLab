import { useId } from 'react'
import type { GradientVisualDataset } from '../../data/multipleChoiceDatasets'
import { descentArrows, descentRuns, descentStart, formatLoss, nonconvexLoss, singleParameterLoss } from '../../lib/gradientDescent'
import type { Vec2 } from '../../types'

const teal = '#1c7b8a', orange = '#b9601b', purple = '#77326b'

export function ContourPlot({ arrows = false, points = [] }: { arrows?: boolean; points?: Vec2[] }) {
  const id = useId()
  const x = (v: number) => 46 + (v + 5) * 34
  const y = (v: number) => 454 - (v + 7) * 34
  return <svg viewBox="0 0 500 500" style={{ width: '100%' }} role="img"
    aria-label={arrows ? 'Loss contours with four equal-length candidate directions from S' : 'Loss contours with the gradient descent path from S'}>
    <defs>
      <clipPath id={`${id}-clip`}><rect x="46" y="46" width="408" height="408" /></clipPath>
      <marker id={`${id}-arrow`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill={purple} /></marker>
    </defs>
    {[-5, -3, -1, 1, 3, 5, 7].map(v => <g key={`a${v}`}>
      <line x1={x(v)} x2={x(v)} y1="46" y2="454" stroke="#e5e0d6" />
      <text x={x(v)} y="477" textAnchor="middle" fontSize="14">{v}</text>
    </g>)}
    {[-7, -5, -3, -1, 1, 3, 5].map(v => <g key={`b${v}`}>
      <line x1="46" x2="454" y1={y(v)} y2={y(v)} stroke="#e5e0d6" />
      <text x="36" y={y(v) + 5} textAnchor="end" fontSize="14">{v}</text>
    </g>)}
    <text x="250" y="497" textAnchor="middle" fontSize="16">Parameter a</text>
    <text x="10" y="24" fontSize="16">Parameter b</text>
    <g clipPath={`url(#${id}-clip)`}>
      {[2, 5, 10, 17, 26, 37].map(level => <g key={level}>
        <ellipse cx={x(1)} cy={y(-1)} rx={34 * Math.sqrt(level - 1)} ry={17 * Math.sqrt(level - 1)} fill="none" stroke={teal} opacity="0.65" strokeWidth="1.6" />
        <text x={x(1 + Math.sqrt(level - 1) * 0.7)} y={y(-1 - Math.sqrt(level - 1) * Math.sqrt(0.51) / 2)} fill="#16505b" fontSize="13" paintOrder="stroke" stroke="#fffaf2" strokeWidth="4">{level}</text>
      </g>)}
      <path d={`M${x(1)-6},${y(-1)-6}l12,12m0,-12l-12,12`} stroke="#273942" strokeWidth="2.5" />
      {points.length > 0 && <>
        <polyline points={points.map(p => `${x(p[0])},${y(p[1])}`).join(' ')} fill="none" stroke={orange} strokeWidth="2.5" />
        {points.map((p, i) => <circle key={i} cx={x(p[0])} cy={y(p[1])} r={i === points.length - 1 ? 5 : 3} fill={i === points.length - 1 ? purple : orange} stroke="white" strokeWidth="1" />)}
      </>}
      {arrows && descentArrows.map(({ label, vector }) => {
        const norm = Math.hypot(...vector)
        const end: Vec2 = [descentStart[0] + 1.8 * vector[0] / norm, descentStart[1] + 1.8 * vector[1] / norm]
        return <g key={label}>
          <line x1={x(descentStart[0])} y1={y(descentStart[1])} x2={x(end[0])} y2={y(end[1])} stroke={purple} strokeWidth="2.5" markerEnd={`url(#${id}-arrow)`} />
          <text x={x(end[0]) + 13 * vector[0] / norm} y={y(end[1]) - 13 * vector[1] / norm + 5} fontWeight="700" textAnchor="middle" fontSize="17" paintOrder="stroke" stroke="#fffaf2" strokeWidth="4">{label}</text>
        </g>
      })}
      <circle cx={x(descentStart[0])} cy={y(descentStart[1])} r="5" fill="#273942" />
      <text x={x(descentStart[0]) - 16} y={y(descentStart[1]) + 19} fontSize="17" fontWeight="700">S</text>
    </g>
  </svg>
}

function Landscape({ nonconvex = false }: { nonconvex?: boolean }) {
  const id = useId()
  const min = nonconvex ? -2 : -2.8, max = nonconvex ? 2.9 : 4.5
  const limit = nonconvex ? 6 : 16
  const loss = nonconvex ? nonconvexLoss : singleParameterLoss
  const x = (w: number) => 48 + (w - min) / (max - min) * 444
  const y = (v: number) => 300 - v / limit * 260
  const marks: Array<[string, number]> = nonconvex ? [['S', -1.7], ['A', -1], ['B', 0], ['C', 2]] : [['A', -2], ['B', 1], ['C', 3]]
  return <svg viewBox="0 0 530 350" style={{ width: '100%' }} role="img" aria-label={nonconvex ? 'Nonconvex loss with two valleys A and C, hilltop B, and starting point S' : 'Convex loss curve and tangent slopes at A, B, and C'}>
    <defs><clipPath id={id}><rect x="48" y="40" width="444" height="260" /></clipPath></defs>
    {(nonconvex ? [0,1,2,3,4,5,6] : [0,4,8,12,16]).map(v => <g key={v}><line x1="48" x2="492" y1={y(v)} y2={y(v)} stroke="#ded9cf" /><text x="37" y={y(v)+5} textAnchor="end" fontSize="15">{v}</text></g>)}
    {(nonconvex ? [-2,-1,0,1,2] : [-2,-1,0,1,2,3,4]).map(v => <text key={v} x={x(v)} y="323" textAnchor="middle" fontSize="15">{v}</text>)}
    <text x="10" y="23" fontSize="17">Loss L(w)</text><text x="270" y="347" textAnchor="middle" fontSize="17">Parameter w</text>
    <g clipPath={`url(#${id})`}>
      <polyline fill="none" stroke={teal} strokeWidth="3" points={Array.from({length:401},(_,i) => {const w=min+(max-min)*i/400;return `${x(w)},${y(loss(w))}`}).join(' ')} />
      {marks.map(([label,w]) => <g key={label}>
        {!nonconvex && <line x1={x(w-.45)} x2={x(w+.45)} y1={y(loss(w)-.45*2*(w-1))} y2={y(loss(w)+.45*2*(w-1))} stroke={orange} strokeWidth="3" />}
        <circle cx={x(w)} cy={y(loss(w))} r="5" fill={purple} />
        <text x={x(w)+(!nonconvex && label==='C' ? -22 : 8)} y={y(loss(w))-12} fontSize="18" fontWeight="700">{label}</text>
      </g>)}
    </g>
  </svg>
}

function RunComparison() {
  const colors = [orange, teal, purple]
  const x = (step: number) => 38 + step / 20 * 230
  const y = (loss: number) => 220 - loss / 90 * 188
  return <>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '1rem' }}>
      {descentRuns.map((run,i) => <div key={run.name}>
        <h4>Run {run.name}</h4>
        <svg viewBox="0 0 300 265" style={{ width: '100%' }} role="img" aria-label={`Run ${run.name}: loss across 20 updates; numerical checkpoints below`}>
          {[0,30,60,90].map(v => <g key={v}><line x1="38" x2="268" y1={y(v)} y2={y(v)} stroke="#ded9cf" /><text x="30" y={y(v)+4} textAnchor="end" fontSize="13">{v}</text></g>)}
          {[0,5,10,15,20].map(v => <text key={v} x={x(v)} y="240" textAnchor="middle" fontSize="13">{v}</text>)}
          <text x="9" y="18" fontSize="14">Loss</text><text x="150" y="261" textAnchor="middle" fontSize="14">Update step</text>
          <line x1="38" x2="268" y1={y(1)} y2={y(1)} stroke="#777" strokeDasharray="4 3" />
          <polyline points={run.trajectory.map((p,j) => `${x(j)},${y(p.loss)}`).join(' ')} stroke={colors[i]} strokeWidth="3" fill="none" />
        </svg>
      </div>)}
    </div>
    <table className="table-grid"><caption>Loss checkpoints (minimum possible: 1)</caption><thead><tr><th>Step</th>{descentRuns.map(r => <th key={r.name}>Run {r.name}</th>)}</tr></thead><tbody>
      {[0,1,5,10,20].map(step => <tr key={step}><th>{step}</th>{descentRuns.map(run => <td key={run.name}>{formatLoss(run.trajectory[step].loss)}</td>)}</tr>)}
    </tbody></table>
  </>
}

export function GradientVisual({ dataset }: { dataset: GradientVisualDataset }) {
  return <section className="data-table">
    <h3 className="panel-title">{({slopes:'A one-parameter loss',contours:'A two-parameter loss',runs:'Three optimization runs',minima:'Two valleys, different losses'})[dataset.mode]}</h3>
    <p className="table-caption">{dataset.caption}</p>
    {dataset.mode === 'contours' ? <div style={{ maxWidth: 570, margin: 'auto' }}><ContourPlot arrows /></div>
      : dataset.mode === 'runs' ? <RunComparison /> : <div style={{ maxWidth: 750, margin: 'auto' }}><Landscape nonconvex={dataset.mode === 'minima'} /></div>}
  </section>
}
