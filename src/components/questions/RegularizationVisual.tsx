import { useId, useState } from 'react'
import type { RegularizationVisualDataset } from '../../data/regularizationAssignment'
import { penaltyCandidates, penaltyExamplePoints, penaltyPath, ridgePrediction, ridgeTrainingPoints, ridgeValidationPoints } from '../../lib/regularizationConcepts'

const teal = '#1c7b8a', orange = '#b9601b', purple = '#77326b'
const colors = [teal, orange, purple]

export function RidgeModelPlot({ weights }: { weights: number[] }) {
  const id = useId()
  const x = (v: number) => 52 + (v + 2.2) / 4.4 * 440
  const y = (v: number) => 330 - v / 5 * 280
  return <svg viewBox="0 0 540 405" style={{ width: '100%' }} role="img" aria-label="Degree-8 fitted curve, training circles, and validation crosses on fixed axes">
    <defs><clipPath id={id}><rect x="52" y="50" width="440" height="280" /></clipPath></defs>
    {[0,1,2,3,4,5].map(v => <g key={v}><line x1="52" x2="492" y1={y(v)} y2={y(v)} stroke="#ded9cf" /><text x="42" y={y(v)+5} textAnchor="end" fontSize="15">{v}</text></g>)}
    {[-2,-1,0,1,2].map(v => <text key={v} x={x(v)} y="350" textAnchor="middle" fontSize="15">{v}</text>)}
    <text x="12" y="25" fontSize="16">Target y</text><text x="270" y="375" textAnchor="middle" fontSize="16">Input x</text>
    <g clipPath={`url(#${id})`}>
      <polyline points={Array.from({length:401},(_,i) => { const a=-2+4*i/400;return `${x(a)},${y(ridgePrediction(weights,a))}` }).join(' ')} fill="none" stroke={purple} strokeWidth="3" />
      {ridgeTrainingPoints.map(([a,b],i) => <circle key={`t${i}`} cx={x(a)} cy={y(b)} r="5" fill={teal} stroke="white" strokeWidth="1" />)}
      {ridgeValidationPoints.map(([a,b],i) => <path key={`v${i}`} d={`M${x(a)-5},${y(b)-5}l10,10m0,-10l-10,10`} stroke={orange} strokeWidth="2.5" />)}
    </g>
    <circle cx="65" cy="396" r="4" fill={teal} /><text x="76" y="401" fontSize="13">Training</text>
    <path d="M194,391l10,10m0,-10l-10,10" stroke={orange} strokeWidth="2" /><text x="214" y="401" fontSize="13">Validation</text>
    <line x1="351" x2="375" y1="396" y2="396" stroke={purple} strokeWidth="3" /><text x="384" y="401" fontSize="13">Model</text>
  </svg>
}

function FitComparison() {
  const [showResiduals, setShowResiduals] = useState(false)
  const x = (v: number) => 45 + (v+1.6) / 3.2 * 200
  const y = (v: number) => 224 - (v+4)/10*190
  return <>
    <label className="panel-note"><input type="checkbox" checked={showResiduals} onChange={e => setShowResiduals(e.target.checked)} /> Show vertical training residuals</label>
    <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(210px,1fr))',gap:'1rem'}}>
      {penaltyCandidates.map((model,i) => <div key={model.name}><h4>Line {model.name}: y = 1 + {model.slope}x</h4>
        <svg viewBox="0 0 270 275" style={{width:'100%'}} role="img" aria-label={`Line ${model.name}, slope ${model.slope}, and the same four training points`}>
          {[-4,-2,0,2,4,6].map(v => <g key={v}><line x1="45" x2="245" y1={y(v)} y2={y(v)} stroke="#ded9cf" /><text x="36" y={y(v)+4} textAnchor="end" fontSize="12">{v}</text></g>)}
          {[-1.5,0,1.5].map(v => <text key={v} x={x(v)} y="245" textAnchor="middle" fontSize="12">{v}</text>)}
          <text x="10" y="20" fontSize="14">Target y</text><text x="145" y="270" textAnchor="middle" fontSize="14">Input x</text>
          <line x1={x(-1.6)} x2={x(1.6)} y1={y(1-1.6*model.slope)} y2={y(1+1.6*model.slope)} stroke={colors[i]} strokeWidth="3" />
          {penaltyExamplePoints.map(([a,b],j) => <g key={j}>
            {showResiduals && <line x1={x(a)} x2={x(a)} y1={y(b)} y2={y(1+model.slope*a)} stroke="#687078" strokeDasharray="3 2" strokeWidth="2" />}
            <circle cx={x(a)} cy={y(b)} r="4" fill="#273942" />
          </g>)}
        </svg>
      </div>)}
    </div>
    <table className="table-grid"><caption>Candidate statistics; λ = 1, objective = training MSE + λw²</caption><thead><tr><th>Candidate</th><th>Slope w</th><th>Training MSE</th></tr></thead><tbody>
      {penaltyCandidates.map(model => <tr key={model.name}><th>{model.name}</th><td>{model.slope.toFixed(1)}</td><td>{model.trainingMSE.toFixed(2)}</td></tr>)}
    </tbody></table>
  </>
}

function CoefficientPaths() {
  const [strength, setStrength] = useState(0.6)
  const x = (v: number) => 46+v/2*280
  const y = (v: number) => 260-(v+1.1)/3.3*216
  return <>
    <p className="panel-note">Data loss: ½Σ(wⱼ − zⱼ)², with z = (2, −1, 0.4). The penalty conventions are λΣ|wⱼ| for L1 and (λ/2)Σwⱼ² for L2. The plots show the resulting exact minimizers, in an unknown order.</p>
    <label className="slider-shell"><span>Explore penalty strength λ: {strength.toFixed(1)}</span>
      <input className="range-input" type="range" min="0" max="2" step="0.1" value={strength} aria-label="Explore coefficient paths" onChange={e => setStrength(Number(e.target.value))} />
    </label>
    <p className="panel-note">Exploration does not change the question’s fixed reference at λ = 0.6, shown below.</p>
    <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(260px,1fr))',gap:'1rem'}}>
      {(['l1','l2'] as const).map((kind,i) => <div key={kind}><h4>Plot {i===0?'A':'B'}</h4>
        <svg viewBox="0 0 360 315" style={{width:'100%'}} role="img" aria-label={`Coefficient paths in plot ${i===0?'A':'B'}, with inspected strength ${strength.toFixed(1)}`}>
          {[-1,0,1,2].map(v => <g key={v}><line x1="46" x2="326" y1={y(v)} y2={y(v)} stroke={v===0?'#879197':'#ded9cf'} /><text x="34" y={y(v)+5} textAnchor="end" fontSize="14">{v}</text></g>)}
          {[0,0.5,1,1.5,2].map(v => <text key={v} x={x(v)} y="282" textAnchor="middle" fontSize="14">{v}</text>)}
          <text x="10" y="23" fontSize="16">Coefficient value</text><text x="185" y="307" textAnchor="middle" fontSize="16">Penalty strength λ</text>
          <line x1={x(strength)} x2={x(strength)} y1="44" y2="260" stroke="#273942" strokeDasharray="3 4" />
          {[0,1,2].map(j => <g key={j}>
            <polyline points={Array.from({length:101},(_,k)=>`${x(k/50)},${y(penaltyPath(kind,k/50)[j])}`).join(' ')} fill="none" stroke={colors[j]} strokeWidth="3" strokeDasharray={j===0?undefined:j===1?'8 4':'2 4'} />
            <circle cx={x(strength)} cy={y(penaltyPath(kind,strength)[j])} r="4" fill={colors[j]} stroke="white" />
          </g>)}
        </svg>
        <output aria-label={`Plot ${i===0?'A':'B'} inspected coefficients`}>At λ = {strength.toFixed(1)}: ({penaltyPath(kind,strength).map(w=>w.toFixed(3)).join(', ')})</output>
      </div>)}
    </div>
    <p className="panel-note">w₁: solid teal; w₂: long-dashed orange; w₃: dotted purple. Values are ordered (w₁, w₂, w₃).</p>
    <table className="table-grid"><caption>Fixed reference for the question: λ = 0.6</caption><thead><tr><th>Plot</th><th>w₁</th><th>w₂</th><th>w₃</th></tr></thead><tbody>
      {(['l1','l2'] as const).map((kind,i) => <tr key={kind}><th>{i===0?'A':'B'}</th>{penaltyPath(kind,.6).map((w,j)=><td key={j}>{w.toFixed(3)}</td>)}</tr>)}
    </tbody></table>
  </>
}

function ScalingComparison() {
  return <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(260px,1fr))',gap:'1rem'}}>
    {[1,100].map(scale => <div key={scale}><h4>{scale===1?'Meters: y = 1 + 2d':'Centimeters: y = 1 + 0.02d'}</h4>
      <svg viewBox="0 0 330 285" style={{width:'100%'}} role="img" aria-label={`Same physical predictions with distance in ${scale===1?'meters':'centimeters'}`}>
        {[1,2,3,4,5].map(v => <g key={v}><line x1="43" x2="290" y1={240-v*38} y2={240-v*38} stroke="#ded9cf" /><text x="32" y={245-v*38} fontSize="14" textAnchor="end">{v}</text></g>)}
        <line x1="43" x2="290" y1="202" y2="50" stroke={teal} strokeWidth="3" />
        {[0,.5,1,1.5,2].map(v => <g key={v}><circle cx={43+v*123.5} cy={240-(1+2*v)*38} r="4" fill={purple} /><text x={43+v*123.5} y="250" textAnchor="middle" fontSize="14">{v*scale}</text></g>)}
        <text x="10" y="23" fontSize="16">Prediction y</text><text x="167" y="280" textAnchor="middle" fontSize="15">Distance d ({scale===1?'meters':'centimeters'})</text>
      </svg>
    </div>)}
  </div>
}

function EvaluationSets() {
  return <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(190px,1fr))',gap:'1rem'}}>
    {['Training','Validation','Test'].map((label,i) => <section className="status-panel" key={label}>
      <h4>{label} set</h4>
      <svg viewBox="0 0 180 80" style={{width:'100%',maxWidth:240}} role="img" aria-label={`${label} observations, disjoint from the other sets`}>
        {Array.from({length:10},(_,j) => i===0?<circle key={j} cx={18+36*(j%5)} cy={22+36*Math.floor(j/5)} r="6" fill={colors[i]} />:i===1?<path key={j} d={`M${12+36*(j%5)},${16+36*Math.floor(j/5)}l12,12m0,-12l-12,12`} stroke={colors[i]} strokeWidth="3" />:<rect key={j} x={12+36*(j%5)} y={16+36*Math.floor(j/5)} width="12" height="12" fill={colors[i]} />)}
      </svg>
      <p className="panel-note">{i===0?'Used to fit coefficients and preprocessing.':i===1?'Available while comparing candidate models.':'Kept untouched during model development.'}</p>
    </section>)}
  </div>
}

export function RegularizationVisual({ dataset }: { dataset: RegularizationVisualDataset }) {
  return <section className="data-table">
    <h3 className="panel-title">{{fit:'Fit versus coefficient size',paths:'Two coefficient paths',scaling:'Same predictions, different coefficient units',evaluation:'Separate fitting, selection, and evaluation'}[dataset.mode]}</h3>
    <p className="table-caption">{dataset.caption}</p>
    {dataset.mode==='fit'?<FitComparison />:dataset.mode==='paths'?<CoefficientPaths />:dataset.mode==='scaling'?<ScalingComparison />:<EvaluationSets />}
  </section>
}
