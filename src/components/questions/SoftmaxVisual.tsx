import { useId, useState } from 'react'
import type { SoftmaxVisualDataset } from '../../data/softmaxAssignment'
import { baseSoftmaxScores, boundaryPoints, boundaryScores, categoricalLoss, comparisonPredictions, multiclassPredictions, scoreGradient, softmaxClasses, softmaxProbabilities, transformedScores, updateProbabilities } from '../../lib/softmaxConcepts'

const colors = ['#1c7b8a','#bd681f','#77326b']

function ProbabilityBars({ probabilities, title }: { probabilities:number[]; title:string }) {
  return <div>
    <h4>{title}</h4>
    <svg viewBox="0 0 360 225" style={{width:'100%'}} role="img" aria-label={`${title}; probabilities for classes A, B, C`}>
      {[0,.25,.5,.75,1].map(p=><g key={p}><line x1="35" x2="330" y1={185-p*145} y2={185-p*145} stroke="#ded9cf" /><text x="28" y={190-p*145} textAnchor="end" fontSize="12">{p}</text></g>)}
      {probabilities.map((p,i)=><g key={i}>
        <rect x={60+i*93} y={185-p*145} width="46" height={p*145} fill={colors[i]} rx="3" />
        <text x={83+i*93} y={175-p*145} textAnchor="middle" fontWeight="700" fontSize="15">{p.toFixed(3)}</text>
        <text x={83+i*93} y="210" textAnchor="middle" fontWeight="700" fontSize="17">{softmaxClasses[i]}</text>
      </g>)}
      <text x="8" y="19" fontSize="14">Probability</text>
    </svg>
  </div>
}

function ScoreExplorer() {
  const [shift,setShift]=useState(0), [spread,setSpread]=useState(1)
  const scores=transformedScores(shift,spread), probabilities=softmaxProbabilities(scores)
  return <div className="visual-grid">
    <ProbabilityBars probabilities={probabilities} title="Current probabilities" />
    <div className="projection-panel">
      <label className="slider-shell"><span>Common shift: {shift.toFixed(1)}</span><input className="range-input" aria-label="Common score shift" type="range" min="-3" max="3" step=".1" value={shift} onChange={e=>setShift(Number(e.target.value))} /></label>
      <label className="slider-shell"><span>Spread multiplier: {spread.toFixed(1)}</span><input className="range-input" aria-label="Score spread multiplier" type="range" min=".5" max="2" step=".1" value={spread} onChange={e=>setSpread(Number(e.target.value))} /></label>
      <p><output aria-label="Current class scores">Scores: ({scores.map(s=>s.toFixed(2)).join(', ')})</output></p>
      <p><output aria-label="Current class probabilities">Probabilities: ({probabilities.map(p=>p.toFixed(4)).join(', ')})</output></p>
      <p className="panel-note">New score = spread × original score + common shift. Softmax exponentiates and normalizes these scores across the three classes.</p>
      <button type="button" className="button secondary" onClick={()=>{setShift(0);setSpread(1)}}>Reset scores</button>
    </div>
  </div>
}

function BoundaryExplorer() {
  const [point,setPoint]=useState<[number,number]>([.5,-.5])
  const id=useId()
  const toX=(v:number)=>42+(v+2)*90, toY=(v:number)=>402-(v+2)*90
  const scores=boundaryScores(...point), probs=softmaxProbabilities(scores)
  const leaders=softmaxClasses.filter((_,i)=>Math.abs(scores[i]-Math.max(...scores))<1e-10)
  return <div className="visual-grid">
    <div>
      <svg viewBox="0 0 445 445" style={{width:'100%'}} role="img" aria-label="Feature plane with A–B score-tie line and fixed points P and Q; use the probe controls">
        <defs><clipPath id={id}><rect x="42" y="42" width="360" height="360" /></clipPath></defs>
        {[-2,-1,0,1,2].map(v=><g key={v}>
          <line x1={toX(v)} x2={toX(v)} y1="42" y2="402" stroke={v===0?'#9a948a':'#e4dfd5'} /><line x1="42" x2="402" y1={toY(v)} y2={toY(v)} stroke={v===0?'#9a948a':'#e4dfd5'} />
          <text x={toX(v)} y="423" textAnchor="middle" fontSize="14">{v}</text><text x="30" y={toY(v)+5} textAnchor="end" fontSize="14">{v}</text>
        </g>)}
        <line x1={toX(-2)} y1={toY(-2)} x2={toX(2)} y2={toY(2)} stroke="#777" strokeWidth="2" strokeDasharray="7 5" />
        <text x="220" y="443" textAnchor="middle" fontSize="16">Feature x₁</text><text x="10" y="23" fontSize="16">Feature x₂</text>
        {boundaryPoints.map(p=><g key={p.name}><circle cx={toX(p.x)} cy={toY(p.y)} r="6" fill="#263941" /><text x={toX(p.x)+10} y={toY(p.y)-12} fontWeight="700" fontSize="19">{p.name}</text></g>)}
        <g clipPath={`url(#${id})`}><circle cx={toX(point[0])} cy={toY(point[1])} r="10" fill="none" stroke="#bd681f" strokeWidth="3" /></g>
      </svg>
      <p className="panel-note">P and Q stay fixed for the questions. The open orange circle is your movable probe.</p>
    </div>
    <div className="projection-panel">
      {([0,1] as const).map(i=><label className="slider-shell" key={i}><span>Probe x{i===0?'₁':'₂'}: {point[i].toFixed(1)}</span><input className="range-input" type="range" aria-label={`Probe feature ${i+1}`} min="-2" max="2" step=".1" value={point[i]} onChange={e=>setPoint(previous=>i===0?[Number(e.target.value),previous[1]]:[previous[0],Number(e.target.value)])} /></label>)}
      <div className="question-controls">{boundaryPoints.map(p=><button type="button" className="button secondary" key={p.name} onClick={()=>setPoint([p.x,p.y])}>Inspect {p.name}</button>)}</div>
      <p><output aria-label="Probe scores">Scores A, B, C: ({scores.map(s=>s.toFixed(1)).join(', ')})</output></p>
      <p><output aria-label="Leading probe classes">Highest score: {leaders.join(' and ')}{leaders.length>1?' (tie)':''}</output></p>
      <ProbabilityBars probabilities={probs} title="Probe probabilities" />
    </div>
  </div>
}

function LearningSignal() {
  const [rate,setRate]=useState(0)
  // log(p) is one score vector whose softmax equals the supplied probabilities.
  const gradient=scoreGradient(updateProbabilities,1)
  const newScores=updateProbabilities.map((p,i)=>Math.log(p)-rate*gradient[i])
  const probabilities=softmaxProbabilities(newScores)
  return <div className="visual-grid">
    <ProbabilityBars probabilities={probabilities} title={rate===0?'Starting probabilities':'After one score update'} />
    <div className="projection-panel">
      <label className="slider-shell"><span>Score-step learning rate: {rate.toFixed(2)}</span><input className="range-input" aria-label="Score update learning rate" type="range" min="0" max="1" step=".05" value={rate} onChange={e=>setRate(Number(e.target.value))} /></label>
      <p><output aria-label="Score changes">Score changes: ({gradient.map(g=>(-rate*g).toFixed(3)).join(', ')})</output></p>
      <p><output aria-label="Score update cross entropy">Cross-entropy: {categoricalLoss(probabilities,1).toFixed(4)}</output></p>
      <p className="panel-note">Every slider change restarts from (0.50, 0.30, 0.20) and takes exactly one step. Rate 0 shows the starting state only. The questions ask about the direction of a positive-rate step, not a particular rate.</p>
    </div>
  </div>
}

export function SoftmaxVisual({dataset}:{dataset:SoftmaxVisualDataset}) {
  return <section className="data-table">
    <h3 className="panel-title">{({probabilities:'One distribution per observation',explorer:'Score transformation experiment',boundary:'Class competition in feature space',loss:'Same observed class, two predictions',updates:'One score-gradient update'})[dataset.mode]}</h3>
    <p className="table-caption">{dataset.caption}</p>
    {dataset.mode==='explorer'?<ScoreExplorer />:dataset.mode==='boundary'?<BoundaryExplorer />:dataset.mode==='updates'?<LearningSignal />:
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit, minmax(250px, 1fr))',gap:'1rem'}}>
        {(dataset.mode==='probabilities'?multiclassPredictions:comparisonPredictions).map(item=><ProbabilityBars key={item.name} probabilities={item.probabilities} title={`${dataset.mode==='probabilities'?'Observation':'Model'} ${item.name}`} />)}
      </div>}
    {dataset.mode==='loss' && <p className="panel-note">One-hot targets select the observed class: loss = −ln(probability assigned to B). The other bars still count toward the probability total of 1.</p>}
    {dataset.mode==='explorer' && <p className="panel-note">Original scores: ({baseSoftmaxScores.join(', ')}). The questions below compare shifts at spread 1 and spreads 1 versus 2 at shift 0.</p>}
  </section>
}
