import { useState } from 'react'
import type { LogisticVisualDataset } from '../../data/logisticAssignment'
import { binaryCrossEntropy, logisticBoundaryPoints, logisticLossLabels, logisticLossModels, logisticScoreExamples, sigmoid } from '../../lib/logisticConcepts'

const teal = '#1c7b8a', orange = '#b9601b', purple = '#77326b'

function SigmoidPlot() {
  const [score, setScore] = useState(0)
  const x = (score: number) => 64 + (score + 4) * 72
  const y = (probability: number) => 300 - probability * 256
  return <><svg viewBox="0 0 690 360" style={{ width: '100%' }} role="img" aria-label="Sigmoid curve mapping linear scores from negative 4 to 4 to probabilities from 0 to 1">
    {[0, 0.25, 0.5, 0.75, 1].map(p => <g key={p}>
      <line x1="64" x2="640" y1={y(p)} y2={y(p)} stroke="#ded9cf" />
      <text x="52" y={y(p) + 5} textAnchor="end" fontSize="15">{p}</text>
    </g>)}
    {[-4, -2, 0, 2, 4].map(z => <g key={z}>
      <line x1={x(z)} x2={x(z)} y1="44" y2="300" stroke="#eee8df" />
      <text x={x(z)} y="325" textAnchor="middle" fontSize="16">{z}</text>
    </g>)}
    <line x1="64" x2="640" y1={y(0.5)} y2={y(0.5)} stroke={orange} strokeDasharray="6 5" strokeWidth="2" />
    <polyline fill="none" stroke={teal} strokeWidth="3" points={Array.from({ length: 201 }, (_, i) => {
      const z = -4 + 8 * i / 200
      return `${x(z)},${y(sigmoid(z))}`
    }).join(' ')} />
    {logisticScoreExamples.map(({ id, score }) => <g key={id}>
      <line x1={x(score)} x2={x(score)} y1="300" y2={y(sigmoid(score))} stroke={purple} strokeDasharray="4 4" />
      <circle cx={x(score)} cy={y(sigmoid(score))} r="6" fill={purple} />
      <text x={x(score) - 14} y={y(sigmoid(score)) - 13} textAnchor="end" fontWeight="700" fontSize="18">{id}</text>
    </g>)}
    <circle cx={x(score)} cy={y(sigmoid(score))} r="10" fill="none" stroke={orange} strokeWidth="3" />
    <text x="18" y="24" fontSize="17">Predicted probability p of class 1</text>
    <text x="350" y="353" textAnchor="middle" fontSize="17">Linear score z</text>
  </svg>
    <label className="slider-shell"><span>Explore the linear score: {score.toFixed(1)}</span>
      <input className="range-input" type="range" min="-4" max="4" step="0.1" value={score} aria-label="Explore linear score"
        onChange={event => setScore(Number(event.target.value))} />
    </label>
    <p aria-live="polite"><output aria-label="Explored probability">At z = {score.toFixed(1)}, p = {sigmoid(score).toFixed(3)}</output>. Orange ring: explored value. The three reference points A, B, C stay fixed.</p>
  </>
}

function BoundaryPlot() {
  const [threshold, setThreshold] = useState(0.5)
  // Equal length for one feature unit on both axes.
  const x = (value: number) => 74 + value * 120
  const y = (value: number) => 406 - value * 120
  const scoreCutoff = Math.log(threshold / (1 - threshold))
  const boundaryHeight = 2 + scoreCutoff / 2
  const lastX = Math.min(4, 2 * boundaryHeight)
  return <><svg viewBox="0 0 630 468" style={{ width: '100%' }} role="img" aria-label="Feature-space reference decision boundary x1 plus 2 x2 equals 4, four observation locations, and an adjustable threshold boundary">
    {[0, 1, 2, 3, 4].map(value => <g key={`x${value}`}>
      <line x1={x(value)} x2={x(value)} y1="46" y2="406" stroke="#ded9cf" />
      <text x={x(value)} y="431" textAnchor="middle" fontSize="16">{value}</text>
    </g>)}
    {[0, 1, 2, 3].map(value => <g key={`y${value}`}>
      <line x1="74" x2="554" y1={y(value)} y2={y(value)} stroke="#ded9cf" />
      <text x="59" y={y(value) + 5} textAnchor="end" fontSize="16">{value}</text>
    </g>)}
    <line x1={x(0)} y1={y(2)} x2={x(4)} y2={y(0)} stroke={teal} strokeWidth="3" />
    <text x={x(0.1)} y={y(2.12)} fontSize="16" fill={teal}>p = 0.5</text>
    {threshold !== 0.5 && <line x1={x(0)} y1={y(boundaryHeight)} x2={x(lastX)} y2={y(boundaryHeight - lastX / 2)} stroke={orange} strokeWidth="3" strokeDasharray="8 5" />}
    {logisticBoundaryPoints.map(({ id, point: [x1, x2] }) => <g key={id}>
      <circle cx={x(x1)} cy={y(x2)} r="6" fill={purple} stroke="white" strokeWidth="2" />
      <text x={x(x1) + 12} y={y(x2) + (id === 'C' ? -12 : 23)} fontSize="16" fontWeight="600"
        paintOrder="stroke" stroke="#fffaf2" strokeWidth="4">{id} ({x1}, {x2})</text>
    </g>)}
    <text x="18" y="25" fontSize="17">Feature x₂</text>
    <text x="315" y="461" textAnchor="middle" fontSize="17">Feature x₁</text>
  </svg>
    <label className="slider-shell"><span>Explore the probability threshold: {threshold.toFixed(2)}</span>
      <input className="range-input" type="range" min="0.25" max="0.75" step="0.05" value={threshold} aria-label="Explore boundary threshold"
        onChange={event => setThreshold(Number(event.target.value))} />
    </label>
    <p aria-live="polite"><output aria-label="Explored boundary">Explored boundary: x₁ + 2x₂ − 4 = {scoreCutoff.toFixed(3)}</output>.</p>
    <p className="panel-note">The solid teal reference stays at threshold 0.5; the dashed orange boundary uses the slider value (they coincide at 0.5). Class 1 lies above the chosen boundary. Weights and probabilities stay fixed. Answer the questions at their stated thresholds.</p>
  </>
}

function LossComparison() {
  const y = (loss: number) => 240 - loss * 40
  return <>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
      {logisticLossModels.map((model, modelIndex) => <div key={model.id}>
        <h4 className="panel-title">Model {model.id}</h4>
        <svg viewBox="0 0 350 294" style={{ width: '100%' }} role="img" aria-label={`Model ${model.id}: cross-entropy contributions for observations 1 through 4`}>
          {[0, 1, 2, 3, 4, 5].map(loss => <g key={loss}>
            <line x1="40" x2="331" y1={y(loss)} y2={y(loss)} stroke="#ded9cf" />
            <text x="30" y={y(loss) + 5} textAnchor="end" fontSize="14">{loss}</text>
          </g>)}
          <text x="9" y="22" fontSize="15">Loss for one observation</text>
          {model.probabilities.map((p, index) => {
            const loss = binaryCrossEntropy(p, logisticLossLabels[index])
            const x = 57 + 68 * index
            return <g key={index}>
              <rect x={x} y={y(loss)} width="40" height={240 - y(loss)} fill={modelIndex === 0 ? teal : orange} />
              <text x={x + 20} y={y(loss) - 8} fontSize="14" textAnchor="middle">{loss.toFixed(3)}</text>
              <text x={x + 20} y="264" textAnchor="middle" fontSize="15">{index + 1}</text>
            </g>
          })}
          <text x="182" y="289" textAnchor="middle" fontSize="15">Observation</text>
        </svg>
      </div>)}
    </div>
    <table className="table-grid">
      <caption>Probabilities of class 1, before applying the threshold</caption>
      <thead><tr><th>Observation</th><th>True class</th>{logisticLossModels.map(model => <th key={model.id}>Model {model.id}: p</th>)}</tr></thead>
      <tbody>{logisticLossLabels.map((label, index) => <tr key={index}>
        <th>{index + 1}</th><td>{label}</td>{logisticLossModels.map(model => <td key={model.id}>{model.probabilities[index].toFixed(2)}</td>)}
      </tr>)}</tbody>
    </table>
  </>
}

export function LogisticVisual({ dataset }: { dataset: LogisticVisualDataset }) {
  const titles = { sigmoid: 'A score is not a probability', boundary: 'A boundary in feature space', loss: 'Compare probability quality, not just accuracy' }
  return <section className="data-table">
    <h3 className="panel-title">{titles[dataset.mode]}</h3>
    <p className="table-caption">{dataset.caption}</p>
    {dataset.mode === 'loss' ? <LossComparison /> : <div style={{ maxWidth: 720, margin: 'auto' }}>
      {dataset.mode === 'sigmoid' ? <SigmoidPlot /> : <BoundaryPlot />}
    </div>}
  </section>
}
