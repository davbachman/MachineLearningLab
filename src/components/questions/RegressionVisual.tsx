import { useId } from 'react'
import type { RegressionVisualDataset } from '../../data/multipleChoiceDatasets'
import { polynomialErrorRows, polynomialExampleModels, polynomialExamplePoints } from '../../data/polynomialVisualData'

export function RegressionVisual({ dataset }: { dataset: RegressionVisualDataset }) {
  const clipId = useId()
  if (dataset.mode === 'curves') {
    const x = (v: number) => 40 + (v + 2.2) / 4.4 * 360
    const y = (v: number) => 235 - v / 5 * 210
    return <section className="data-table">
      <h3 className="panel-title">Same data, three fitted models</h3>
      <p className="table-caption">{dataset.caption}</p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '1rem' }}>
      {polynomialExampleModels.map(model => <div key={model.name}>
        <h4>Model {model.name}</h4>
        <svg viewBox="0 0 440 270" style={{ width: '100%' }} role="img" aria-label={`Model ${model.name}: fitted curve and training observations`}>
          <defs><clipPath id={`${clipId}-${model.name}`}><rect x="40" y="25" width="360" height="210" /></clipPath></defs>
          {[0, 1, 2, 3, 4, 5].map(v => <g key={v}><line x1="40" x2="400" y1={y(v)} y2={y(v)} stroke="#ded9cf" /><text x="30" y={y(v) + 4} textAnchor="end" fontSize="17">{v}</text></g>)}
          {[-2, -1, 0, 1, 2].map(v => <text key={v} x={x(v)} y="253" textAnchor="middle" fontSize="17">{v}</text>)}
          <text x="425" y="254">x</text><text x="12" y="20">y</text>
          <g clipPath={`url(#${clipId}-${model.name})`}>
            <polyline fill="none" stroke="#cc7c31" strokeWidth="3" points={Array.from({ length: 161 }, (_,i) => {
              const v = -2 + i / 40
              const prediction = model.coefficients.reduce((sum,c,power) => sum + c * v ** power, 0)
              return `${x(v)},${y(prediction)}`
            }).join(' ')} />
            {polynomialExamplePoints.map(([a,b],i) => <circle key={i} cx={x(a)} cy={y(b)} r="4.5" fill="#1c7b8a" stroke="#18323a" />)}
          </g>
        </svg>
      </div>)}
      </div>
      <p className="panel-note">Teal dots: training observations. Orange: fitted model.</p>
    </section>
  }
  const x = (degree: number) => 48 + (degree - 1) / 7 * 350
  const y = (mse: number) => 255 - mse / 300 * 220
  return <section className="data-table">
    <h3 className="panel-title">Error versus polynomial degree</h3>
    <p className="table-caption">{dataset.caption}</p>
    <svg viewBox="0 0 440 302" style={{ width: '100%' }} role="img" aria-label="Training and validation MSE versus polynomial degree; exact values in the table below">
      {[0, 50, 100, 150, 200, 250, 300].map(v => <g key={v}><line x1="48" x2="398" y1={y(v)} y2={y(v)} stroke="#ded9cf" /><text x="40" y={y(v)+4} textAnchor="end" fontSize="12">{v}</text></g>)}
      <text x="8" y="18" fontSize="13">MSE</text><text x="220" y="296" fontSize="13" textAnchor="middle">Polynomial degree</text>
      {[1,2].map((column) => <g key={column}>
        <polyline fill="none" stroke={column === 1 ? '#1c7b8a' : '#cc7c31'} strokeWidth="2.5" points={polynomialErrorRows.map(row => `${x(row[0])},${y(row[column])}`).join(' ')} />
        {polynomialErrorRows.map(row => column === 1
          ? <circle key={row[0]} cx={x(row[0])} cy={y(row[column])} r="4" fill="#1c7b8a" />
          : <rect key={row[0]} x={x(row[0])-4} y={y(row[column])-4} width="8" height="8" fill="#cc7c31" />)}
      </g>)}
      {polynomialErrorRows.map(row => <text key={row[0]} x={x(row[0])} y="276" textAnchor="middle" fontSize="12">{row[0]}</text>)}
    </svg>
    <table className="table-grid"><thead><tr><th>Degree</th><th>Training MSE</th><th>Validation MSE</th></tr></thead>
      <tbody>{polynomialErrorRows.map(row => <tr key={row[0]}>{row.map((v,i) => <td key={i}>{i ? v.toFixed(2) : v}</td>)}</tr>)}</tbody>
    </table>
  </section>
}
