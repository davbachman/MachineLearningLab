import { useState } from 'react'
import { QuestionFrame } from '../QuestionFrame'
import type { AttemptOutcome, QuestionState, TableEntryQuestionSpec } from '../../types'
import { handCalculationDatasets } from '../../data/pcaDatasets'
import { DEFAULT_HINT_SCHEDULE } from '../../lib/assignmentState'
import { formatNumber } from '../../lib/pcaMath'

interface TableEntryQuestionProps {
  question: TableEntryQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
  onGiveUp: (answer: unknown) => void
}

function blankMatrix(rows: number, columns: number) {
  return Array.from({ length: rows }, () => Array.from({ length: columns }, () => ''))
}

function parseMatrix(inputs: string[][]) {
  const parsed = inputs.map((row) =>
    row.map((value) => (value.trim() === '' ? Number.NaN : Number(value))),
  )
  if (parsed.some((row) => row.some((value) => Number.isNaN(value)))) {
    return null
  }
  return parsed
}

function parseVector(inputs: string[]) {
  const parsed = inputs.map((value) => (value.trim() === '' ? Number.NaN : Number(value)))
  if (parsed.some((value) => Number.isNaN(value))) {
    return null
  }
  return parsed
}

export function TableEntryQuestion({
  question,
  state,
  questionNumber,
  totalQuestions,
  hints,
  onAttempt,
  onGiveUp,
}: TableEntryQuestionProps) {
  const dataset = handCalculationDatasets[question.datasetId]
  const [centeredInputs, setCenteredInputs] = useState(() => blankMatrix(dataset.rows.length, 2))
  const [covarianceInputs, setCovarianceInputs] = useState(() => blankMatrix(2, 2))
  const [directionInputs, setDirectionInputs] = useState(['', ''])
  const [feedback, setFeedback] = useState('')
  const [stepIndex, setStepIndex] = useState(0)
  const resolved = state.status === 'correct' || state.status === 'gave_up'
  const schedule = question.hintSchedule ?? DEFAULT_HINT_SCHEDULE
  const shownCenteredInputs =
    state.status === 'gave_up'
      ? dataset.centeredRows.map((row) => row.map(formatNumber))
      : centeredInputs
  const shownCovarianceInputs =
    state.status === 'gave_up'
      ? dataset.covariance.map((row) => row.map(formatNumber))
      : covarianceInputs
  const shownDirectionInputs =
    state.status === 'gave_up' ? dataset.firstDirection.map(formatNumber) : directionInputs
  const shownStepIndex = state.status === 'gave_up' ? 2 : stepIndex

  const handleCenteredChange = (rowIndex: number, columnIndex: number, value: string) => {
    setCenteredInputs((current) =>
      current.map((row, currentRowIndex) =>
        currentRowIndex === rowIndex
          ? row.map((cell, currentColumnIndex) => (currentColumnIndex === columnIndex ? value : cell))
          : row,
      ),
    )
  }

  const handleCovarianceChange = (rowIndex: number, columnIndex: number, value: string) => {
    setCovarianceInputs((current) =>
      current.map((row, currentRowIndex) =>
        currentRowIndex === rowIndex
          ? row.map((cell, currentColumnIndex) => (currentColumnIndex === columnIndex ? value : cell))
          : row,
      ),
    )
  }

  const handleDirectionChange = (index: number, value: string) => {
    setDirectionInputs((current) =>
      current.map((cell, currentIndex) => (currentIndex === index ? value : cell)),
    )
  }

  const invalidAttempt = (message: string, answer: unknown) => {
    const nextIncorrectAttempts = state.incorrectAttempts + 1
    setFeedback(schedule.includes(nextIncorrectAttempts) ? 'Not yet. A new hint appeared below.' : message)
    onAttempt('incorrect', answer)
  }

  const checkCurrentStep = () => {
    if (stepIndex === 0) {
      const parsed = parseMatrix(shownCenteredInputs)
      if (!parsed) {
        invalidAttempt('Enter a number in every centered-data box before checking.', {
          step: 'centeredData',
          values: shownCenteredInputs,
        })
        return
      }

      const result = question.validator({ step: 'centeredData', values: parsed })
      if (!result.correct) {
        invalidAttempt(result.message ?? 'Check the column means again.', {
          step: 'centeredData',
          values: parsed,
        })
        return
      }

      onAttempt('progress', { step: 'centeredData', values: parsed })
      setFeedback('Centered data confirmed. Now compute the covariance matrix.')
      setStepIndex(1)
      return
    }

    if (stepIndex === 1) {
      const parsed = parseMatrix(shownCovarianceInputs)
      if (!parsed) {
        invalidAttempt('Enter a number in every covariance box before checking.', {
          step: 'covariance',
          values: shownCovarianceInputs,
        })
        return
      }

      const result = question.validator({ step: 'covariance', values: parsed })
      if (!result.correct) {
        invalidAttempt(result.message ?? 'The covariance entries do not match yet.', {
          step: 'covariance',
          values: parsed,
        })
        return
      }

      onAttempt('progress', { step: 'covariance', values: parsed })
      setFeedback('Covariance confirmed. Finish with a first principal direction.')
      setStepIndex(2)
      return
    }

    const parsed = parseVector(shownDirectionInputs)
    if (!parsed) {
      invalidAttempt('Enter both coordinates for the principal direction before checking.', {
        step: 'direction',
        values: shownDirectionInputs,
      })
      return
    }

    const result = question.validator({ step: 'direction', values: parsed })
    if (!result.correct) {
      invalidAttempt(result.message ?? 'That vector is not aligned with the first principal direction yet.', {
        step: 'direction',
        values: parsed,
      })
      return
    }

    setFeedback(question.successCopy ?? 'Correct! Good Job!')
    onAttempt('correct', { step: 'direction', values: parsed })
  }

  const revealAnswer = () => {
    setFeedback('Answer revealed. The diagonal line direction is the only direction with large variance.')
    onGiveUp(
      stepIndex === 0
        ? { step: 'centeredData', values: centeredInputs }
        : stepIndex === 1
          ? { step: 'covariance', values: covarianceInputs }
          : { step: 'direction', values: directionInputs },
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
          <button type="button" className="button" onClick={checkCurrentStep} disabled={resolved}>
            {stepIndex < 2 ? 'Check this step' : 'Check direction'}
          </button>
          <button type="button" className="button-secondary" onClick={revealAnswer} disabled={resolved}>
            Give up
          </button>
        </>
      }
    >
      <div className="table-shell">
        <section className="data-table">
          <h3 className="panel-title">Original data</h3>
          <table className="table-grid">
            <thead>
              <tr>
                {dataset.headers.map((header) => (
                  <th key={`${dataset.id}-${header}`}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dataset.rows.map((row, rowIndex) => (
                <tr key={`${dataset.id}-original-${rowIndex}`}>
                  {row.map((value, columnIndex) => (
                    <td key={`${dataset.id}-original-${rowIndex}-${columnIndex}`}>{value}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="answer-panel">
          <div className="substep-shell">
            <div className="substep-header">
              <h3 className="substep-title">1. Center the data</h3>
              <span className="substep-status">{shownStepIndex > 0 || resolved ? 'Locked in' : 'Current step'}</span>
            </div>
            <table className="input-grid">
              <tbody>
                {shownCenteredInputs.map((row, rowIndex) => (
                  <tr key={`${question.id}-center-row-${rowIndex}`}>
                    {row.map((value, columnIndex) => (
                      <td key={`${question.id}-center-cell-${rowIndex}-${columnIndex}`}>
                        <label className="sr-only" htmlFor={`${question.id}-center-${rowIndex}-${columnIndex}`}>
                          Centered entry {rowIndex + 1}, {columnIndex + 1}
                        </label>
                        <input
                          id={`${question.id}-center-${rowIndex}-${columnIndex}`}
                          className="numeric-input"
                          value={value}
                          onChange={(event) =>
                            handleCenteredChange(rowIndex, columnIndex, event.target.value)
                          }
                          disabled={shownStepIndex > 0 || resolved}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="substep-shell">
            <div className="substep-header">
              <h3 className="substep-title">2. Enter the covariance matrix</h3>
              <span className="substep-status">
                {shownStepIndex === 1 && !resolved
                  ? 'Current step'
                  : shownStepIndex > 1 || resolved
                    ? 'Locked in'
                    : 'Locked'}
              </span>
            </div>
            <table className="matrix-grid">
              <tbody>
                {shownCovarianceInputs.map((row, rowIndex) => (
                  <tr key={`${question.id}-cov-row-${rowIndex}`}>
                    {row.map((value, columnIndex) => (
                      <td key={`${question.id}-cov-cell-${rowIndex}-${columnIndex}`}>
                        <label className="sr-only" htmlFor={`${question.id}-cov-${rowIndex}-${columnIndex}`}>
                          Covariance entry {rowIndex + 1}, {columnIndex + 1}
                        </label>
                        <input
                          id={`${question.id}-cov-${rowIndex}-${columnIndex}`}
                          className="numeric-input"
                          value={value}
                          onChange={(event) =>
                            handleCovarianceChange(rowIndex, columnIndex, event.target.value)
                          }
                          disabled={shownStepIndex < 1 || shownStepIndex > 1 || resolved}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="substep-shell">
            <div className="substep-header">
              <h3 className="substep-title">3. Enter the first principal direction</h3>
              <span className="substep-status">
                {shownStepIndex === 2 && !resolved ? 'Current step' : resolved ? 'Revealed or correct' : 'Locked'}
              </span>
            </div>
            <div className="vector-input-row">
              {shownDirectionInputs.map((value, index) => (
                <div key={`${question.id}-dir-${index}`} style={{ flex: '1 1 120px' }}>
                  <label className="sr-only" htmlFor={`${question.id}-dir-input-${index}`}>
                    Principal direction coordinate {index + 1}
                  </label>
                  <input
                    id={`${question.id}-dir-input-${index}`}
                    className="numeric-input"
                    value={value}
                    onChange={(event) => handleDirectionChange(index, event.target.value)}
                    disabled={shownStepIndex < 2 || resolved}
                  />
                </div>
              ))}
            </div>
          </div>
        </section>
      </div>
    </QuestionFrame>
  )
}
