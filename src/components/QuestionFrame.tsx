import type { PropsWithChildren, ReactNode } from 'react'
import type { QuestionSpec, QuestionState } from '../types'

interface QuestionFrameProps extends PropsWithChildren {
  question: QuestionSpec
  questionNumber: number
  totalQuestions: number
  state: QuestionState
  feedback?: string
  hints: string[]
  controls: ReactNode
}

function renderStatusLabel(state: QuestionState) {
  if (state.status === 'correct') {
    return 'Correct! Good Job!'
  }

  if (state.status === 'gave_up') {
    return 'Answer revealed'
  }

  return 'In progress'
}

export function QuestionFrame({
  question,
  questionNumber,
  totalQuestions,
  state,
  feedback,
  hints,
  controls,
  children,
}: QuestionFrameProps) {
  return (
    <section className="question-frame">
      <header className="question-header">
        <div>
          <div className="question-kicker">Question {questionNumber}</div>
          <h2 className="question-title">{question.title}</h2>
          <p className="question-prompt">{question.prompt}</p>
          {question.instructions ? (
            <p className="question-instructions">{question.instructions}</p>
          ) : null}
        </div>
        <div className="question-status-row">
          <span className="question-index-chip">
            {questionNumber} / {totalQuestions}
          </span>
          <span className={`status-pill ${state.status}`}>
            {renderStatusLabel(state)}
          </span>
        </div>
      </header>

      {children}

      <div className="question-controls">{controls}</div>

      {feedback ? <p className="feedback">{feedback}</p> : null}

      <div className="card-list">
        {hints.length ? (
          <aside className="hint-panel">
            <h4>Hints unlocked</h4>
            <ul className="hint-list">
              {hints.map((hint, index) => (
                <li key={`${question.id}-hint-${index}`}>{hint}</li>
              ))}
            </ul>
          </aside>
        ) : null}

        {(state.status === 'correct' || state.status === 'gave_up') && question.reveal.explanation ? (
          <aside className="status-panel">
            <h4>{state.status === 'correct' ? 'Why this works' : 'Revealed answer'}</h4>
            <p className="reveal-copy">{question.reveal.explanation}</p>
          </aside>
        ) : null}
      </div>
    </section>
  )
}
