import { lazy, Suspense, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  applyAttemptOutcome,
  applyGiveUp,
  clearAssignmentProgress,
  getCurrentOrFirstActiveIndex,
  getResolvedCount,
  getVisibleHints,
  isAssignmentComplete,
  markAssignmentExported,
} from '../lib/assignmentState'
import { clearAssignmentState, loadAssignmentState, saveAssignmentState } from '../lib/storage'
import { downloadSubmissionExport } from '../lib/export'
import type { AssignmentSpec, QuestionState } from '../types'
import { useDebugState } from '../lib/debugState'
import { DecisionTreeQuestion } from './questions/DecisionTreeQuestion'
import { Kmeans2DQuestion } from './questions/Kmeans2DQuestion'
import { Knn2DQuestion } from './questions/Knn2DQuestion'
import { MultipleChoiceQuestion } from './questions/MultipleChoiceQuestion'
import { Pca2DQuestion } from './questions/Pca2DQuestion'
import { TableEntryQuestion } from './questions/TableEntryQuestion'
import { CodeLabQuestion } from './questions/CodeLabQuestion'
import type { CodeLabSubmission } from '../lib/codeLab'
import { RandomForestQuestion } from './questions/RandomForestQuestion'

const Pca3DQuestion = lazy(async () => {
  const module = await import('./questions/Pca3DQuestion')
  return { default: module.Pca3DQuestion }
})

interface AssignmentPageProps {
  assignment: AssignmentSpec
}

function questionStatusLabel(questionState: QuestionState) {
  if (questionState.status === 'correct') {
    return 'Correct'
  }

  if (questionState.status === 'gave_up') {
    return 'Revealed'
  }

  if (questionState.status === 'active') {
    return 'Current'
  }

  return 'Locked'
}

export function AssignmentPage({ assignment }: AssignmentPageProps) {
  const [state, setState] = useState(() => loadAssignmentState(assignment))
  const [viewIndex, setViewIndex] = useState(() => getCurrentOrFirstActiveIndex(loadAssignmentState(assignment)))

  useEffect(() => {
    saveAssignmentState(assignment, state)
  }, [assignment, state])

  const activeViewIndex =
    state.questionStates[viewIndex]?.status === 'locked'
      ? getCurrentOrFirstActiveIndex(state)
      : viewIndex
  const currentQuestion = assignment.questions[activeViewIndex]
  const currentQuestionState = state.questionStates[activeViewIndex]
  const resolvedCount = getResolvedCount(state)
  const complete = isAssignmentComplete(state)
  const progressPercent = (resolvedCount / assignment.questions.length) * 100

  useDebugState({
    route: 'assignment',
    assignmentId: assignment.id,
    questionId: currentQuestion.id,
    questionNumber: activeViewIndex + 1,
    status: currentQuestionState.status,
    attempts: currentQuestionState.attempts,
    incorrectAttempts: currentQuestionState.incorrectAttempts,
    complete,
  })

  const persistNextState = (nextState: typeof state) => {
    setState(nextState)
    saveAssignmentState(assignment, nextState)
  }

  const handleAttempt = (outcome: 'incorrect' | 'progress' | 'correct', answer: unknown) => {
    persistNextState(applyAttemptOutcome(state, assignment, activeViewIndex, outcome, answer))
  }

  const handleGiveUp = (answer: unknown) => {
    persistNextState(applyGiveUp(state, assignment, activeViewIndex, answer))
  }

  const handleReset = () => {
    clearAssignmentState(assignment.id)
    const initial = clearAssignmentProgress(assignment)
    setState(initial)
    setViewIndex(0)
    saveAssignmentState(assignment, initial)
  }

  const handleDownload = () => {
    const nextState = markAssignmentExported(state)
    persistNextState(nextState)
    downloadSubmissionExport(assignment, nextState)
  }

  const nextUnlocked = state.questionStates[activeViewIndex + 1]?.status === 'active'
  const hints = getVisibleHints(currentQuestion, currentQuestionState)

  return (
    <main className="app-shell">
      <div className="surface assignment-shell">
        <header className="assignment-header">
          <div>
            <div className="assignment-kicker">
              <Link to="/">Home</Link> / {assignment.topic}
            </div>
            <h1 className="assignment-title">{assignment.title}</h1>
            <p className="assignment-subtitle">{assignment.description}</p>
          </div>

          <div style={{ minWidth: 'min(100%, 340px)', display: 'grid', gap: '0.85rem' }}>
            <div className="progress-bar-shell" aria-hidden="true">
              <div className="progress-bar-fill" style={{ width: `${progressPercent}%` }} />
            </div>
            <div className="assignment-actions">
              <span className="status-pill">
                {resolvedCount} / {assignment.questions.length} resolved
              </span>
              <button type="button" className="button-secondary" disabled={!complete} onClick={handleDownload}>
                Download JSON
              </button>
              <button type="button" className="button-ghost" onClick={handleReset}>
                Reset assignment
              </button>
            </div>
          </div>
        </header>

        <div className="assignment-layout">
          <aside className="progress-panel">
            <h3>Question flow</h3>
            <div className="progress-list">
              {assignment.questions.map((question, index) => {
                const questionState = state.questionStates[index]
                return (
                  <div
                    key={question.id}
                    className={`progress-item ${index === activeViewIndex ? 'active' : ''} ${questionState.status === 'locked' ? 'locked' : ''}`}
                  >
                    <div className="progress-label">
                      <span>
                        {index + 1}. {question.title}
                      </span>
                      <span className={`status-pill ${questionState.status}`}>{questionStatusLabel(questionState)}</span>
                    </div>
                    <div className="progress-description">
                      Attempts: {questionState.attempts} | Hints shown: {questionState.hintsShown}
                    </div>
                  </div>
                )
              })}
            </div>
          </aside>

          <section className="question-strip">
            {currentQuestion.kind === 'pca2dLine' ? (
              <Pca2DQuestion
                key={currentQuestion.id}
                question={currentQuestion}
                state={currentQuestionState}
                questionNumber={activeViewIndex + 1}
                totalQuestions={assignment.questions.length}
                hints={hints}
                onAttempt={handleAttempt}
                onGiveUp={handleGiveUp}
              />
            ) : null}

            {currentQuestion.kind === 'multipleChoice' ? (
              <MultipleChoiceQuestion
                key={currentQuestion.id}
                question={currentQuestion}
                state={currentQuestionState}
                questionNumber={activeViewIndex + 1}
                totalQuestions={assignment.questions.length}
                hints={hints}
                onAttempt={handleAttempt}
                onGiveUp={handleGiveUp}
              />
            ) : null}

            {currentQuestion.kind === 'kmeans2d' ? (
              <Kmeans2DQuestion
                key={currentQuestion.id}
                question={currentQuestion}
                state={currentQuestionState}
                questionNumber={activeViewIndex + 1}
                totalQuestions={assignment.questions.length}
                hints={hints}
                onAttempt={handleAttempt}
                onGiveUp={handleGiveUp}
              />
            ) : null}

            {currentQuestion.kind === 'knn2d' ? (
              <Knn2DQuestion
                key={currentQuestion.id}
                question={currentQuestion}
                state={currentQuestionState}
                questionNumber={activeViewIndex + 1}
                totalQuestions={assignment.questions.length}
                hints={hints}
                onAttempt={handleAttempt}
                onGiveUp={handleGiveUp}
              />
            ) : null}

            {currentQuestion.kind === 'decisionTree' ? (
              <DecisionTreeQuestion
                key={currentQuestion.id}
                question={currentQuestion}
                state={currentQuestionState}
                questionNumber={activeViewIndex + 1}
                totalQuestions={assignment.questions.length}
                hints={hints}
                onAttempt={handleAttempt}
                onGiveUp={handleGiveUp}
              />
            ) : null}

            {currentQuestion.kind === 'randomForest' ? (
              <RandomForestQuestion
                key={currentQuestion.id}
                question={currentQuestion}
                state={currentQuestionState}
                questionNumber={activeViewIndex + 1}
                totalQuestions={assignment.questions.length}
                hints={hints}
                onAttempt={handleAttempt}
                onGiveUp={handleGiveUp}
              />
            ) : null}

            {currentQuestion.kind === 'pca3dPlane' ? (
              <Suspense fallback={<section className="question-frame"><p className="feedback">Loading the 3D PCA scene…</p></section>}>
                <Pca3DQuestion
                  key={currentQuestion.id}
                  question={currentQuestion}
                  state={currentQuestionState}
                  questionNumber={activeViewIndex + 1}
                  totalQuestions={assignment.questions.length}
                  hints={hints}
                  onAttempt={handleAttempt}
                  onGiveUp={handleGiveUp}
                />
              </Suspense>
            ) : null}

            {currentQuestion.kind === 'tableEntry' ? (
              <TableEntryQuestion
                key={currentQuestion.id}
                question={currentQuestion}
                state={currentQuestionState}
                questionNumber={activeViewIndex + 1}
                totalQuestions={assignment.questions.length}
                hints={hints}
                onAttempt={handleAttempt}
                onGiveUp={handleGiveUp}
              />
            ) : null}

            {currentQuestion.kind === 'codeLab' ? (
              <CodeLabQuestion
                key={currentQuestion.id}
                question={currentQuestion}
                state={currentQuestionState}
                questionNumber={activeViewIndex + 1}
                totalQuestions={assignment.questions.length}
                hints={hints}
                initialSubmission={
                  currentQuestionState.latestAnswer &&
                  typeof currentQuestionState.latestAnswer === 'object' &&
                  !Array.isArray(currentQuestionState.latestAnswer)
                    ? (currentQuestionState.latestAnswer as unknown as CodeLabSubmission)
                    : undefined
                }
                onAttempt={handleAttempt}
                onGiveUp={handleGiveUp}
              />
            ) : null}

            <div className="inline-actions" style={{ marginTop: '1rem' }}>
              <Link className="button-ghost" to="/">
                Back to assignments
              </Link>
              <button
                type="button"
                className="button"
                onClick={() => setViewIndex((current) => current + 1)}
                disabled={!nextUnlocked}
              >
                Next question
              </button>
              {complete ? (
                <button type="button" className="button-secondary" onClick={handleDownload}>
                  Download completed JSON
                </button>
              ) : null}
            </div>
          </section>
        </div>
      </div>
    </main>
  )
}
