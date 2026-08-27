import { Link } from 'react-router-dom'
import type { AssignmentSpec } from '../types'
import {
  createInitialAssignmentState,
  getResolvedCount,
  isAssignmentComplete,
} from '../lib/assignmentState'
import { loadAssignmentState, clearAssignmentState, saveAssignmentState } from '../lib/storage'
import { downloadSubmissionExport } from '../lib/export'
import { useDebugState } from '../lib/debugState'

interface HomePageProps {
  assignments: AssignmentSpec[]
}

export function HomePage({ assignments }: HomePageProps) {
  const states = assignments.map((assignment) => {
    const state = loadAssignmentState(assignment)
    return {
      assignment,
      state,
      resolved: getResolvedCount(state),
      complete: isAssignmentComplete(state),
    }
  })

  useDebugState({
    route: 'home',
    assignments: states.map(({ assignment, resolved, complete }) => ({
      id: assignment.id,
      resolved,
      total: assignment.questions.length,
      complete,
    })),
  })

  const resetAssignment = (assignment: AssignmentSpec) => {
    clearAssignmentState(assignment.id)
    saveAssignmentState(assignment, createInitialAssignmentState(assignment))
    window.location.reload()
  }

  return (
    <main className="app-shell">
      <div className="surface hero-shell">
        <header className="hero-header">
          <div>
            <div className="hero-kicker">CS158 interactive check-ins</div>
            <h1 className="hero-title">Machine learning</h1>
            <p className="hero-copy">
              Each assignment is a guided sequence of questions with unlimited retries, optional give-up
              reveals, browser-local progress, and a JSON export for Gradescope upload at the end.
            </p>
          </div>
        </header>

        <section className="assignments-grid">
          {states.map(({ assignment, state, resolved, complete }) => (
            <article key={assignment.id} className="assignment-card">
              <div>
                <div className="assignment-meta">
                  <span className="pill">{assignment.topic}</span>
                  <span>
                    {resolved} / {assignment.questions.length} resolved
                  </span>
                </div>
                <h2>{assignment.title}</h2>
                <p className="assignment-subtitle">{assignment.description}</p>
              </div>

              <div className="progress-bar-shell" aria-hidden="true">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${(resolved / assignment.questions.length) * 100}%` }}
                />
              </div>

              <div className="assignment-actions">
                <Link className="button" to={`/assignment/${assignment.id}`}>
                  {resolved > 0 ? 'Resume assignment' : 'Start assignment'}
                </Link>

                <button
                  type="button"
                  className="button-ghost"
                  onClick={() => resetAssignment(assignment)}
                >
                  Reset
                </button>

                <button
                  type="button"
                  className="button-secondary"
                  disabled={!complete}
                  onClick={() => downloadSubmissionExport(assignment, state)}
                >
                  Download JSON
                </button>
              </div>
            </article>
          ))}
        </section>
      </div>
    </main>
  )
}
