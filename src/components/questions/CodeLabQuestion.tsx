import { useState } from 'react'
import { DEFAULT_HINT_SCHEDULE } from '../../lib/assignmentState'
import { tokenizePythonLine } from '../../lib/pythonSyntax'
import {
  makeCodeLabSubmission,
  validateCodeLabStage,
  type CodeLabAnswers,
  type CodeLabQuestionSpec,
  type CodeLabSubmission,
  type CodeLabStageKind,
} from '../../lib/codeLab'
import type { AttemptOutcome, QuestionSpec, QuestionState } from '../../types'
import { QuestionFrame } from '../QuestionFrame'
import './CodeLabQuestion.css'

interface CodeLabQuestionProps {
  question: CodeLabQuestionSpec
  state: QuestionState
  questionNumber: number
  totalQuestions: number
  hints: string[]
  initialSubmission?: CodeLabSubmission
  onAttempt: (outcome: AttemptOutcome, answer: unknown) => void
}

const stageKickers: Record<CodeLabStageKind, string> = {
  executionTrace: 'Execution trace',
  diagnoseMutation: 'Mutation diagnosis',
  distinguishingTest: 'Distinguishing test',
}

function copyInitialAnswers(
  question: CodeLabQuestionSpec,
  initialSubmission: CodeLabSubmission | undefined,
): CodeLabAnswers {
  if (
    !initialSubmission ||
    initialSubmission.formatVersion !== 1 ||
    initialSubmission.questionId !== question.id ||
    initialSubmission.variantId !== question.variantId ||
    !initialSubmission.stages ||
    typeof initialSubmission.stages !== 'object' ||
    Array.isArray(initialSubmission.stages)
  ) {
    return {}
  }

  return Object.fromEntries(
    Object.entries(initialSubmission.stages).map(([stageId, fields]) => [
      stageId,
      { ...fields },
    ]),
  )
}

function firstUnfinishedStage(question: CodeLabQuestionSpec, answers: CodeLabAnswers) {
  const index = question.stages.findIndex(
    (stage) => !validateCodeLabStage(stage, answers[stage.id]).correct,
  )
  return index === -1 ? question.stages.length - 1 : index
}

function actionLabel(kind: CodeLabStageKind) {
  if (kind === 'executionTrace') {
    return 'Check predicted trace'
  }
  if (kind === 'diagnoseMutation') {
    return 'Check diagnosis'
  }
  return 'Check distinguishing test'
}

function statementIdForLine(line: string) {
  return line.match(/# (S\d+)\s*$/)?.[1]
}

function PythonTokens({ line }: { line: string }) {
  if (!line) {
    return <> </>
  }

  return (
    <>
      {tokenizePythonLine(line).map((token, index) => (
        <span className={`python-token-${token.kind}`} key={`${index}-${token.kind}`}>
          {token.text}
        </span>
      ))}
    </>
  )
}

function HighlightedPython({ source }: { source: string }) {
  return (
    <code>
      {source.split('\n').map((line, index) => (
        <span className="code-lab-highlight-line" key={`python-line-${index}`}>
          <PythonTokens line={line} />
        </span>
      ))}
    </code>
  )
}

function CodePanel({ question }: { question: CodeLabQuestionSpec }) {
  return (
    <section className="code-lab-code-panel" aria-label="Reference Python implementation">
      <div className="code-lab-panel-header">
        <div>
          <span className="code-lab-eyebrow">Reference implementation</span>
          <h3>Readable {question.language}</h3>
        </div>
        <span className="code-lab-language-chip">{question.language}</span>
      </div>
      <pre className="code-lab-code">
        <code>
          {question.code.split('\n').map((line, index) => {
            const statementId = statementIdForLine(line)
            return (
              <span
                className={statementId ? 'code-lab-code-line tagged' : 'code-lab-code-line'}
                data-statement-id={statementId}
                key={`${question.id}-line-${index}`}
              >
                <span className="code-lab-line-number" aria-hidden="true">
                  {index + 1}
                </span>
                <span className="code-lab-code-text">
                  <PythonTokens line={line} />
                </span>
              </span>
            )
          })}
        </code>
      </pre>
    </section>
  )
}

export function CodeLabQuestion({
  question,
  state,
  questionNumber,
  totalQuestions,
  hints,
  initialSubmission,
  onAttempt,
}: CodeLabQuestionProps) {
  const [answers, setAnswers] = useState<CodeLabAnswers>(() =>
    copyInitialAnswers(question, initialSubmission),
  )
  const [stageIndex, setStageIndex] = useState(() =>
    state.status === 'correct' || state.status === 'gave_up'
      ? 0
      : firstUnfinishedStage(question, copyInitialAnswers(question, initialSubmission)),
  )
  const [feedback, setFeedback] = useState('')
  const currentStage = question.stages[stageIndex]
  const schedule = question.hintSchedule ?? DEFAULT_HINT_SCHEDULE

  const displayedAnswer = (stageId: string, fieldId: string) =>
    answers[stageId]?.[fieldId]

  const selectAnswer = (stageId: string, fieldId: string, optionId: string) => {
    setAnswers((current) => ({
      ...current,
      [stageId]: {
        ...current[stageId],
        [fieldId]: optionId,
      },
    }))
  }

  const submitCurrentStage = () => {
    const submission = makeCodeLabSubmission(question, answers)
    const result = validateCodeLabStage(currentStage, answers[currentStage.id])

    if (!result.correct) {
      const nextIncorrectAttempts = state.incorrectAttempts + 1
      const baseMessage = result.missingFieldIds.length
        ? 'Choose one answer for every field in this stage.'
        : 'At least one selection is not consistent with the code trace.'
      setFeedback(
        schedule.includes(nextIncorrectAttempts)
          ? 'Not yet. A new hint appeared below.'
          : baseMessage,
      )
      onAttempt('incorrect', submission)
      return
    }

    if (stageIndex < question.stages.length - 1) {
      setStageIndex((current) => current + 1)
      setFeedback(currentStage.successCopy)
      onAttempt('progress', submission)
      return
    }

    const finalResult = question.validator(submission)
    if (!finalResult.correct) {
      setFeedback(finalResult.message ?? 'One or more Code Lab answers are not correct yet.')
      onAttempt('incorrect', submission)
      return
    }

    setFeedback(question.successCopy ?? 'Correct! Good Job!')
    onAttempt('correct', submission)
  }

  return (
    <QuestionFrame
      question={question as unknown as QuestionSpec}
      questionNumber={questionNumber}
      totalQuestions={totalQuestions}
      state={state}
      feedback={feedback}
      hints={hints}
      controls={
        <>
          <button type="button" className="button" onClick={submitCurrentStage}>
            {actionLabel(currentStage.kind)}
          </button>
        </>
      }
    >
      <div className="code-lab-workspace">
        <CodePanel question={question} />

        <div className="code-lab-run-context">
          <section className="code-lab-fixture" aria-label={question.fixtureTitle}>
            <span className="code-lab-eyebrow">{question.fixtureTitle}</span>
            <h3>{question.fixtureHeading ?? 'Values defined before the run'}</h3>
            <pre>
              <HighlightedPython source={question.fixture} />
            </pre>
          </section>

          <section className="code-lab-invocation" aria-label="Executed Python statements">
            <span className="code-lab-eyebrow">Executed Python</span>
            <h3>{question.invocationTitle}</h3>
            <p>
              {question.invocationLead ??
                'After defining the values at left, Python executes the following:'}
            </p>
            <pre>
              <HighlightedPython source={question.invocation} />
            </pre>
            <p className="code-lab-invocation-note">
              The execution-trace questions below ask you to predict values produced by these
              statements.
            </p>
          </section>
        </div>
      </div>

      <ol className="code-lab-stage-list">
        {question.stages.map((stage, index) => {
          const isPast = index < stageIndex
          const isCurrent = index === stageIndex
          const isLocked = index > stageIndex
          const stageStatus = isPast ? 'Locked in' : isCurrent ? 'Current stage' : 'Locked'

          return (
            <li
              className={`code-lab-stage ${isCurrent ? 'current' : ''} ${isLocked ? 'locked' : ''}`}
              key={stage.id}
              aria-current={isCurrent ? 'step' : undefined}
            >
              <div className="code-lab-stage-header">
                <div>
                  <span className="code-lab-eyebrow">{stageKickers[stage.kind]}</span>
                  <h3>{stage.title}</h3>
                </div>
                <span className="status-pill">{stageStatus}</span>
              </div>
              <p className="code-lab-stage-prompt">{stage.prompt}</p>

              <div className="code-lab-fields">
                {stage.fields.map((field) => {
                  return (
                    <fieldset
                      className="code-lab-field"
                      disabled={!isCurrent}
                      key={field.id}
                    >
                      <legend>{field.label}</legend>
                      <div className="choice-list">
                        {field.options.map((option) => {
                          const checked = displayedAnswer(stage.id, field.id) === option.id
                          return (
                            <label
                              className={`choice-option ${checked ? 'selected' : ''}`}
                              key={option.id}
                            >
                              <input
                                type="radio"
                                name={`${question.id}-${stage.id}-${field.id}`}
                                value={option.id}
                                checked={checked}
                                onChange={() => selectAnswer(stage.id, field.id, option.id)}
                              />
                              <span className="choice-copy">
                                <strong>
                                  {stage.kind === 'executionTrace' ? (
                                    <code className="code-lab-choice-answer">{option.label}</code>
                                  ) : (
                                    option.label
                                  )}
                                </strong>
                                {option.description ? <span>{option.description}</span> : null}
                              </span>
                            </label>
                          )
                        })}
                      </div>
                    </fieldset>
                  )
                })}
              </div>

              {isPast ? <p className="code-lab-stage-success">{stage.successCopy}</p> : null}
            </li>
          )
        })}
      </ol>
    </QuestionFrame>
  )
}
