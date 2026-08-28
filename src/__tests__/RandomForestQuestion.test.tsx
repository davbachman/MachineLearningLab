import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { RandomForestQuestion } from '../components/questions/RandomForestQuestion'
import { randomForestAssignment } from '../data/randomForestAssignment'
import type { QuestionState, RandomForestQuestionSpec } from '../types'

const activeState: QuestionState = {
  status: 'active',
  attempts: 0,
  incorrectAttempts: 0,
  hintsShown: 0,
  resolvedAt: null,
  latestAnswer: null,
  attemptHistory: [],
}

afterEach(cleanup)

describe('RandomForestQuestion', () => {
  it('submits bootstrap counts and OOB selections as a raw structured payload', async () => {
    const user = userEvent.setup()
    const onAttempt = vi.fn()
    const question = randomForestAssignment.questions[0] as RandomForestQuestionSpec

    render(
      <RandomForestQuestion
        question={question}
        state={activeState}
        questionNumber={1}
        totalQuestions={randomForestAssignment.questions.length}
        hints={[]}
        onAttempt={onAttempt}
      />,
    )

    const counts: Record<string, string> = {
      A: '2', B: '0', C: '3', D: '1', E: '0', F: '1', G: '0', H: '1',
    }
    for (const [rowId, value] of Object.entries(counts)) {
      await user.type(screen.getByLabelText(`Multiplicity for row ${rowId}`), value)
    }
    for (const rowId of ['B', 'E', 'G']) {
      await user.click(screen.getByLabelText(`Row ${rowId} is out of bag`))
    }
    await user.click(screen.getByRole('button', { name: 'Check answer' }))

    expect(onAttempt).toHaveBeenCalledWith('correct', {
      multiplicities: { A: 2, B: 0, C: 3, D: 1, E: 0, F: 1, G: 0, H: 1 },
      oobIds: ['B', 'E', 'G'],
    })
  })

  it('has no give-up action and keeps a completed question editable', () => {
    const question = randomForestAssignment.questions[0] as RandomForestQuestionSpec

    render(
      <RandomForestQuestion
        question={question}
        state={{ ...activeState, status: 'correct', resolvedAt: '2026-08-27T12:00:00.000Z' }}
        questionNumber={1}
        totalQuestions={randomForestAssignment.questions.length}
        hints={[]}
        onAttempt={vi.fn()}
      />,
    )

    expect(screen.queryByRole('button', { name: /give up/i })).not.toBeInTheDocument()
    expect(screen.getByLabelText('Multiplicity for row A')).toBeEnabled()
    expect(screen.getByRole('button', { name: 'Check answer' })).toBeEnabled()
  })
})
