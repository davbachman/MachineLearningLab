import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { DecisionTreeQuestion } from '../components/questions/DecisionTreeQuestion'
import { decisionTreeAssignment } from '../data/decisionTreeAssignment'
import type { DecisionTreeQuestionSpec } from '../types'

describe('DecisionTreeQuestion', () => {
  it('records the complete three-stage trace for a proposed split', async () => {
    const user = userEvent.setup()
    const onAttempt = vi.fn()
    const question = decisionTreeAssignment.questions[1] as DecisionTreeQuestionSpec

    render(
      <DecisionTreeQuestion
        question={question}
        state={{
          status: 'active',
          attempts: 0,
          incorrectAttempts: 0,
          hintsShown: 0,
          resolvedAt: null,
          latestAnswer: null,
          attemptHistory: [],
        }}
        questionNumber={2}
        totalQuestions={decisionTreeAssignment.questions.length}
        hints={[]}
        onAttempt={onAttempt}
        onGiveUp={vi.fn()}
      />,
    )

    for (const rowId of ['A', 'B', 'C', 'D']) {
      await user.click(screen.getByRole('button', { name: `Row ${rowId}` }))
    }
    await user.click(screen.getByRole('button', { name: 'Check row routing' }))

    expect(onAttempt).toHaveBeenLastCalledWith(
      'progress',
      expect.objectContaining({ step: 'partition', leftRowIds: ['A', 'B', 'C', 'D'] }),
    )

    await user.type(screen.getByLabelText('Left child did not pass count'), '3')
    await user.type(screen.getByLabelText('Left child passed count'), '1')
    await user.type(screen.getByLabelText('Right child did not pass count'), '1')
    await user.type(screen.getByLabelText('Right child passed count'), '3')
    await user.type(screen.getByLabelText('Left child Gini'), '0.375')
    await user.type(screen.getByLabelText('Right child Gini'), '0.375')
    await user.click(screen.getByRole('button', { name: 'Check child statistics' }))

    expect(onAttempt).toHaveBeenLastCalledWith(
      'progress',
      expect.objectContaining({
        step: 'childStats',
        counts: {
          left: { negative: 3, positive: 1 },
          right: { negative: 1, positive: 3 },
        },
        values: { left: 0.375, right: 0.375 },
      }),
    )
    expect(screen.getByLabelText('Weighted split Gini')).toBeInTheDocument()

    await user.type(screen.getByLabelText('Weighted split Gini'), '0.375')
    await user.click(screen.getByRole('button', { name: 'Check weighted Gini' }))

    expect(onAttempt).toHaveBeenLastCalledWith(
      'correct',
      expect.objectContaining({ step: 'weightedGini', value: 0.375 }),
    )
  })
})
