import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { TableEntryQuestion } from '../components/questions/TableEntryQuestion'
import { pcaAssignment } from '../data/pcaAssignment'
import type { TableEntryQuestionSpec } from '../types'

describe('TableEntryQuestion', () => {
  it('enforces substep order and advances after a correct centered-data submission', async () => {
    const user = userEvent.setup()
    const onAttempt = vi.fn()
    const question = pcaAssignment.questions[6] as TableEntryQuestionSpec

    render(
      <TableEntryQuestion
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
        questionNumber={7}
        totalQuestions={pcaAssignment.questions.length}
        hints={[]}
        onAttempt={onAttempt}
      />,
    )

    const covarianceInput = screen.getByLabelText('Covariance entry 1, 1')
    expect(covarianceInput).toBeDisabled()

    await user.type(screen.getByLabelText('Centered entry 1, 1'), '-3')
    await user.type(screen.getByLabelText('Centered entry 1, 2'), '-6')
    await user.type(screen.getByLabelText('Centered entry 2, 1'), '-1')
    await user.type(screen.getByLabelText('Centered entry 2, 2'), '-2')
    await user.type(screen.getByLabelText('Centered entry 3, 1'), '1')
    await user.type(screen.getByLabelText('Centered entry 3, 2'), '2')
    await user.type(screen.getByLabelText('Centered entry 4, 1'), '3')
    await user.type(screen.getByLabelText('Centered entry 4, 2'), '6')

    await user.click(screen.getByRole('button', { name: 'Check this step' }))

    expect(onAttempt).toHaveBeenCalledWith('progress', {
      step: 'centeredData',
      values: [
        [-3, -6],
        [-1, -2],
        [1, 2],
        [3, 6],
      ],
    })
    expect(covarianceInput).not.toBeDisabled()
  })
})
