import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { Knn2DQuestion } from '../components/questions/Knn2DQuestion'
import { knnAssignment } from '../data/knnAssignment'
import type { Knn2DQuestionSpec } from '../types'

describe('Knn2DQuestion', () => {
  it('advances from k = 3 to k = 5 after the first correct prediction', async () => {
    const user = userEvent.setup()
    const onAttempt = vi.fn()
    const question = knnAssignment.questions[0] as Knn2DQuestionSpec

    render(
      <Knn2DQuestion
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
        questionNumber={1}
        totalQuestions={knnAssignment.questions.length}
        hints={[]}
        onAttempt={onAttempt}
      />,
    )

    await user.click(screen.getByRole('button', { name: 'Teal' }))
    await user.click(screen.getByRole('button', { name: 'Check k = 3' }))

    expect(onAttempt).toHaveBeenCalledWith('progress', {
      step: 0,
      predictedLabel: 0,
    })
    expect(screen.getByRole('button', { name: 'Check k = 5' })).toBeInTheDocument()
    expect(screen.getByText('k = 3 is correct. Now repeat the prediction for k = 5.')).toBeInTheDocument()
  })
})
