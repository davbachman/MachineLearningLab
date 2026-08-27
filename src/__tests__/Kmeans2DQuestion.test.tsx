import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { Kmeans2DQuestion } from '../components/questions/Kmeans2DQuestion'
import { kmeansAssignment } from '../data/kmeansAssignment'
import { kmeansInteractiveDatasets } from '../data/kmeansDatasets'
import type { Kmeans2DQuestionSpec } from '../types'

describe('Kmeans2DQuestion', () => {
  it('advances from the first assignment step to the first centroid-update step', async () => {
    const user = userEvent.setup()
    const onAttempt = vi.fn()
    const question = kmeansAssignment.questions[0] as Kmeans2DQuestionSpec
    const dataset = kmeansInteractiveDatasets.fullIteration

    render(
      <Kmeans2DQuestion
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
        totalQuestions={kmeansAssignment.questions.length}
        hints={[]}
        onAttempt={onAttempt}
        onGiveUp={vi.fn()}
      />,
    )

    for (const [index, assignment] of (dataset.iterationAssignments?.[0] ?? []).entries()) {
      const point = screen.getByRole('button', { name: `Point ${index + 1}` })
      for (let click = 0; click < assignment; click += 1) {
        await user.click(point)
      }
    }

    await user.click(screen.getByRole('button', { name: 'Check assignments' }))

    expect(onAttempt).toHaveBeenCalledWith(
      'progress',
      expect.objectContaining({ step: 'assignments', iteration: 0 }),
    )
    expect(screen.getByRole('button', { name: 'Check centroids' })).toBeInTheDocument()
    expect(
      screen.getByText('Iteration 1 assignments confirmed. Now drag the centroids to the means of the colored points.'),
    ).toBeInTheDocument()
  })
})
