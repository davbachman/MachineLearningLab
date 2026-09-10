import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { Knn2DQuestion } from '../components/questions/Knn2DQuestion'
import { knnAssignment } from '../data/knnAssignment'
import type { Knn2DQuestionSpec } from '../types'
import { knnInteractiveDatasets } from '../data/knnDatasets'
import { classifyGridCells, classifyKnnPoint, createGridCenters } from '../lib/knnMath'

afterEach(cleanup)

describe('Knn2DQuestion', () => {
  it('supports inspection, preserves k=1 colors, and marks only incorrect cells at k=5', async () => {
    const user = userEvent.setup()
    const onAttempt = vi.fn()
    const question = knnAssignment.questions[1] as Knn2DQuestionSpec
    const dataset = knnInteractiveDatasets.decisionBoundary
    if (dataset.kind !== 'decisionBoundary') throw new Error('Expected boundary dataset')
    const { container } = render(<Knn2DQuestion question={question}
      state={{ status: 'active', attempts: 0, incorrectAttempts: 0, hintsShown: 0,
        resolvedAt: null, latestAnswer: null, attemptHistory: [] }}
      questionNumber={2} totalQuestions={knnAssignment.questions.length} hints={[]} onAttempt={onAttempt} />)
    const cells = screen.getAllByRole('button', { name: /^Cell row/ })
    const brushes = within(screen.getByRole('radiogroup', { name: 'Choose a class brush' }))
    await user.click(screen.getByRole('checkbox', { name: /Inspect neighbors/ }))
    await user.click(cells[0])
    expect(container.querySelectorAll('[data-neighbor-index]')).toHaveLength(1)
    expect(cells[0]).toHaveAttribute('fill', '#ebe2d3')
    expect(onAttempt).not.toHaveBeenCalled()
    await user.click(screen.getByRole('checkbox', { name: /Inspect neighbors/ }))

    const targets = (k: number) => classifyGridCells(dataset.trainingPoints, dataset.bounds,
      dataset.gridColumns, dataset.gridRows, k, dataset.metric)
    const first = targets(1)
    for (const label of [0, 1]) {
      await user.click(brushes.getByRole('button', { name: dataset.classes[label].label }))
      for (const [index, value] of first.entries()) if (value === label) await user.click(cells[index])
    }
    await user.click(screen.getByRole('button', { name: 'Check k = 1 boundary' }))
    expect(onAttempt).toHaveBeenLastCalledWith('progress', { step: 0, cells: first })
    for (const [index, label] of first.entries()) {
      expect(cells[index]).toHaveAttribute('fill', dataset.classes[label].color)
    }

    await user.hover(cells[15])
    const center = createGridCenters(dataset.bounds, dataset.gridColumns, dataset.gridRows)[15]
    const expectedNeighbors = classifyKnnPoint(dataset.trainingPoints, center, 5, dataset.metric).neighbors
    expect(Array.from(container.querySelectorAll('[data-neighbor-index]'), (element) =>
      Number(element.getAttribute('data-neighbor-index')))).toEqual(expectedNeighbors.map((neighbor) => neighbor.index))

    await user.click(screen.getByRole('button', { name: 'Check k = 5 boundary' }))
    const second = targets(5)
    const wrong = second.flatMap((label, index) => label === first[index] ? [] : [index])
    expect(wrong.length).toBeGreaterThan(0)
    expect(screen.getAllByRole('button', { name: /needs correction/ })).toHaveLength(wrong.length)
    for (const index of wrong) {
      expect(cells[index]).toHaveAccessibleName(/needs correction/)
      await user.click(brushes.getByRole('button', { name: dataset.classes[second[index]].label }))
      await user.click(cells[index])
      expect(cells[index]).not.toHaveAccessibleName(/needs correction/)
    }
    await user.click(screen.getByRole('button', { name: 'Check k = 5 boundary' }))
    expect(onAttempt).toHaveBeenLastCalledWith('correct', { step: 1, cells: second })
    expect(screen.queryByRole('button', { name: /needs correction/ })).not.toBeInTheDocument()
  })

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
