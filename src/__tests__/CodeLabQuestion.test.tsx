import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { CodeLabQuestion } from '../components/questions/CodeLabQuestion'
import { pcaCodeLab } from '../data/codeLabQuestions'
import type { QuestionState } from '../types'

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

function correctOptionInput(stageIndex: number, fieldIndex: number) {
  const field = pcaCodeLab.stages[stageIndex].fields[fieldIndex]
  return document.querySelector<HTMLInputElement>(
    `input[value="${field.correctOptionId}"]`,
  )!
}

describe('CodeLabQuestion', () => {
  it('shows the exact Python command that connects the fixture to the trace questions', () => {
    render(
      <CodeLabQuestion
        question={pcaCodeLab}
        state={activeState}
        questionNumber={7}
        totalQuestions={7}
        hints={[]}
        onAttempt={vi.fn()}
        onGiveUp={vi.fn()}
      />,
    )

    const executedCommand = screen.getByRole('region', { name: 'Executed Python statements' })
    expect(within(executedCommand).getByText('Statements to trace')).toBeInTheDocument()
    expect(executedCommand).toHaveTextContent('projected = pca.fit_transform(X_trace)')
    expect(executedCommand.querySelector('.python-token-operator')).toHaveTextContent('=')
    expect(
      within(executedCommand).getByText(/execution-trace questions below ask you to predict/),
    ).toBeInTheDocument()
  })

  it('syntax-colors the reference implementation and executable setup', () => {
    render(
      <CodeLabQuestion
        question={pcaCodeLab}
        state={activeState}
        questionNumber={7}
        totalQuestions={7}
        hints={[]}
        onAttempt={vi.fn()}
        onGiveUp={vi.fn()}
      />,
    )

    const implementation = screen.getByRole('region', {
      name: 'Reference Python implementation',
    })
    const setup = screen.getByRole('region', { name: 'Notebook class with a small trace matrix' })

    expect(implementation.querySelector('.python-token-keyword')).toHaveTextContent('import')
    expect(implementation.querySelector('.python-token-class')).toHaveTextContent('PCA')
    expect(implementation.querySelector('.python-token-comment')).toHaveTextContent('# S1')
    expect(setup.querySelector('.python-token-number')).toHaveTextContent('2.0')
  })

  it('locks a correct trace and sends the raw structured answer with progress', async () => {
    const user = userEvent.setup()
    const onAttempt = vi.fn()

    render(
      <CodeLabQuestion
        question={pcaCodeLab}
        state={activeState}
        questionNumber={7}
        totalQuestions={7}
        hints={[]}
        onAttempt={onAttempt}
        onGiveUp={vi.fn()}
      />,
    )

    await user.click(correctOptionInput(0, 0))
    await user.click(correctOptionInput(0, 1))
    await user.click(correctOptionInput(0, 2))
    await user.click(screen.getByRole('button', { name: 'Check predicted trace' }))

    expect(onAttempt).toHaveBeenCalledWith('progress', {
      formatVersion: 1,
      questionId: 'pca-code-lab',
      variantId: 'pca-notebook-v1',
      stages: {
        TRACE: {
          BASIS_SHAPE: 'SHAPE_2_1',
          PROJECTED: 'PROJECTED_NEG2_0_2',
          NEW_PROJECTED: 'NEW_NEG1_1',
        },
        MUTATION: {},
        TEST: {},
      },
    })
    expect(screen.getByRole('button', { name: 'Check diagnosis' })).toBeEnabled()
    expect(screen.getAllByText(/Correct:.*single principal direction.*\[-1, 1\]/)).toHaveLength(2)
  })

  it('records the current pre-reveal draft when the student gives up', async () => {
    const user = userEvent.setup()
    const onGiveUp = vi.fn()

    render(
      <CodeLabQuestion
        question={pcaCodeLab}
        state={activeState}
        questionNumber={7}
        totalQuestions={7}
        hints={[]}
        onAttempt={vi.fn()}
        onGiveUp={onGiveUp}
      />,
    )

    await user.click(screen.getByRole('radio', { name: '(1, 2)' }))
    await user.click(screen.getByRole('button', { name: 'Give up' }))

    expect(onGiveUp).toHaveBeenCalledWith({
      formatVersion: 1,
      questionId: 'pca-code-lab',
      variantId: 'pca-notebook-v1',
      stages: {
        TRACE: { BASIS_SHAPE: 'SHAPE_1_2' },
        MUTATION: {},
        TEST: {},
      },
    })
  })

  it('returns one complete autograder-ready payload after all three stages', async () => {
    const user = userEvent.setup()
    const onAttempt = vi.fn()

    render(
      <CodeLabQuestion
        question={pcaCodeLab}
        state={activeState}
        questionNumber={7}
        totalQuestions={7}
        hints={[]}
        onAttempt={onAttempt}
        onGiveUp={vi.fn()}
      />,
    )

    for (const [fieldIndex] of pcaCodeLab.stages[0].fields.entries()) {
      await user.click(correctOptionInput(0, fieldIndex))
    }
    await user.click(screen.getByRole('button', { name: 'Check predicted trace' }))

    await user.click(correctOptionInput(1, 0))
    await user.click(screen.getByRole('button', { name: 'Check diagnosis' }))

    await user.click(correctOptionInput(2, 0))
    await user.click(screen.getByRole('button', { name: 'Check distinguishing test' }))

    expect(onAttempt).toHaveBeenLastCalledWith('correct', {
      formatVersion: 1,
      questionId: 'pca-code-lab',
      variantId: 'pca-notebook-v1',
      stages: {
        TRACE: {
          BASIS_SHAPE: 'SHAPE_2_1',
          PROJECTED: 'PROJECTED_NEG2_0_2',
          NEW_PROJECTED: 'NEW_NEG1_1',
        },
        MUTATION: { CHANGED_S6_EFFECT: 'CHANGED_10_12' },
        TEST: { TEST_ID: 'TEST_TRANSFORM_MEAN_ZERO' },
      },
    })
  })

  it('keeps autograder IDs in radio values and submissions, not in student-facing text', () => {
    const { container } = render(
      <CodeLabQuestion
        question={pcaCodeLab}
        state={activeState}
        questionNumber={7}
        totalQuestions={7}
        hints={[]}
        onAttempt={vi.fn()}
        onGiveUp={vi.fn()}
      />,
    )

    expect(container).not.toHaveTextContent('SHAPE_2_1')
    expect(container).not.toHaveTextContent('CHANGED_10_12')
    expect(container.querySelector('input[value="SHAPE_2_1"]')).toBeInTheDocument()
  })
})
