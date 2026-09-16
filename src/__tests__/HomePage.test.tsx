import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { HomePage } from '../components/HomePage'
import { publishedAssignments } from '../data/assignments'

describe('HomePage', () => {
  beforeEach(() => {
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      value: {
        length: 0,
        clear: () => undefined,
        getItem: () => null,
        key: () => null,
        removeItem: () => undefined,
        setItem: () => undefined,
      },
    })
  })

  afterEach(cleanup)

  it('uses notebook-aligned numbers for every published assignment card', () => {
    render(
      <MemoryRouter>
        <HomePage assignments={publishedAssignments} />
      </MemoryRouter>,
    )

    expect(screen.getAllByRole('heading', { level: 2 }).map((heading) => heading.textContent)).toEqual([
      '0. Numpy',
      '1. Principal Component Analysis',
      '2. K-means Clustering',
      '3. k-Nearest Neighbors',
      '4. Linear Regression',
      '5. Polynomial Regression and Overfitting',
      '6. Gradient Descent',
      '7. Batch Gradient Descent',
      '8. Regularization',
      '9. Logistic Regression',
      '10. Softmax',
    ])
  })

  it('allows every untouched assignment to be downloaded', () => {
    render(
      <MemoryRouter>
        <HomePage assignments={publishedAssignments} />
      </MemoryRouter>,
    )

    const downloadButtons = screen.getAllByRole('button', { name: 'Download JSON' })
    expect(downloadButtons).toHaveLength(publishedAssignments.length)
    downloadButtons.forEach((button) => expect(button).toBeEnabled())
  })
})
