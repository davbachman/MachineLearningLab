import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it } from 'vitest'
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
      '5. Linear Regression',
      '6. Polynomial Regression',
      '7. Overfitting',
      '8. Gradient Descent',
      '9. Batch Gradient Descent',
      '10. Regularization',
      '11. Logistic Regression',
      '12. Softmax',
    ])
  })
})
