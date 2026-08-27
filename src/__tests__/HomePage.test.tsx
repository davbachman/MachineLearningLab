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

  it('numbers the published assignment cards from zero', () => {
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
    ])
  })
})
