import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { AssignmentPage } from '../components/AssignmentPage'
import { numpyAssignment } from '../data/numpyAssignment'

describe('AssignmentPage', () => {
  beforeEach(() => {
    const values = new Map<string, string>()
    Object.defineProperty(window, 'localStorage', {
      configurable: true,
      value: {
        get length() {
          return values.size
        },
        clear: () => values.clear(),
        getItem: (key: string) => values.get(key) ?? null,
        key: (index: number) => Array.from(values.keys())[index] ?? null,
        removeItem: (key: string) => values.delete(key),
        setItem: (key: string, value: string) => values.set(key, value),
      },
    })
  })

  afterEach(cleanup)

  it('lets students jump directly to any question from the sidebar', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter>
        <AssignmentPage assignment={numpyAssignment} />
      </MemoryRouter>,
    )

    expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent(
      'Construct and Reshape Arrays',
    )

    await user.click(screen.getByRole('button', { name: /5\. Notebook Lab: Trace the NumPy Arrays/ }))
    expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent(
      'Notebook Lab: Trace the NumPy Arrays',
    )

    await user.click(screen.getByRole('button', { name: /1\. Construct and Reshape Arrays/ }))
    expect(screen.getByRole('heading', { level: 2 })).toHaveTextContent(
      'Construct and Reshape Arrays',
    )
    expect(screen.queryByRole('button', { name: /give up/i })).not.toBeInTheDocument()
  })
})
