import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import { RegressionVisual } from '../components/questions/RegressionVisual'

afterEach(cleanup)
it('renders three curve panels on identical scales without revealing degrees', () => {
  render(<RegressionVisual dataset={{ id:'test',kind:'regressionVisual',mode:'curves',caption:'Same training points.' }} />)
  expect(screen.getAllByRole('img')).toHaveLength(3)
  expect(screen.getByRole('heading',{name:'Model B'})).toBeInTheDocument()
  expect(screen.queryByText('Degree 1')).not.toBeInTheDocument()
})
it('provides a numeric table alongside the training/validation graph', () => {
  render(<RegressionVisual dataset={{id:'test',kind:'regressionVisual',mode:'errors',caption:'Held-out errors.'}} />)
  expect(screen.getByRole('img')).toHaveAttribute('aria-label', expect.stringContaining('Training and validation MSE'))
  expect(screen.getAllByRole('row')).toHaveLength(9)
  expect(screen.getByText('24.33')).toBeInTheDocument()
  expect(screen.getByText('293.00')).toBeInTheDocument()
})
