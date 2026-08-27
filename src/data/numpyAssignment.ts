import type { AssignmentSpec, MultipleChoiceQuestionSpec } from '../types'
import { validateMultipleChoiceSelections } from '../lib/questionValidation'
import { numpyNotebookLab } from './codeLabQuestions'

export const numpyAssignment: AssignmentSpec = {
  id: 'numpy',
  version: 1,
  title: 'Numpy',
  topic: 'Python Foundations',
  description:
    'Practice the NumPy array shapes, indexing, slicing, Boolean masks, reductions, and matrix operations used throughout the course.',
  published: true,
  questions: [
    {
      id: 'numpy-shapes-indexing',
      kind: 'multipleChoice',
      title: 'Construct and Reshape Arrays',
      prompt:
        'Trace the three array definitions in the reference table, then determine their shapes and one indexed value.',
      instructions:
        'Remember that a one-dimensional array does not have a row dimension. NumPy indices begin at zero.',
      datasetId: 'numpyMiniArrays',
      parts: [
        {
          id: 'y-shape',
          prompt: 'What is `y.shape`?',
          correctOptionId: 'shape-12',
          options: [
            {
              id: 'shape-12',
              title: '(12,)',
              description: 'A one-dimensional NumPy array records one axis in its shape.',
            },
            {
              id: 'shape-1-12',
              title: '(1, 12)',
              description: 'This would be a two-dimensional row array.',
            },
            {
              id: 'shape-12-1',
              title: '(12, 1)',
              description: 'This would be a two-dimensional column array.',
            },
          ],
        },
        {
          id: 'Y-shape',
          prompt: 'What is `Y.shape`?',
          correctOptionId: 'shape-3-4',
          options: [
            {
              id: 'shape-3-4',
              title: '(3, 4)',
              description: 'The reshape call requests three rows and four columns.',
            },
            {
              id: 'shape-4-3',
              title: '(4, 3)',
              description: 'This reverses the two dimensions passed to reshape.',
            },
            {
              id: 'shape-12',
              title: '(12,)',
              description: 'This is the shape before reshaping.',
            },
          ],
        },
        {
          id: 'indexed-value',
          prompt: 'What is `Y[1, 2]`?',
          correctOptionId: 'value-7',
          options: [
            {
              id: 'value-6',
              title: '6',
              description: 'This is one column to the left of the requested entry.',
            },
            {
              id: 'value-7',
              title: '7',
              description: 'Row 1 is the second row, and column 2 is the third column.',
            },
            {
              id: 'value-8',
              title: '8',
              description: 'This is the last entry in the second row.',
            },
          ],
        },
        {
          id: 'newaxis-shape',
          prompt: 'What is `(y[:, np.newaxis]).shape`?',
          correctOptionId: 'shape-12-1',
          options: [
            {
              id: 'shape-12',
              title: '(12,)',
              description: '`np.newaxis` adds a dimension, so the shape cannot remain one-dimensional.',
            },
            {
              id: 'shape-1-12',
              title: '(1, 12)',
              description: 'This would add the new axis before the existing one.',
            },
            {
              id: 'shape-12-1',
              title: '(12, 1)',
              description: 'The new axis is inserted after the existing array index.',
            },
          ],
        },
      ],
      hintSchedule: [2, 4],
      hints: [
        '`np.arange(1, 13)` includes 1 but stops before 13, so it creates twelve values.',
        'Write Y as three rows of four values before applying the two zero-based indices.',
      ],
      validator: validateMultipleChoiceSelections(
        {
          'y-shape': 'shape-12',
          'Y-shape': 'shape-3-4',
          'indexed-value': 'value-7',
          'newaxis-shape': 'shape-12-1',
        },
        'At least one shape or indexed value does not follow the displayed NumPy operations.',
      ),
      reveal: {
        explanation:
          '`y` is a one-dimensional array with shape `(12,)`. Reshaping places the values into three rows of four, so `Y[1, 2]` is 7. Inserting a new axis after the existing index changes the shape to `(12, 1)`.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies MultipleChoiceQuestionSpec,
    {
      id: 'numpy-slicing-masks',
      kind: 'multipleChoice',
      title: 'Slice and Filter an Array',
      prompt:
        'Use the same array Y to trace a rectangular slice and a Boolean mask.',
      instructions:
        'Slice endpoints are excluded. In the mask, `&` requires every condition to be true for an entry to remain.',
      datasetId: 'numpyMiniArrays',
      parts: [
        {
          id: 'rectangular-slice',
          prompt: 'What is `Y[0:2, 1:3].tolist()`?',
          correctOptionId: 'slice-2-3-6-7',
          options: [
            {
              id: 'slice-2-3-6-7',
              title: '[[2, 3], [6, 7]]',
              description: 'The slice keeps rows 0 and 1 and columns 1 and 2.',
            },
            {
              id: 'slice-1-2-5-6',
              title: '[[1, 2], [5, 6]]',
              description: 'This starts the column slice at column 0.',
            },
            {
              id: 'slice-2-3-4-6-7-8',
              title: '[[2, 3, 4], [6, 7, 8]]',
              description: 'This incorrectly includes the ending column index 3.',
            },
          ],
        },
        {
          id: 'boolean-filter',
          prompt: 'What is `Y[(Y % 3 == 0) & (Y > 3)].tolist()`?',
          correctOptionId: 'filter-6-9-12',
          options: [
            {
              id: 'filter-6-9-12',
              title: '[6, 9, 12]',
              description: 'These entries satisfy both the divisibility and strict-bound conditions.',
            },
            {
              id: 'filter-3-6-9-12',
              title: '[3, 6, 9, 12]',
              description: 'The strict comparison excludes 3 itself.',
            },
            {
              id: 'filter-4-through-12',
              title: '[4, 5, 6, 7, 8, 9, 10, 11, 12]',
              description: 'This applies only the greater-than condition.',
            },
          ],
        },
      ],
      hintSchedule: [2, 4],
      hints: [
        'The slice `1:3` includes positions 1 and 2, not position 3.',
        'Evaluate each Boolean condition separately, then keep only entries where both masks are true.',
      ],
      validator: validateMultipleChoiceSelections(
        {
          'rectangular-slice': 'slice-2-3-6-7',
          'boolean-filter': 'filter-6-9-12',
        },
        'At least one result includes an entry outside the requested slice or Boolean mask.',
      ),
      reveal: {
        explanation:
          'The rectangular slice selects two rows and two columns, producing `[[2, 3], [6, 7]]`. Boolean indexing flattens the selected entries into a one-dimensional result and retains 6, 9, and 12.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies MultipleChoiceQuestionSpec,
    {
      id: 'numpy-elementwise-reductions',
      kind: 'multipleChoice',
      title: 'Apply Functions and Reduce Axes',
      prompt:
        'Let `Z = np.sqrt(Y) + X`. Trace one elementwise value and two reductions of Y.',
      instructions:
        '`axis=0` reduces down the rows and leaves one result per column. `axis=1` reduces across the columns and leaves one result per row.',
      datasetId: 'numpyMiniArrays',
      parts: [
        {
          id: 'elementwise-value',
          prompt: 'What is `Z[1, 2]`?',
          correctOptionId: 'one-plus-sqrt-7',
          options: [
            {
              id: 'one-plus-sqrt-7',
              title: '1 + √7 ≈ 3.646',
              description: 'The operations are applied to the corresponding entry 7 in Y and entry 1 in X.',
            },
            {
              id: 'sqrt-8',
              title: '√8 ≈ 2.828',
              description: 'NumPy does not add the arrays before applying the square root here.',
            },
            {
              id: 'one-plus-sqrt-6',
              title: '1 + √6 ≈ 3.449',
              description: 'This uses the entry one column to the left.',
            },
          ],
        },
        {
          id: 'column-maxima',
          prompt: 'What is `Y.max(axis=0).tolist()`?',
          correctOptionId: 'max-9-10-11-12',
          options: [
            {
              id: 'max-9-10-11-12',
              title: '[9, 10, 11, 12]',
              description: 'Reducing axis 0 leaves the maximum of each column.',
            },
            {
              id: 'max-4-8-12',
              title: '[4, 8, 12]',
              description: 'These are the row maxima produced by reducing axis 1.',
            },
            {
              id: 'max-12',
              title: '[12]',
              description: 'This reduces the whole array rather than a specified axis.',
            },
          ],
        },
        {
          id: 'row-sums',
          prompt: 'What is `Y.sum(axis=1).tolist()`?',
          correctOptionId: 'sums-10-26-42',
          options: [
            {
              id: 'sums-10-26-42',
              title: '[10, 26, 42]',
              description: 'Reducing axis 1 produces one sum for each row.',
            },
            {
              id: 'sums-15-18-21-24',
              title: '[15, 18, 21, 24]',
              description: 'These are column sums produced by reducing axis 0.',
            },
            {
              id: 'sum-78',
              title: '[78]',
              description: 'This is the sum of the entire array rather than one sum per row.',
            },
          ],
        },
      ],
      hintSchedule: [2, 4],
      hints: [
        'Elementwise operations do not change which row and column an entry occupies.',
        'The dimension named by `axis` disappears from the result.',
      ],
      validator: validateMultipleChoiceSelections(
        {
          'elementwise-value': 'one-plus-sqrt-7',
          'column-maxima': 'max-9-10-11-12',
          'row-sums': 'sums-10-26-42',
        },
        'At least one elementwise value or axis reduction is incorrect.',
      ),
      reveal: {
        explanation:
          'The entry at `[1, 2]` is computed from 7 and 1, so it is `sqrt(7) + 1`. Reducing axis 0 produces four column maxima; reducing axis 1 produces three row sums.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies MultipleChoiceQuestionSpec,
    {
      id: 'numpy-matrix-product',
      kind: 'multipleChoice',
      title: 'Read a Matrix Product',
      prompt:
        'Let `P = np.matmul(X, Y.T)`. Determine the result shape and the values in its first row.',
      instructions:
        'Write the operand shapes after transposing Y. Each entry of P is a dot product between one row of X and one row of Y.',
      datasetId: 'numpyMiniArrays',
      parts: [
        {
          id: 'product-shape',
          prompt: 'What is `P.shape`?',
          correctOptionId: 'shape-3-3',
          options: [
            {
              id: 'shape-3-3',
              title: '(3, 3)',
              description: 'The product keeps the outer dimensions of `(3, 4) @ (4, 3)`.',
            },
            {
              id: 'shape-3-4',
              title: '(3, 4)',
              description: 'This is the shape of X and Y before the transpose and product.',
            },
            {
              id: 'shape-4-4',
              title: '(4, 4)',
              description: 'The matching inner dimension is summed over rather than retained.',
            },
          ],
        },
        {
          id: 'first-product-row',
          prompt: 'What is `P[0].tolist()`?',
          correctOptionId: 'row-10-26-42',
          options: [
            {
              id: 'row-10-26-42',
              title: '[10.0, 26.0, 42.0]',
              description: 'A row of four ones dotted with each row of Y produces that row’s sum.',
            },
            {
              id: 'row-1-2-3-4',
              title: '[1.0, 2.0, 3.0, 4.0]',
              description: 'This copies the first row of Y instead of taking dot products.',
            },
            {
              id: 'row-15-18-21-24',
              title: '[15.0, 18.0, 21.0, 24.0]',
              description: 'These are the sums of the columns of Y.',
            },
          ],
        },
      ],
      hintSchedule: [2, 4],
      hints: [
        'After transposing Y, the multiplication is `(3, 4) @ (4, 3)`.',
        'Every row of X contains four ones, so its dot product with a row of Y is simply the sum of that Y row.',
      ],
      validator: validateMultipleChoiceSelections(
        {
          'product-shape': 'shape-3-3',
          'first-product-row': 'row-10-26-42',
        },
        'Check the transposed operand shape and compute each dot product again.',
      ),
      reveal: {
        explanation:
          '`Y.T` has shape `(4, 3)`, so multiplying it by X produces a `(3, 3)` matrix. Because each row of X contains only ones, every output row contains the three row sums of Y: 10, 26, and 42.',
      },
      successCopy: 'Correct! Good Job!',
    } satisfies MultipleChoiceQuestionSpec,
    numpyNotebookLab,
  ],
}
