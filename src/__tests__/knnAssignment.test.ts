import { describe, expect, it } from 'vitest'
import { knnAssignment } from '../data/knnAssignment'
import { knnInteractiveDatasets } from '../data/knnDatasets'
import { classifyGridCells } from '../lib/knnMath'

describe('knnAssignment validators', () => {
  it('publishes both renamed notebook-based k-NN labs as assignment version 8', () => {
    expect(knnAssignment.version).toBe(8)
    expect(knnAssignment.questions.slice(-2).map((question) => question.id)).toEqual([
      'knn-code-lab',
      'knn-evaluation-notebook-lab',
    ])
    expect(knnAssignment.questions.slice(-2).map((question) => question.title)).toEqual([
      'Notebook Lab: Read the k-NN Implementation',
      'Notebook Lab: Evaluate and Select a k-NN Model',
    ])
    expect(knnAssignment.questions.at(-2)?.prompt).toContain('3KNN.ipynb')
    expect(knnAssignment.questions.at(-1)?.prompt).toContain('4KNN.ipynb')
  })

  it('checks all three k values in the classification-sequence question', () => {
    const question = knnAssignment.questions[0]

    expect(question.validator({ step: 0, predictedLabel: 0 }).correct).toBe(true)
    expect(question.validator({ step: 1, predictedLabel: 1 }).correct).toBe(true)
    expect(question.validator({ step: 2, predictedLabel: 1 }).correct).toBe(true)
    expect(question.validator({ step: 1, predictedLabel: 0 }).correct).toBe(false)
  })

  it('checks the painted decision boundary for both values of k', () => {
    const question = knnAssignment.questions[1]
    const dataset = knnInteractiveDatasets.decisionBoundary
    if (dataset.kind !== 'decisionBoundary') {
      throw new Error('Expected the decisionBoundary dataset.')
    }

    const k1Cells = classifyGridCells(
      dataset.trainingPoints,
      dataset.bounds,
      dataset.gridColumns,
      dataset.gridRows,
      dataset.kSequence[0],
      dataset.metric,
    )
    const k5Cells = classifyGridCells(
      dataset.trainingPoints,
      dataset.bounds,
      dataset.gridColumns,
      dataset.gridRows,
      dataset.kSequence[1],
      dataset.metric,
    )

    expect(question.validator({ step: 0, cells: k1Cells }).correct).toBe(true)
    expect(question.validator({ step: 1, cells: k5Cells }).correct).toBe(true)
    expect(question.validator({ step: 1, cells: k1Cells }).correct).toBe(false)
  })

  it('places each decision-boundary training point in a distinct grid cell', () => {
    const dataset = knnInteractiveDatasets.decisionBoundary
    if (dataset.kind !== 'decisionBoundary') {
      throw new Error('Expected the decisionBoundary dataset.')
    }

    const occupiedCells = new Set<string>()

    for (const entry of dataset.trainingPoints) {
      const column = Math.min(
        dataset.gridColumns - 1,
        Math.max(
          0,
          Math.floor(
            ((entry.point[0] - dataset.bounds.minX) / (dataset.bounds.maxX - dataset.bounds.minX)) *
              dataset.gridColumns,
          ),
        ),
      )
      const row = Math.min(
        dataset.gridRows - 1,
        Math.max(
          0,
          Math.floor(
            ((entry.point[1] - dataset.bounds.minY) / (dataset.bounds.maxY - dataset.bounds.minY)) *
              dataset.gridRows,
          ),
        ),
      )
      const key = `${row},${column}`

      expect(occupiedCells.has(key)).toBe(false)
      occupiedCells.add(key)
    }
  })

  it('gives each decision-boundary cell center a unique class label', () => {
    const dataset = knnInteractiveDatasets.decisionBoundary
    if (dataset.kind !== 'decisionBoundary') {
      throw new Error('Expected the decisionBoundary dataset.')
    }

    for (const k of dataset.kSequence) {
      for (let row = 0; row < dataset.gridRows; row += 1) {
        for (let column = 0; column < dataset.gridColumns; column += 1) {
          const x0 =
            dataset.bounds.minX +
            (column / dataset.gridColumns) * (dataset.bounds.maxX - dataset.bounds.minX)
          const x1 =
            dataset.bounds.minX +
            ((column + 1) / dataset.gridColumns) * (dataset.bounds.maxX - dataset.bounds.minX)
          const y0 =
            dataset.bounds.minY +
            (row / dataset.gridRows) * (dataset.bounds.maxY - dataset.bounds.minY)
          const y1 =
            dataset.bounds.minY +
            ((row + 1) / dataset.gridRows) * (dataset.bounds.maxY - dataset.bounds.minY)
          const center: [number, number] = [(x0 + x1) / 2, (y0 + y1) / 2]

          const ordered = dataset.trainingPoints
            .map((entry, index) => ({
              index,
              label: entry.label,
              distance: Math.hypot(entry.point[0] - center[0], entry.point[1] - center[1]),
            }))
            .sort((left, right) => left.distance - right.distance || left.index - right.index)

          const topNeighbors = ordered.slice(0, k)
          const labelCounts = new Map<number, { count: number; totalDistance: number }>()

          for (const neighbor of topNeighbors) {
            const current = labelCounts.get(neighbor.label) ?? { count: 0, totalDistance: 0 }
            labelCounts.set(neighbor.label, {
              count: current.count + 1,
              totalDistance: current.totalDistance + neighbor.distance,
            })
          }

          const summaries = [...labelCounts.entries()]
            .map(([label, summary]) => ({ label, ...summary }))
            .sort(
              (left, right) =>
                right.count - left.count ||
                left.totalDistance - right.totalDistance ||
                left.label - right.label,
            )

          const tiedTopPrediction =
            summaries.length > 1 &&
            summaries[0].count === summaries[1].count &&
            Math.abs(summaries[0].totalDistance - summaries[1].totalDistance) < 1e-9

          const tiedNearestOppositeClass =
            k === 1 &&
            ordered.length > 1 &&
            ordered[0].label !== ordered[1].label &&
            Math.abs(ordered[0].distance - ordered[1].distance) < 1e-9

          expect(tiedTopPrediction || tiedNearestOppositeClass).toBe(false)
        }
      }
    }
  })

  it('accepts the held-out best k and rejects a worse choice', () => {
    const question = knnAssignment.questions[2]

    expect(question.validator({ k: 5 }).correct).toBe(true)
    expect(question.validator({ k: 1 }).correct).toBe(false)
  })

  it('checks the scaling-trap predictions before and after normalization', () => {
    const question = knnAssignment.questions[3]

    expect(question.validator({ step: 'raw', predictedLabel: 1 }).correct).toBe(true)
    expect(question.validator({ step: 'normalized', predictedLabel: 0 }).correct).toBe(true)
    expect(question.validator({ step: 'normalized', predictedLabel: 1 }).correct).toBe(false)
  })

  it('checks the Euclidean and Manhattan predictions separately', () => {
    const question = knnAssignment.questions[4]

    expect(question.validator({ step: 'euclidean', predictedLabel: 0 }).correct).toBe(true)
    expect(question.validator({ step: 'manhattan', predictedLabel: 1 }).correct).toBe(true)
    expect(question.validator({ step: 'manhattan', predictedLabel: 0 }).correct).toBe(false)
  })

  it('accepts an adversarial point that flips the prediction', () => {
    const question = knnAssignment.questions[5]

    expect(question.validator({ point: [0.06, 0.12] }).correct).toBe(true)
    expect(question.validator({ point: [2.4, 2.0] }).correct).toBe(false)
  })

  it('checks the train/test accuracy summary question', () => {
    const question = knnAssignment.questions[6]

    expect(
      question.validator({
        selectedIds: {
          'train-correct': 'train-7',
          'train-accuracy': 'train-87-5',
          'test-correct': 'test-2',
          'test-accuracy': 'test-50',
          'future-performance': 'use-test',
        },
      }).correct,
    ).toBe(true)

    expect(
      question.validator({
        selectedIds: {
          'train-correct': 'train-8',
          'train-accuracy': 'train-100',
          'test-correct': 'test-3',
          'test-accuracy': 'test-75',
          'future-performance': 'use-train',
        },
      }).correct,
    ).toBe(false)
  })
})
