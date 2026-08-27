import { describe, expect, it } from 'vitest'
import { knnInteractiveDatasets } from '../data/knnDatasets'
import {
  applyMinMaxNormalizer,
  classifyKnnPoint,
  computeAccuracy,
  findBestK,
  fitMinMaxNormalizer,
} from '../lib/knnMath'

describe('knnMath', () => {
  it('breaks vote ties using the smaller total neighbor distance', () => {
    const prediction = classifyKnnPoint(
      [
        { point: [0.1, 0], label: 0 },
        { point: [0.2, 0], label: 0 },
        { point: [0.05, 0], label: 1 },
        { point: [0.08, 0], label: 1 },
      ],
      [0, 0],
      4,
      'euclidean',
    )

    expect(prediction.label).toBe(1)
  })

  it('applies min-max normalization feature-wise', () => {
    const normalizer = fitMinMaxNormalizer([
      [10, 2],
      [30, 6],
      [50, 10],
    ])

    expect(applyMinMaxNormalizer([30, 6], normalizer)).toEqual([0.5, 0.5])
  })

  it('finds the held-out best k for the model-selection dataset', () => {
    const dataset = knnInteractiveDatasets.bestK
    if (dataset.kind !== 'bestK') {
      throw new Error('Expected the bestK dataset.')
    }

    const bestChoice = findBestK(
      dataset.trainingPoints,
      dataset.testPoints,
      dataset.kValues,
      dataset.metric,
    )

    expect(bestChoice.k).toBe(5)
    expect(computeAccuracy(dataset.trainingPoints, dataset.testPoints, 5, dataset.metric)).toBe(1)
  })
})
