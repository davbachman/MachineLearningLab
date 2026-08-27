import { describe, expect, it } from 'vitest'
import { kmeansInteractiveDatasets } from '../data/kmeansDatasets'
import {
  multipleChoiceDatasets,
  type MultipleChoiceMetricComparisonDataset,
} from '../data/multipleChoiceDatasets'
import {
  assignmentsMatchUpToPermutation,
  centroidsMatchUpToPermutation,
} from '../lib/kmeansMath'

describe('kmeansMath helpers', () => {
  it('detects the Manhattan reassignment on the lower-right points', () => {
    const dataset = multipleChoiceDatasets.metricComparison as MultipleChoiceMetricComparisonDataset

    expect(dataset.euclideanAssignments).not.toEqual(dataset.manhattanAssignments)
    expect(dataset.euclideanAssignments[6]).not.toBe(dataset.manhattanAssignments[6])
    expect(dataset.euclideanAssignments[7]).not.toBe(dataset.manhattanAssignments[7])
  })

  it('treats cluster labels as interchangeable when matching assignments', () => {
    expect(assignmentsMatchUpToPermutation([0, 0, 1, 1], [1, 1, 0, 0], 2)).toBe(true)
    expect(assignmentsMatchUpToPermutation([0, 1, 0, 1], [0, 0, 1, 1], 2)).toBe(false)
  })

  it('matches centroid sets up to permutation within tolerance', () => {
    expect(
      centroidsMatchUpToPermutation(
        [
          [4.02, 3.52],
          [-4.01, -0.48],
        ],
        [
          [-4, -0.5],
          [4, 3.5],
        ],
        0.08,
      ),
    ).toBe(true)

    expect(
      centroidsMatchUpToPermutation(
        [
          [3.2, 3.5],
          [-4.01, -0.48],
        ],
        [
          [-4, -0.5],
          [4, 3.5],
        ],
        0.08,
      ),
    ).toBe(false)
  })

  it('shows the right centroid shifting toward the outlier', () => {
    const dataset = kmeansInteractiveDatasets.outlierShift
    const comparison = dataset.comparisonCentroids ?? []
    const target = dataset.targetCentroids ?? []

    expect(target[1][0]).toBeGreaterThan(comparison[1][0])
    expect(target[1][1]).toBeGreaterThan(comparison[1][1])
  })
})
