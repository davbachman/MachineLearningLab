import { describe, expect, it } from 'vitest'
import { decisionTreeDatasets } from '../data/decisionTreeDatasets'
import {
  evaluateSplit,
  findBestSplit,
  formatGini,
  giniFromCounts,
} from '../lib/decisionTreeMath'

describe('decisionTreeMath', () => {
  it('computes binary Gini impurity from class counts', () => {
    expect(giniFromCounts({ negative: 4, positive: 0 })).toBe(0)
    expect(giniFromCounts({ negative: 2, positive: 2 })).toBe(0.5)
    expect(giniFromCounts({ negative: 1, positive: 3 })).toBe(0.375)
  })

  it('scores a threshold split with weighted child impurities', () => {
    const dataset = decisionTreeDatasets.splitScore
    if (dataset.kind !== 'splitScore') {
      throw new Error('Expected the splitScore dataset.')
    }
    const result = evaluateSplit(dataset.rows, { featureId: 'hours', threshold: 4.5 })

    expect(result.left.counts).toEqual({ negative: 3, positive: 1 })
    expect(result.right.counts).toEqual({ negative: 1, positive: 3 })
    expect(result.left.gini).toBeCloseTo(0.375)
    expect(result.right.gini).toBeCloseTo(0.375)
    expect(result.weightedGini).toBeCloseTo(0.375)
    expect(formatGini(result.weightedGini)).toBe('0.375')
  })

  it('finds the lowest weighted-Gini threshold among candidates', () => {
    const dataset = decisionTreeDatasets.bestThreshold
    if (dataset.kind !== 'bestThreshold') {
      throw new Error('Expected the bestThreshold dataset.')
    }
    const best = findBestSplit(dataset.rows, dataset.candidateSplits)

    expect(best.split.featureId).toBe('hours')
    expect(best.split.threshold).toBe(3.5)
    expect(best.weightedGini).toBeCloseTo(0.2)
  })

  it('finds the second split that makes a depth-2 tree pure', () => {
    const dataset = decisionTreeDatasets.depthTwo
    if (dataset.kind !== 'depthTwo') {
      throw new Error('Expected the depthTwo dataset.')
    }
    const root = findBestSplit(dataset.rows, dataset.rootCandidates)
    const rightRows = root.right.rows
    const child = findBestSplit(rightRows, dataset.rightChildCandidates)

    expect(root.split.featureId).toBe('hours')
    expect(root.split.threshold).toBe(3.5)
    expect(child.split.featureId).toBe('quiz')
    expect(child.split.threshold).toBe(6)
    expect(child.weightedGini).toBe(0)
  })

  it('finds the non-central optimum in the revised visual split dataset', () => {
    const dataset = decisionTreeDatasets.visualSplit
    if (dataset.kind !== 'visualSplit') {
      throw new Error('Expected the visualSplit dataset.')
    }
    const best = findBestSplit(dataset.rows, dataset.candidateSplits)

    expect(best.split.id).toBe('x-2-5')
    expect(best.left.rows.map((row) => row.id)).toEqual(['A', 'B', 'C'])
    expect(best.left.gini).toBe(0)
    expect(best.weightedGini).toBeCloseTo(0.311688)
  })

  it('recomputes the visual child split using only the chosen right branch', () => {
    const dataset = decisionTreeDatasets.visualDepthTwo
    if (dataset.kind !== 'visualDepthTwo') {
      throw new Error('Expected the visualDepthTwo dataset.')
    }
    const root = findBestSplit(dataset.rows, dataset.rootCandidates)
    const child = findBestSplit(root.right.rows, dataset.rightChildCandidates)

    expect(root.split.id).toBe('x-4-5')
    expect(root.weightedGini).toBeCloseTo(0.404762)
    expect(child.split.id).toBe('y-3-5')
    expect(child.weightedGini).toBeCloseTo(1 / 6)
  })
})
