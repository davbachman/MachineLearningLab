import { describe, expect, it } from 'vitest'
import {
  aggregateOobPredictions,
  bootstrapMultiplicities,
  correlatedEnsembleVariance,
  findBestStump,
  forestVotes,
  oobAccuracy,
  outOfBagIds,
  predictForest,
  predictStump,
  trainedStumpFromEvaluation,
} from '../lib/randomForestMath'
import {
  randomForestDatasets,
  type RandomForestBootstrapDataset,
  type RandomForestFeatureGeometryDataset,
  type RandomForestOobDataset,
  type RandomForestVoteGeometryDataset,
} from '../data/randomForestDatasets'

describe('randomForestMath', () => {
  it('preserves bootstrap multiplicity and identifies omitted OOB rows', () => {
    const dataset = randomForestDatasets.bootstrapAudit as RandomForestBootstrapDataset
    const rowIds = dataset.rows.map((row) => row.id)

    expect(bootstrapMultiplicities(rowIds, dataset.draws)).toEqual({
      A: 2,
      B: 0,
      C: 3,
      D: 1,
      E: 0,
      F: 1,
      G: 0,
      H: 1,
    })
    expect(outOfBagIds(rowIds, dataset.draws)).toEqual(['B', 'E', 'G'])
  })

  it('fits different best stumps when the trees see different feature subsets', () => {
    const dataset = randomForestDatasets.featureSubsamplingGeometry as RandomForestFeatureGeometryDataset
    const [treeA, treeB] = dataset.trees
    const bestA = findBestStump(dataset.rows, treeA.candidates, treeA.draws)
    const bestB = findBestStump(dataset.rows, treeB.candidates, treeB.draws)

    expect(bestA.candidate.id).toBe('tree-a-x-3.5')
    expect(bestA.weightedGini).toBeCloseTo(0.1875)
    expect(bestB.candidate.id).toBe('tree-b-y-4.5')
    expect(bestB.weightedGini).toBeCloseTo(11 / 30, 8)

    const stumpA = trainedStumpFromEvaluation(bestA)
    const stumpB = trainedStumpFromEvaluation(bestB)
    expect(dataset.probes.map((probe) => predictStump(stumpA, probe))).toEqual([0, 1])
    expect(dataset.probes.map((probe) => predictStump(stumpB, probe))).toEqual([1, 0])
  })

  it('aggregates geometric stump votes instead of using proximity', () => {
    const dataset = randomForestDatasets.forestVoteGeometry as RandomForestVoteGeometryDataset

    expect(dataset.probes.map((probe) => predictForest(dataset.trees, probe))).toEqual([0, 0, 1, 1, 1])
    expect(forestVotes(dataset.trees, dataset.probes[0])).toEqual([0, 1, 0, 0, 1])
  })

  it('uses only each row’s OOB predictions and computes a 0.75 estimate', () => {
    const dataset = randomForestDatasets.oobEstimate as RandomForestOobDataset
    const predictions = aggregateOobPredictions(
      dataset.rows.map((row) => row.id),
      dataset.trees,
    )

    expect(predictions.map((entry) => entry.votes)).toEqual([
      [0, 0, 1],
      [0, 0, 1],
      [0, 1, 0],
      [1, 1, 0],
      [1, 1, 0],
      [1, 1, 0],
      [1, 1, 1],
      [0, 1, 0],
    ])
    expect(predictions.map((entry) => entry.prediction)).toEqual([0, 0, 0, 1, 1, 1, 1, 0])
    expect(oobAccuracy(dataset.rows, dataset.trees)).toBe(0.75)
  })

  it('shows the correlation floor in the ensemble-variance approximation', () => {
    expect(correlatedEnsembleVariance(0.24, 0.08, 25)).toBeCloseTo(0.028032)
    expect(correlatedEnsembleVariance(0.24, 0.45, 100)).toBeCloseTo(0.10932)
    expect(correlatedEnsembleVariance(0.24, 0.02, 16)).toBeCloseTo(0.0195)
  })
})
