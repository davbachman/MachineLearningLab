export type RandomForestLabel = 0 | 1
export type RandomForestFeature = 'x' | 'y'

export interface RandomForestClassSpec {
  label: string
  color: string
}

export interface RandomForestPoint {
  id: string
  x: number
  y: number
  label: RandomForestLabel
}

export interface RandomForestSplitCandidate {
  id: string
  feature: RandomForestFeature
  threshold: number
}

export interface RandomForestStump extends RandomForestSplitCandidate {
  leftLabel: RandomForestLabel
  rightLabel: RandomForestLabel
}

interface RandomForestBaseDataset {
  id: string
  kind:
    | 'bootstrapAudit'
    | 'featureSubsamplingGeometry'
    | 'forestVoteGeometry'
    | 'oobEstimate'
    | 'varianceReduction'
    | 'codeTrace'
  label: string
  classes: RandomForestClassSpec[]
}

export interface RandomForestBootstrapDataset extends RandomForestBaseDataset {
  kind: 'bootstrapAudit'
  rows: RandomForestPoint[]
  draws: string[]
}

export interface FeatureSubsetTreeSpec {
  id: string
  label: string
  allowedFeature: RandomForestFeature
  draws: string[]
  candidates: RandomForestSplitCandidate[]
  color: string
}

export interface RandomForestFeatureGeometryDataset extends RandomForestBaseDataset {
  kind: 'featureSubsamplingGeometry'
  rows: RandomForestPoint[]
  trees: FeatureSubsetTreeSpec[]
  probes: Array<Pick<RandomForestPoint, 'id' | 'x' | 'y'>>
  bounds: RandomForestPlotBounds
}

export interface RandomForestPlotBounds {
  minX: number
  maxX: number
  minY: number
  maxY: number
}

export interface RandomForestVoteGeometryDataset extends RandomForestBaseDataset {
  kind: 'forestVoteGeometry'
  rows: RandomForestPoint[]
  trees: RandomForestStump[]
  probes: Array<Pick<RandomForestPoint, 'id' | 'x' | 'y'>>
  bounds: RandomForestPlotBounds
}

export interface RandomForestOobDataset extends RandomForestBaseDataset {
  kind: 'oobEstimate'
  rows: RandomForestPoint[]
  trees: Array<{
    id: string
    predictions: Record<string, RandomForestLabel>
  }>
}

export interface RandomForestVarianceDataset extends RandomForestBaseDataset {
  kind: 'varianceReduction'
  singleTreeVariance: number
  scenarios: Array<{
    id: string
    label: string
    treeCount: number
    correlation: number
  }>
  interventions: Array<{
    id: string
    label: string
    description: string
  }>
}

export interface RandomForestCodeTraceDataset extends RandomForestBaseDataset {
  kind: 'codeTrace'
  statements: Array<{
    id: string
    code: string
  }>
  rows: Array<{
    id: string
    features: [number, number]
    label: RandomForestLabel
  }>
  bootstrapDraws: string[]
  selectedFeatureIndices: number[]
  query: [number, number]
  treeVotes: RandomForestLabel[]
  statementOptions: Array<{
    id: string
    label: string
  }>
}

export type RandomForestDataset =
  | RandomForestBootstrapDataset
  | RandomForestFeatureGeometryDataset
  | RandomForestVoteGeometryDataset
  | RandomForestOobDataset
  | RandomForestVarianceDataset
  | RandomForestCodeTraceDataset

export const randomForestClasses: RandomForestClassSpec[] = [
  { label: 'Class 0', color: '#b95f43' },
  { label: 'Class 1', color: '#1c7b8a' },
]

const forestRows: RandomForestPoint[] = [
  { id: 'A', x: 1, y: 1, label: 0 },
  { id: 'B', x: 2, y: 3, label: 0 },
  { id: 'C', x: 3, y: 4, label: 0 },
  { id: 'D', x: 4, y: 2, label: 1 },
  { id: 'E', x: 5, y: 7, label: 0 },
  { id: 'F', x: 6, y: 5, label: 1 },
  { id: 'G', x: 7, y: 6, label: 1 },
  { id: 'H', x: 8, y: 8, label: 1 },
]

const plotBounds: RandomForestPlotBounds = {
  minX: 0,
  maxX: 9,
  minY: 0,
  maxY: 9,
}

export const randomForestDatasets: Record<string, RandomForestDataset> = {
  bootstrapAudit: {
    id: 'bootstrapAudit',
    kind: 'bootstrapAudit',
    label: 'One bootstrap sample of eight training rows',
    classes: randomForestClasses,
    rows: forestRows,
    draws: ['C', 'A', 'C', 'H', 'F', 'C', 'A', 'D'],
  },
  featureSubsamplingGeometry: {
    id: 'featureSubsamplingGeometry',
    kind: 'featureSubsamplingGeometry',
    label: 'Two trees, two random feature subsets',
    classes: randomForestClasses,
    rows: forestRows,
    trees: [
      {
        id: 'tree-a',
        label: 'Tree A (x only)',
        allowedFeature: 'x',
        draws: ['A', 'B', 'B', 'C', 'D', 'E', 'F', 'G'],
        color: '#8f5f32',
        candidates: [
          { id: 'tree-a-x-2.5', feature: 'x', threshold: 2.5 },
          { id: 'tree-a-x-3.5', feature: 'x', threshold: 3.5 },
          { id: 'tree-a-x-4.5', feature: 'x', threshold: 4.5 },
          { id: 'tree-a-x-5.5', feature: 'x', threshold: 5.5 },
        ],
      },
      {
        id: 'tree-b',
        label: 'Tree B (y only)',
        allowedFeature: 'y',
        draws: ['A', 'C', 'D', 'E', 'F', 'F', 'G', 'H'],
        color: '#725fa3',
        candidates: [
          { id: 'tree-b-y-2.5', feature: 'y', threshold: 2.5 },
          { id: 'tree-b-y-4.5', feature: 'y', threshold: 4.5 },
          { id: 'tree-b-y-6.5', feature: 'y', threshold: 6.5 },
        ],
      },
    ],
    probes: [
      { id: 'P', x: 2.5, y: 6.5 },
      { id: 'Q', x: 6.5, y: 3 },
    ],
    bounds: plotBounds,
  },
  forestVoteGeometry: {
    id: 'forestVoteGeometry',
    kind: 'forestVoteGeometry',
    label: 'Aggregate five overlapping stump predictions',
    classes: randomForestClasses,
    rows: forestRows,
    trees: [
      { id: 'T1', feature: 'x', threshold: 3.5, leftLabel: 0, rightLabel: 1 },
      { id: 'T2', feature: 'y', threshold: 4.5, leftLabel: 0, rightLabel: 1 },
      { id: 'T3', feature: 'x', threshold: 5.5, leftLabel: 0, rightLabel: 1 },
      { id: 'T4', feature: 'y', threshold: 2.5, leftLabel: 1, rightLabel: 0 },
      { id: 'T5', feature: 'x', threshold: 7, leftLabel: 1, rightLabel: 0 },
    ],
    probes: [
      { id: 'P', x: 2, y: 6 },
      { id: 'Q', x: 4, y: 3 },
      { id: 'R', x: 6, y: 6 },
      { id: 'S', x: 7.5, y: 2 },
      { id: 'T', x: 5, y: 5 },
    ],
    bounds: plotBounds,
  },
  oobEstimate: {
    id: 'oobEstimate',
    kind: 'oobEstimate',
    label: 'Out-of-bag votes (blank means the row trained that tree)',
    classes: randomForestClasses,
    rows: forestRows,
    trees: [
      { id: 'T1', predictions: { A: 0, C: 0, E: 1, G: 1 } },
      { id: 'T2', predictions: { B: 0, C: 1, F: 1, H: 0 } },
      { id: 'T3', predictions: { A: 0, D: 1, E: 1, H: 1 } },
      { id: 'T4', predictions: { B: 0, D: 1, F: 1, G: 1 } },
      { id: 'T5', predictions: { A: 1, C: 0, F: 0, H: 0 } },
      { id: 'T6', predictions: { B: 1, D: 0, E: 0, G: 1 } },
    ],
  },
  varianceReduction: {
    id: 'varianceReduction',
    kind: 'varianceReduction',
    label: 'How tree correlation limits variance reduction',
    classes: randomForestClasses,
    singleTreeVariance: 0.24,
    scenarios: [
      { id: 'A', label: 'Forest A', treeCount: 25, correlation: 0.08 },
      { id: 'B', label: 'Forest B', treeCount: 100, correlation: 0.45 },
      { id: 'C', label: 'Forest C', treeCount: 16, correlation: 0.02 },
    ],
    interventions: [
      {
        id: 'feature-subsampling',
        label: 'Sample a subset of features at each split',
        description: 'Trees are less likely to repeat the same dominant split.',
      },
      {
        id: 'same-bootstrap',
        label: 'Give every tree the same bootstrap sample',
        description: 'Trees see more similar training data.',
      },
      {
        id: 'same-seed',
        label: 'Reuse one random seed for every tree',
        description: 'Trees repeat more of the same random choices.',
      },
    ],
  },
  codeTrace: {
    id: 'codeTrace',
    kind: 'codeTrace',
    label: 'Trace a compact random-forest implementation',
    classes: randomForestClasses,
    statements: [
      { id: 'S1', code: 'for tree_seed in seeds:' },
      { id: 'S2', code: '    sample = rng.choice(n, size=n, replace=True)' },
      { id: 'S3', code: '    features = rng.choice(p, size=m, replace=False)' },
      { id: 'S4', code: '    tree.fit(X[sample][:, features], y[sample])' },
      { id: 'S5', code: '    vote = tree.predict(X_query[:, features])' },
      { id: 'S6', code: '    votes.append(vote)' },
      { id: 'S7', code: 'return majority_vote(votes, axis=0)' },
    ],
    rows: [
      { id: 'A', features: [1, 9], label: 0 },
      { id: 'B', features: [2, 4], label: 0 },
      { id: 'C', features: [8, 6], label: 1 },
      { id: 'D', features: [9, 2], label: 1 },
    ],
    bootstrapDraws: ['C', 'A', 'C', 'D'],
    selectedFeatureIndices: [1],
    query: [7, 5],
    treeVotes: [1, 0, 1, 1, 0],
    statementOptions: [
      { id: 'S2', label: 'S2' },
      { id: 'S3', label: 'S3' },
      { id: 'S4', label: 'S4' },
      { id: 'S5', label: 'S5' },
      { id: 'S7', label: 'S7' },
    ],
  },
}
