export type DecisionTreeLabel = 0 | 1

export interface DecisionTreeClassSpec {
  label: string
  color: string
}

export interface DecisionTreeCounts {
  negative: number
  positive: number
}

export interface DecisionTreeFeatureSpec {
  id: string
  label: string
}

export interface DecisionTreeRow {
  id: string
  features: Record<string, number>
  label: DecisionTreeLabel
}

export interface DecisionTreeSplit {
  id?: string
  featureId: string
  threshold: number
}

interface DecisionTreeBaseDataset {
  id: string
  kind:
    | 'giniWarmup'
    | 'splitScore'
    | 'bestThreshold'
    | 'bestRootSplit'
    | 'visualSplit'
    | 'visualDepthTwo'
    | 'depthTwo'
  label: string
  classes: DecisionTreeClassSpec[]
  features: DecisionTreeFeatureSpec[]
}

export interface DecisionTreeGiniWarmupDataset extends DecisionTreeBaseDataset {
  kind: 'giniWarmup'
  nodes: Array<{
    id: string
    label: string
    counts: DecisionTreeCounts
  }>
}

export interface DecisionTreeTableDataset extends DecisionTreeBaseDataset {
  kind: 'splitScore' | 'bestThreshold' | 'bestRootSplit' | 'depthTwo'
  rows: DecisionTreeRow[]
}

export interface DecisionTreeSplitScoreDataset extends DecisionTreeTableDataset {
  kind: 'splitScore'
  split: DecisionTreeSplit
}

export interface DecisionTreeBestSplitDataset extends DecisionTreeTableDataset {
  kind: 'bestThreshold' | 'bestRootSplit'
  candidateSplits: DecisionTreeSplit[]
}

export interface DecisionTreeVisualSplitDataset extends DecisionTreeBaseDataset {
  kind: 'visualSplit'
  rows: DecisionTreeRow[]
  candidateSplits: DecisionTreeSplit[]
  bounds: {
    minX: number
    maxX: number
    minY: number
    maxY: number
  }
}

export interface DecisionTreeVisualDepthTwoDataset extends DecisionTreeBaseDataset {
  kind: 'visualDepthTwo'
  rows: DecisionTreeRow[]
  rootCandidates: DecisionTreeSplit[]
  rightChildCandidates: DecisionTreeSplit[]
  bounds: {
    minX: number
    maxX: number
    minY: number
    maxY: number
  }
}

export interface DecisionTreeDepthTwoDataset extends DecisionTreeTableDataset {
  kind: 'depthTwo'
  rootCandidates: DecisionTreeSplit[]
  rightChildCandidates: DecisionTreeSplit[]
}

export type DecisionTreeDataset =
  | DecisionTreeGiniWarmupDataset
  | DecisionTreeSplitScoreDataset
  | DecisionTreeBestSplitDataset
  | DecisionTreeVisualSplitDataset
  | DecisionTreeVisualDepthTwoDataset
  | DecisionTreeDepthTwoDataset

export const defaultDecisionTreeClasses: DecisionTreeClassSpec[] = [
  { label: 'Did not pass', color: '#b95f43' },
  { label: 'Passed', color: '#1c7b8a' },
]

const hoursFeature: DecisionTreeFeatureSpec = {
  id: 'hours',
  label: 'Study hours',
}

const attendanceFeature: DecisionTreeFeatureSpec = {
  id: 'attendance',
  label: 'Attendance',
}

const quizFeature: DecisionTreeFeatureSpec = {
  id: 'quiz',
  label: 'Practice quiz',
}

export const decisionTreeDatasets: Record<string, DecisionTreeDataset> = {
  giniWarmup: {
    id: 'giniWarmup',
    kind: 'giniWarmup',
    label: 'Compare impurity levels from class counts',
    classes: defaultDecisionTreeClasses,
    features: [],
    nodes: [
      { id: 'a', label: 'Node A', counts: { negative: 4, positive: 0 } },
      { id: 'b', label: 'Node B', counts: { negative: 3, positive: 1 } },
      { id: 'c', label: 'Node C', counts: { negative: 2, positive: 2 } },
      { id: 'd', label: 'Node D', counts: { negative: 1, positive: 3 } },
    ],
  },
  splitScore: {
    id: 'splitScore',
    kind: 'splitScore',
    label: 'Score one proposed threshold split',
    classes: defaultDecisionTreeClasses,
    features: [hoursFeature],
    split: {
      featureId: 'hours',
      threshold: 4.5,
    },
    rows: [
      { id: 'A', features: { hours: 1 }, label: 0 },
      { id: 'B', features: { hours: 2 }, label: 0 },
      { id: 'C', features: { hours: 3 }, label: 1 },
      { id: 'D', features: { hours: 4 }, label: 0 },
      { id: 'E', features: { hours: 6 }, label: 1 },
      { id: 'F', features: { hours: 7 }, label: 1 },
      { id: 'G', features: { hours: 8 }, label: 1 },
      { id: 'H', features: { hours: 9 }, label: 0 },
    ],
  },
  bestThreshold: {
    id: 'bestThreshold',
    kind: 'bestThreshold',
    label: 'Choose the best threshold for one feature',
    classes: defaultDecisionTreeClasses,
    features: [hoursFeature],
    candidateSplits: [
      { id: 'hours-2-5', featureId: 'hours', threshold: 2.5 },
      { id: 'hours-3-5', featureId: 'hours', threshold: 3.5 },
      { id: 'hours-5-5', featureId: 'hours', threshold: 5.5 },
      { id: 'hours-6-5', featureId: 'hours', threshold: 6.5 },
    ],
    rows: [
      { id: 'A', features: { hours: 1 }, label: 0 },
      { id: 'B', features: { hours: 2 }, label: 0 },
      { id: 'C', features: { hours: 3 }, label: 0 },
      { id: 'D', features: { hours: 4 }, label: 1 },
      { id: 'E', features: { hours: 5 }, label: 1 },
      { id: 'F', features: { hours: 6 }, label: 1 },
      { id: 'G', features: { hours: 7 }, label: 0 },
      { id: 'H', features: { hours: 8 }, label: 1 },
    ],
  },
  bestRootSplit: {
    id: 'bestRootSplit',
    kind: 'bestRootSplit',
    label: 'Compare root splits across two features',
    classes: defaultDecisionTreeClasses,
    features: [hoursFeature, attendanceFeature],
    candidateSplits: [
      { id: 'hours-5-5', featureId: 'hours', threshold: 5.5 },
      { id: 'hours-7-5', featureId: 'hours', threshold: 7.5 },
      { id: 'hours-9-5', featureId: 'hours', threshold: 9.5 },
      { id: 'attendance-77', featureId: 'attendance', threshold: 77 },
      { id: 'attendance-81', featureId: 'attendance', threshold: 81 },
      { id: 'attendance-85', featureId: 'attendance', threshold: 85 },
    ],
    rows: [
      { id: 'A', features: { hours: 1, attendance: 86 }, label: 1 },
      { id: 'B', features: { hours: 2, attendance: 72 }, label: 1 },
      { id: 'C', features: { hours: 3, attendance: 65 }, label: 0 },
      { id: 'D', features: { hours: 4, attendance: 90 }, label: 1 },
      { id: 'E', features: { hours: 5, attendance: 78 }, label: 0 },
      { id: 'F', features: { hours: 6, attendance: 84 }, label: 1 },
      { id: 'G', features: { hours: 7, attendance: 69 }, label: 0 },
      { id: 'H', features: { hours: 8, attendance: 88 }, label: 1 },
      { id: 'I', features: { hours: 9, attendance: 76 }, label: 0 },
      { id: 'J', features: { hours: 10, attendance: 82 }, label: 0 },
      { id: 'K', features: { hours: 11, attendance: 74 }, label: 0 },
      { id: 'L', features: { hours: 12, attendance: 92 }, label: 0 },
    ],
  },
  visualSplit: {
    id: 'visualSplit',
    kind: 'visualSplit',
    label: 'Place one axis-aligned split on a scatterplot',
    classes: defaultDecisionTreeClasses,
    features: [
      { id: 'x', label: 'Feature x' },
      { id: 'y', label: 'Feature y' },
    ],
    candidateSplits: [
      { id: 'x-1-5', featureId: 'x', threshold: 1.5 },
      { id: 'x-2-5', featureId: 'x', threshold: 2.5 },
      { id: 'x-3-5', featureId: 'x', threshold: 3.5 },
      { id: 'x-4-5', featureId: 'x', threshold: 4.5 },
      { id: 'x-5-5', featureId: 'x', threshold: 5.5 },
      { id: 'x-6-5', featureId: 'x', threshold: 6.5 },
      { id: 'x-7-5', featureId: 'x', threshold: 7.5 },
      { id: 'y-1-5', featureId: 'y', threshold: 1.5 },
      { id: 'y-2-5', featureId: 'y', threshold: 2.5 },
      { id: 'y-3-5', featureId: 'y', threshold: 3.5 },
      { id: 'y-4-5', featureId: 'y', threshold: 4.5 },
      { id: 'y-5-5', featureId: 'y', threshold: 5.5 },
      { id: 'y-6-5', featureId: 'y', threshold: 6.5 },
    ],
    rows: [
      { id: 'A', features: { x: 1, y: 2 }, label: 0 },
      { id: 'B', features: { x: 2, y: 1 }, label: 0 },
      { id: 'C', features: { x: 2, y: 5 }, label: 0 },
      { id: 'D', features: { x: 3, y: 3 }, label: 0 },
      { id: 'E', features: { x: 4, y: 2 }, label: 0 },
      { id: 'F', features: { x: 6, y: 1 }, label: 0 },
      { id: 'G', features: { x: 3, y: 6 }, label: 1 },
      { id: 'H', features: { x: 4, y: 5 }, label: 1 },
      { id: 'I', features: { x: 5, y: 3 }, label: 1 },
      { id: 'J', features: { x: 6, y: 4 }, label: 1 },
      { id: 'K', features: { x: 7, y: 2 }, label: 1 },
      { id: 'L', features: { x: 7, y: 6 }, label: 1 },
      { id: 'M', features: { x: 8, y: 4 }, label: 1 },
      { id: 'N', features: { x: 8, y: 7 }, label: 1 },
    ],
    bounds: {
      minX: 0,
      maxX: 9,
      minY: 0,
      maxY: 8,
    },
  },
  visualDepthTwo: {
    id: 'visualDepthTwo',
    kind: 'visualDepthTwo',
    label: 'Build two greedy splits from a scatterplot',
    classes: defaultDecisionTreeClasses,
    features: [
      { id: 'x', label: 'Feature x' },
      { id: 'y', label: 'Feature y' },
    ],
    rootCandidates: [
      { id: 'x-2-5', featureId: 'x', threshold: 2.5 },
      { id: 'x-3-5', featureId: 'x', threshold: 3.5 },
      { id: 'x-4-5', featureId: 'x', threshold: 4.5 },
      { id: 'x-5-5', featureId: 'x', threshold: 5.5 },
    ],
    rightChildCandidates: [
      { id: 'y-2-5', featureId: 'y', threshold: 2.5 },
      { id: 'y-3-5', featureId: 'y', threshold: 3.5 },
      { id: 'y-4-5', featureId: 'y', threshold: 4.5 },
      { id: 'x-6-5', featureId: 'x', threshold: 6.5 },
    ],
    rows: [
      { id: 'A', features: { x: 1, y: 2 }, label: 0 },
      { id: 'B', features: { x: 2, y: 5 }, label: 0 },
      { id: 'C', features: { x: 3, y: 3 }, label: 0 },
      { id: 'D', features: { x: 4, y: 6 }, label: 0 },
      { id: 'E', features: { x: 6, y: 2 }, label: 0 },
      { id: 'F', features: { x: 7, y: 3 }, label: 0 },
      { id: 'G', features: { x: 2, y: 1 }, label: 1 },
      { id: 'H', features: { x: 4, y: 2 }, label: 1 },
      { id: 'I', features: { x: 5, y: 1 }, label: 1 },
      { id: 'J', features: { x: 5, y: 4 }, label: 1 },
      { id: 'K', features: { x: 6, y: 5 }, label: 1 },
      { id: 'L', features: { x: 7, y: 6 }, label: 1 },
      { id: 'M', features: { x: 8, y: 4 }, label: 1 },
      { id: 'N', features: { x: 8, y: 7 }, label: 1 },
    ],
    bounds: {
      minX: 0,
      maxX: 9,
      minY: 0,
      maxY: 8,
    },
  },
  depthTwo: {
    id: 'depthTwo',
    kind: 'depthTwo',
    label: 'Complete the second split of a depth-2 tree',
    classes: defaultDecisionTreeClasses,
    features: [hoursFeature, quizFeature],
    rootCandidates: [
      { id: 'hours-3-5', featureId: 'hours', threshold: 3.5 },
      { id: 'hours-5-5', featureId: 'hours', threshold: 5.5 },
      { id: 'hours-6-5', featureId: 'hours', threshold: 6.5 },
    ],
    rightChildCandidates: [
      { id: 'quiz-5', featureId: 'quiz', threshold: 5 },
      { id: 'quiz-6', featureId: 'quiz', threshold: 6 },
      { id: 'hours-6-5', featureId: 'hours', threshold: 6.5 },
    ],
    rows: [
      { id: 'A', features: { hours: 1, quiz: 2 }, label: 0 },
      { id: 'B', features: { hours: 2, quiz: 3 }, label: 0 },
      { id: 'C', features: { hours: 3, quiz: 2 }, label: 0 },
      { id: 'D', features: { hours: 4, quiz: 8 }, label: 1 },
      { id: 'E', features: { hours: 5, quiz: 4 }, label: 0 },
      { id: 'F', features: { hours: 6, quiz: 7 }, label: 1 },
      { id: 'G', features: { hours: 7, quiz: 6 }, label: 0 },
      { id: 'H', features: { hours: 8, quiz: 9 }, label: 1 },
    ],
  },
}
