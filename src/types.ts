import type { CodeLabQuestionSpec } from './lib/codeLab'

export type Vec2 = [number, number]
export type Vec3 = [number, number, number]
export type DistanceMetric = 'euclidean' | 'manhattan'
export interface LabeledPoint2D {
  point: Vec2
  label: number
}

export type QuestionKind =
  | 'pca2dLine'
  | 'pca3dPlane'
  | 'multipleChoice'
  | 'tableEntry'
  | 'kmeans2d'
  | 'knn2d'
  | 'decisionTree'
  | 'randomForest'
  | 'codeLab'
export type QuestionStatus = 'locked' | 'active' | 'correct' | 'gave_up'
export type AttemptOutcome = 'incorrect' | 'progress' | 'correct'

export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue }

export interface ValidationResult {
  correct: boolean
  message?: string
}

export interface RevealSpec {
  explanation: string
}

export interface BaseQuestionSpec<K extends QuestionKind = QuestionKind> {
  id: string
  kind: K
  title: string
  prompt: string
  instructions?: string
  datasetId?: string
  tolerance?: number
  hintSchedule?: number[]
  hints: string[]
  validator: (submission: unknown) => ValidationResult
  reveal: RevealSpec
  successCopy?: string
}

export interface Pca2DLineQuestionSpec extends BaseQuestionSpec<'pca2dLine'> {
  datasetId: string
  minimumVarianceRatio: number
  initialAngleDeg: number
}

export interface Pca3DPlaneQuestionSpec extends BaseQuestionSpec<'pca3dPlane'> {
  datasetId: string
  minimumVarianceRatio: number
  initialAzimuthDeg: number
  initialElevationDeg: number
  initialRollDeg: number
}

export type KmeansInteractionMode =
  | 'placeCentroids'
  | 'assignPoints'
  | 'updateCentroids'
  | 'lloydIteration'
  | 'metricComparison'

export type KnnInteractionMode =
  | 'predictSequence'
  | 'decisionBoundary'
  | 'bestK'
  | 'scalingTrap'
  | 'metricComparison'
  | 'adversarialPlacement'

export type DecisionTreeInteractionMode =
  | 'giniWarmup'
  | 'splitScore'
  | 'bestThreshold'
  | 'bestRootSplit'
  | 'visualSplit'
  | 'depthTwo'

export type RandomForestInteractionMode =
  | 'bootstrapAudit'
  | 'featureSubsamplingGeometry'
  | 'forestVoteGeometry'
  | 'oobEstimate'
  | 'varianceReduction'
  | 'codeTrace'

export interface MultipleChoiceOption {
  id: string
  title: string
  description: string
}

export interface MultipleChoicePart {
  id: string
  prompt: string
  options: MultipleChoiceOption[]
  correctOptionId: string
}

export interface MultipleChoiceQuestionSpec extends BaseQuestionSpec<'multipleChoice'> {
  datasetId: string
  parts: MultipleChoicePart[]
}

export interface Kmeans2DQuestionSpec extends BaseQuestionSpec<'kmeans2d'> {
  datasetId: string
  interactionMode: KmeansInteractionMode
  centroidTolerance: number
}

export interface Knn2DQuestionSpec extends BaseQuestionSpec<'knn2d'> {
  datasetId: string
  interactionMode: KnnInteractionMode
}

export interface DecisionTreeQuestionSpec extends BaseQuestionSpec<'decisionTree'> {
  datasetId: string
  interactionMode: DecisionTreeInteractionMode
}

export interface RandomForestQuestionSpec extends BaseQuestionSpec<'randomForest'> {
  datasetId: string
  interactionMode: RandomForestInteractionMode
}

export interface TableEntryQuestionSpec extends BaseQuestionSpec<'tableEntry'> {
  datasetId: string
  substeps: string[]
}

export type QuestionSpec =
  | Pca2DLineQuestionSpec
  | Pca3DPlaneQuestionSpec
  | MultipleChoiceQuestionSpec
  | Kmeans2DQuestionSpec
  | Knn2DQuestionSpec
  | DecisionTreeQuestionSpec
  | RandomForestQuestionSpec
  | TableEntryQuestionSpec
  | CodeLabQuestionSpec

export interface AssignmentSpec {
  id: string
  displayNumber?: number
  version: number
  title: string
  topic: string
  description: string
  published: boolean
  questions: QuestionSpec[]
}

export interface QuestionState {
  status: QuestionStatus
  attempts: number
  incorrectAttempts: number
  hintsShown: number
  resolvedAt: string | null
  latestAnswer: JsonValue
  attemptHistory: RecordedQuestionAttempt[]
}

export interface RecordedQuestionAttempt {
  attemptNumber: number
  outcome: AttemptOutcome
  submittedAt: string
  answer: JsonValue
}

export interface AssignmentState {
  assignmentId: string
  assignmentVersion: number
  version: number
  questionStates: QuestionState[]
  exportedAt: string | null
}

export interface SubmissionQuestionExport {
  id: string
  status: 'correct' | 'gave_up'
  attempts: number
  hintsShown: number
  latestAnswer: JsonValue
  attemptHistory: RecordedQuestionAttempt[]
}

export interface SubmissionExport {
  formatVersion: number
  assignmentId: string
  assignmentVersion: number
  generatedAt: string
  questions: SubmissionQuestionExport[]
}
