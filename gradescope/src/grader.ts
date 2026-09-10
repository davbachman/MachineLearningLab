import {
  existsSync,
  mkdirSync,
  readFileSync,
  statSync,
  writeFileSync,
} from 'fs'
import { basename, dirname } from 'path'
import { assignmentsById, publishedAssignments } from '../../src/data/assignments'
import { kmeansInteractiveDatasets } from '../../src/data/kmeansDatasets'
import { knnInteractiveDatasets } from '../../src/data/knnDatasets'
import type {
  AssignmentSpec,
  JsonValue,
  MultipleChoiceQuestionSpec,
  QuestionSpec,
  SubmissionQuestionExport,
} from '../../src/types'
import type { CodeLabQuestionSpec, CodeLabSubmission } from '../../src/lib/codeLab'
import { assignmentForSubmissionVersion } from './assignmentVersions'

const SUBMISSION_FORMAT_VERSION = 2
const CODE_LAB_FORMAT_VERSION = 1
const MAX_CONFIG_BYTES = 64 * 1024
const MAX_SUBMISSION_BYTES = 5 * 1024 * 1024
const MAX_JSON_NODES = 150_000
const MAX_JSON_DEPTH = 80

export interface GraderAssignmentConfig {
  assignmentId: string
  assignmentVersion?: number
}

export interface AssignmentSummary {
  id: string
  displayNumber?: number
  title: string
  version: number
  questionCount: number
}

export interface GradescopeTestResult {
  name: string
  score: number
  max_score: number
  status: 'passed' | 'failed'
  output: string
}

export interface GradescopeResults {
  score: number
  tests: GradescopeTestResult[]
  output: string
}

interface SubmissionEnvelope {
  formatVersion: number
  assignmentId: string
  assignmentVersion: number
  questions: SubmissionQuestionExport[]
}

interface PhaseSpec {
  name: string
  matches: (answer: unknown) => boolean
}

interface JsonBudget {
  nodes: number
  seen: WeakSet<object>
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

/**
 * Keep the exported pure grader safe even when it is called directly rather
 * than through the byte-capped CLI. This also rejects values that JSON cannot
 * represent, such as cycles, undefined, and non-finite numbers.
 */
function isSafeJsonValue(
  value: unknown,
  depth = 0,
  budget: JsonBudget = { nodes: 0, seen: new WeakSet<object>() },
): value is JsonValue {
  budget.nodes += 1
  if (budget.nodes > MAX_JSON_NODES || depth > MAX_JSON_DEPTH) return false

  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true
  if (typeof value === 'number') return Number.isFinite(value)
  if (typeof value !== 'object') return false
  if (budget.seen.has(value)) return false
  budget.seen.add(value)

  let safe: boolean
  if (Array.isArray(value)) {
    safe = value.every((entry) => isSafeJsonValue(entry, depth + 1, budget))
  } else {
    safe = Object.entries(value).every(
      ([, entry]) => isSafeJsonValue(entry, depth + 1, budget),
    )
  }
  budget.seen.delete(value)
  return safe
}

function normalizeConfig(config: unknown): GraderAssignmentConfig | null {
  if (typeof config === 'string') {
    return config.length > 0 ? { assignmentId: config } : null
  }
  if (!isRecord(config) || typeof config.assignmentId !== 'string') return null
  if (
    config.assignmentVersion !== undefined &&
    (!Number.isInteger(config.assignmentVersion) || Number(config.assignmentVersion) < 1)
  ) {
    return null
  }
  return {
    assignmentId: config.assignmentId,
    ...(config.assignmentVersion === undefined
      ? {}
      : { assignmentVersion: Number(config.assignmentVersion) }),
  }
}

export function getPublishedAssignmentSummaries(): AssignmentSummary[] {
  return publishedAssignments.map((assignment) => ({
    id: assignment.id,
    displayNumber: assignment.displayNumber,
    title: assignment.title,
    version: assignment.version,
    questionCount: assignment.questions.length,
  }))
}

export const listAssignments = getPublishedAssignmentSummaries

function invalidResults(message: string): GradescopeResults {
  return {
    score: 0,
    output: message,
    tests: [
      {
        name: 'Submission validation',
        score: 0,
        max_score: 100,
        status: 'failed',
        output: message,
      },
    ],
  }
}

function validateSubmissionEnvelope(
  raw: unknown,
  assignment: AssignmentSpec,
):
  | {
      envelope: SubmissionEnvelope
      questionsById: Map<string, SubmissionQuestionExport>
    }
  | { error: string } {
  if (!isSafeJsonValue(raw) || !isRecord(raw)) {
    return { error: 'The uploaded submission is not a valid, safely sized JSON object.' }
  }
  if (raw.formatVersion !== SUBMISSION_FORMAT_VERSION) {
    return {
      error: `Submission format version ${SUBMISSION_FORMAT_VERSION} is required.`,
    }
  }
  if (raw.assignmentId !== assignment.id) {
    return { error: `This autograder accepts only the ${assignment.id} assignment.` }
  }
  if (raw.assignmentVersion !== assignment.version) {
    return {
      error: `This autograder does not support the submitted assignment version (${String(raw.assignmentVersion)}). Its current version is ${assignment.version}. Contact your instructor to check the autograder package before redoing or re-exporting your work.`,
    }
  }
  if (!Array.isArray(raw.questions)) {
    return { error: 'The submission must contain a questions array.' }
  }

  const expectedIds = assignment.questions.map((question) => question.id)
  const expectedIdSet = new Set(expectedIds)
  const questionsById = new Map<string, SubmissionQuestionExport>()
  for (const rawQuestion of raw.questions) {
    if (
      !isRecord(rawQuestion) ||
      typeof rawQuestion.id !== 'string' ||
      !Object.prototype.hasOwnProperty.call(rawQuestion, 'latestAnswer') ||
      !Array.isArray(rawQuestion.attemptHistory)
    ) {
      return { error: 'Every question record must include an ID, latestAnswer, and attemptHistory.' }
    }
    if (!expectedIdSet.has(rawQuestion.id) || questionsById.has(rawQuestion.id)) {
      return {
        error: 'The submission question IDs do not exactly match this assignment and version.',
      }
    }
    questionsById.set(rawQuestion.id, rawQuestion as unknown as SubmissionQuestionExport)
  }

  if (questionsById.size !== expectedIds.length) {
    return {
      error: 'The submission question IDs do not exactly match this assignment and version.',
    }
  }

  return {
    envelope: raw as unknown as SubmissionEnvelope,
    questionsById,
  }
}

function validatorAccepts(question: QuestionSpec, answer: unknown): boolean {
  try {
    return question.validator(answer).correct === true
  } catch {
    return false
  }
}

function makeTest(
  name: string,
  earned: boolean,
  maxScore: number,
  correctOutput = 'Correct.',
): GradescopeTestResult {
  return {
    name,
    score: earned ? maxScore : 0,
    max_score: maxScore,
    status: earned ? 'passed' : 'failed',
    output: earned ? correctOutput : 'Incorrect or missing raw answer.',
  }
}

function gradeMultipleChoice(
  question: MultipleChoiceQuestionSpec,
  rawAnswer: unknown,
  questionPoints: number,
): GradescopeTestResult[] {
  const selectedIds =
    isRecord(rawAnswer) && isRecord(rawAnswer.selectedIds) ? rawAnswer.selectedIds : {}
  const unitPoints = questionPoints / question.parts.length

  return question.parts.map((part) => {
    // Fill every other part canonically, then run the assignment's actual
    // validator. This provides per-part credit without duplicating its rules.
    const candidateSelections = Object.fromEntries(
      question.parts.map((candidatePart) => [
        candidatePart.id,
        candidatePart.id === part.id
          ? selectedIds[part.id]
          : candidatePart.correctOptionId,
      ]),
    )
    const earned = validatorAccepts(question, { selectedIds: candidateSelections })
    return makeTest(`${question.title} — ${part.prompt}`, earned, unitPoints)
  })
}

function isCodeLabEnvelope(
  question: CodeLabQuestionSpec,
  rawAnswer: unknown,
): rawAnswer is CodeLabSubmission {
  return (
    isRecord(rawAnswer) &&
    rawAnswer.formatVersion === CODE_LAB_FORMAT_VERSION &&
    rawAnswer.questionId === question.id &&
    rawAnswer.variantId === question.variantId &&
    isRecord(rawAnswer.stages)
  )
}

function gradeCodeLab(
  question: CodeLabQuestionSpec,
  rawAnswer: unknown,
  questionPoints: number,
): GradescopeTestResult[] {
  const fields = question.stages.flatMap((stage) =>
    stage.fields.map((field) => ({ stage, field })),
  )
  const unitPoints = questionPoints / fields.length
  const envelopeIsValid = isCodeLabEnvelope(question, rawAnswer)

  return fields.map(({ stage, field }) => {
    const submittedStages = envelopeIsValid ? rawAnswer.stages : {}
    const submittedStage = submittedStages[stage.id]
    const submittedValue = isRecord(submittedStage) ? submittedStage[field.id] : undefined

    // As with multiple choice, isolate one raw field while satisfying all
    // others, and ask the code lab's canonical validator for the verdict.
    const stages = Object.fromEntries(
      question.stages.map((candidateStage) => [
        candidateStage.id,
        Object.fromEntries(
          candidateStage.fields.map((candidateField) => [
            candidateField.id,
            candidateStage.id === stage.id && candidateField.id === field.id
              ? submittedValue
              : candidateField.correctOptionId,
          ]),
        ),
      ]),
    )
    const candidate = {
      formatVersion: CODE_LAB_FORMAT_VERSION,
      questionId: question.id,
      variantId: question.variantId,
      stages,
    }
    const earned = envelopeIsValid && validatorAccepts(question, candidate)
    return makeTest(`${question.title} — ${stage.title}: ${field.label}`, earned, unitPoints)
  })
}

function rawAnswers(questionExport: SubmissionQuestionExport): unknown[] {
  const answers: unknown[] = []
  for (const attempt of questionExport.attemptHistory) {
    // Outcomes are deliberately ignored. A phase earns credit only when its
    // stored raw answer passes the current canonical validator.
    if (isRecord(attempt) && Object.prototype.hasOwnProperty.call(attempt, 'answer')) {
      answers.push(attempt.answer)
    }
  }
  answers.push(questionExport.latestAnswer)
  return answers
}

export function stripKmeansReferenceOverrides(answer: unknown): unknown {
  if (!isRecord(answer)) return answer
  const canonicalAnswer = { ...answer }
  delete canonicalAnswer.referenceCentroids
  delete canonicalAnswer.referenceAssignments
  return canonicalAnswer
}

function phasesForQuestion(question: QuestionSpec): PhaseSpec[] | null {
  if (question.kind === 'tableEntry') {
    return ['centeredData', 'covariance', 'direction'].map((step) => ({
      name: step,
      matches: (answer) => isRecord(answer) && answer.step === step,
    }))
  }

  if (question.kind === 'kmeans2d') {
    if (question.interactionMode === 'lloydIteration') {
      const dataset = kmeansInteractiveDatasets[question.datasetId]
      const assignmentCount = dataset?.iterationAssignments?.length ?? 0
      const centroidCount = dataset?.iterationCentroids?.length ?? 0
      if (assignmentCount === 0 || assignmentCount !== centroidCount) {
        throw new Error(`No canonical Lloyd sequence for ${question.datasetId}.`)
      }
      return Array.from({ length: assignmentCount }, (_, iteration) => iteration).flatMap(
        (iteration) =>
          (['assignments', 'centroids'] as const).map((step) => ({
            name: `iteration ${iteration + 1} ${step}`,
            matches: (answer: unknown) =>
              isRecord(answer) && answer.step === step && answer.iteration === iteration,
          })),
      )
    }
    if (question.interactionMode === 'metricComparison') {
      return ['euclidean', 'manhattan'].map((step) => ({
        name: `${step} assignments`,
        matches: (answer) => isRecord(answer) && answer.step === step,
      }))
    }
    return null
  }

  if (question.kind === 'knn2d') {
    const dataset = knnInteractiveDatasets[question.datasetId]
    if (question.interactionMode === 'predictSequence') {
      if (!dataset || dataset.kind !== 'predictSequence') {
        throw new Error(`No canonical prediction sequence for ${question.datasetId}.`)
      }
      return dataset.kSequence.map((k, step) => ({
        name: `k = ${k}`,
        matches: (answer) => isRecord(answer) && answer.step === step,
      }))
    }
    if (question.interactionMode === 'decisionBoundary') {
      if (!dataset || dataset.kind !== 'decisionBoundary') {
        throw new Error(`No canonical boundary sequence for ${question.datasetId}.`)
      }
      return dataset.kSequence.map((k, step) => ({
        name: `k = ${k} boundary`,
        matches: (answer) => isRecord(answer) && answer.step === step,
      }))
    }
    if (question.interactionMode === 'scalingTrap') {
      return ['raw', 'normalized'].map((step) => ({
        name: step === 'raw' ? 'unscaled prediction' : 'normalized prediction',
        matches: (answer) => isRecord(answer) && answer.step === step,
      }))
    }
    if (question.interactionMode === 'metricComparison') {
      return ['euclidean', 'manhattan'].map((step) => ({
        name: `${step} prediction`,
        matches: (answer) => isRecord(answer) && answer.step === step,
      }))
    }
  }

  return null
}

function gradeProgressiveQuestion(
  question: QuestionSpec,
  questionExport: SubmissionQuestionExport,
  phases: PhaseSpec[],
  questionPoints: number,
): GradescopeTestResult[] {
  const answers = rawAnswers(questionExport)
  const unitPoints = questionPoints / phases.length
  const isKmeans = question.kind === 'kmeans2d'

  return phases.map((phase) => {
    const earned = answers
      .filter(phase.matches)
      .some((answer) =>
        validatorAccepts(question, isKmeans ? stripKmeansReferenceOverrides(answer) : answer),
      )
    return makeTest(`${question.title} — ${phase.name}`, earned, unitPoints)
  })
}

function gradeQuestion(
  question: QuestionSpec,
  questionExport: SubmissionQuestionExport,
  questionPoints: number,
): GradescopeTestResult[] {
  if (question.kind === 'multipleChoice') {
    return gradeMultipleChoice(question, questionExport.latestAnswer, questionPoints)
  }
  if (question.kind === 'codeLab') {
    return gradeCodeLab(question, questionExport.latestAnswer, questionPoints)
  }

  const phases = phasesForQuestion(question)
  if (phases) {
    return gradeProgressiveQuestion(question, questionExport, phases, questionPoints)
  }

  // All remaining interactive question kinds are one-shot. In particular,
  // this intentionally does not search attempt history for a more favorable
  // answer or inspect the exported status/outcome fields.
  const rawAnswer =
    question.kind === 'kmeans2d'
      ? stripKmeansReferenceOverrides(questionExport.latestAnswer)
      : questionExport.latestAnswer
  return [
    makeTest(
      question.title,
      validatorAccepts(question, rawAnswer),
      questionPoints,
    ),
  ]
}

function normalizeHundredPointMaximum(
  tests: GradescopeTestResult[],
): GradescopeTestResult[] {
  if (tests.length === 0) return tests
  const normalized = tests.map((test) => ({ ...test }))
  const lastIndex = normalized.length - 1
  const precedingMaximum = normalized
    .slice(0, lastIndex)
    .reduce((sum, test) => sum + test.max_score, 0)
  const finalMaximum = 100 - precedingMaximum
  const finalTest = normalized[lastIndex]
  finalTest.max_score = finalMaximum
  if (finalTest.status === 'passed') finalTest.score = finalMaximum
  return normalized
}

/** Grade one app export using the published assignment's exact validators. */
export function gradeSubmission(configInput: unknown, submissionInput: unknown): GradescopeResults {
  try {
    const config = normalizeConfig(configInput)
    if (!config) return invalidResults('The autograder assignment configuration is invalid.')

    const currentAssignment = assignmentsById[config.assignmentId]
    if (!currentAssignment || !currentAssignment.published) {
      return invalidResults(`Unknown or unpublished assignment: ${config.assignmentId}.`)
    }
    if (
      config.assignmentVersion !== undefined &&
      config.assignmentVersion !== currentAssignment.version
    ) {
      return invalidResults(
        `This autograder was configured for assignment version ${config.assignmentVersion}, but the bundled assignment is version ${currentAssignment.version}.`,
      )
    }

    const assignment = assignmentForSubmissionVersion(
      currentAssignment,
      isRecord(submissionInput) ? submissionInput.assignmentVersion : undefined,
    )
    const validated = validateSubmissionEnvelope(submissionInput, assignment)
    if ('error' in validated) return invalidResults(validated.error)

    const questionPoints = 100 / assignment.questions.length
    const tests = normalizeHundredPointMaximum(
      assignment.questions.flatMap((question) =>
        gradeQuestion(
          question,
          // Completeness was established while validating the ID map above.
          validated.questionsById.get(question.id) as SubmissionQuestionExport,
          questionPoints,
        ),
      ),
    )
    const allPassed = tests.every((test) => test.status === 'passed')
    const rawScore = tests.reduce((sum, test) => sum + test.score, 0)
    const score = allPassed ? 100 : Math.max(0, Math.min(100, rawScore))

    return {
      score,
      tests,
      output: `Graded ${assignment.questions.length} questions for ${assignment.title} (version ${assignment.version}). Exported status and attempt outcomes were ignored.`,
    }
  } catch {
    return invalidResults('The submission could not be graded because its structure is malformed.')
  }
}

export const grade = gradeSubmission

function readJsonFile(path: string, maxBytes: number, label: string): unknown {
  const size = statSync(path).size
  if (size > maxBytes) {
    throw new Error(`${label} exceeds the ${Math.floor(maxBytes / 1024)} KiB size limit.`)
  }
  try {
    return JSON.parse(readFileSync(path, 'utf8')) as unknown
  } catch {
    throw new Error(`${label} is not valid JSON.`)
  }
}

interface CliPaths {
  config: string | undefined
  submission: string | undefined
  results: string
}

function parseCliPaths(args: string[]): CliPaths {
  const positionals: string[] = []
  let config: string | undefined
  let submission: string | undefined
  let results: string | undefined

  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    const next = args[index + 1]
    if (argument === '--config' || argument === '--assignment-config') {
      config = next
      index += 1
    } else if (argument === '--submission') {
      submission = next
      index += 1
    } else if (argument === '--results') {
      results = next
      index += 1
    } else if (!argument.startsWith('--')) {
      positionals.push(argument)
    }
  }

  return {
    config: config ?? positionals[0],
    submission: submission ?? positionals[1],
    results:
      results ??
      positionals[2] ??
      process.env.GRADESCOPE_RESULTS_PATH ??
      '/autograder/results/results.json',
  }
}

function writeResults(path: string, results: GradescopeResults): void {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, `${JSON.stringify(results, null, 2)}\n`, 'utf8')
}

/**
 * CLI contract:
 *   grader.cjs <assignment-config.json> <submission.json> <results.json>
 * or the equivalent --config/--submission/--results flags.
 */
export function runCli(args: string[] = process.argv.slice(2)): number {
  if (args.includes('--list-assignments')) {
    process.stdout.write(`${JSON.stringify(getPublishedAssignmentSummaries(), null, 2)}\n`)
    return 0
  }

  const paths = parseCliPaths(args)
  let results: GradescopeResults
  try {
    if (!paths.config || !paths.submission) {
      throw new Error('The assignment config and submission paths are required.')
    }
    const config = existsSync(paths.config)
      ? readJsonFile(paths.config, MAX_CONFIG_BYTES, 'Assignment configuration')
      : paths.config
    const submission = readJsonFile(paths.submission, MAX_SUBMISSION_BYTES, 'Submission')
    results = gradeSubmission(config, submission)
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Malformed autograder input.'
    results = invalidResults(message)
  }

  try {
    writeResults(paths.results, results)
    return 0
  } catch (error) {
    // A Gradescope run normally supplies a writable result path. The fallback
    // still makes local invocations useful if that mount is unavailable.
    const fallback = 'results.json'
    if (paths.results !== fallback) {
      try {
        writeResults(fallback, results)
        process.stderr.write(`Could not write ${paths.results}; wrote ${fallback} instead.\n`)
        return 0
      } catch {
        // Fall through to the only situation in which no result can be saved.
      }
    }
    const message = error instanceof Error ? error.message : 'Unknown write error.'
    process.stderr.write(`Could not write Gradescope results: ${message}\n`)
    return 1
  }
}

const directlyExecutedName = basename(process.argv[1] ?? '')
if (/^grader\.(?:c?js|mjs|ts)$/.test(directlyExecutedName)) {
  process.exitCode = runCli()
}
