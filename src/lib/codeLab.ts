import type { BaseQuestionSpec, ValidationResult } from '../types'

export type CodeLabStageKind = 'executionTrace' | 'diagnoseMutation' | 'distinguishingTest'

export interface CodeLabChoiceOption {
  id: string
  label: string
  description?: string
}

export interface CodeLabAnswerField {
  id: string
  label: string
  options: CodeLabChoiceOption[]
  correctOptionId: string
}

export interface CodeLabStageSpec {
  id: string
  kind: CodeLabStageKind
  title: string
  prompt: string
  fields: CodeLabAnswerField[]
  successCopy: string
}

/**
 * Standalone until `codeLab` is added to the shared QuestionKind union.
 * Omit keeps the common question contract without forcing a premature edit to
 * src/types.ts while several feature branches are being integrated.
 */
export interface CodeLabQuestionSpec extends Omit<BaseQuestionSpec, 'kind' | 'validator'> {
  kind: 'codeLab'
  language: 'python'
  variantId: string
  code: string
  fixtureTitle: string
  fixtureHeading?: string
  fixture: string
  invocationTitle: string
  invocationLead?: string
  invocation: string
  stages: [CodeLabStageSpec, CodeLabStageSpec, CodeLabStageSpec]
  validator: (submission: unknown) => ValidationResult
}

export interface CodeLabSubmission {
  formatVersion: 1
  questionId: string
  variantId: string
  stages: Record<string, Record<string, string>>
}

export type CodeLabAnswers = Record<string, Record<string, string | undefined>>

export interface CodeLabStageValidation {
  correct: boolean
  missingFieldIds: string[]
  incorrectFieldIds: string[]
}

type CodeLabQuestionCore = Pick<CodeLabQuestionSpec, 'id' | 'variantId' | 'stages'>

export function makeCodeLabSubmission(
  question: CodeLabQuestionCore,
  answers: CodeLabAnswers,
): CodeLabSubmission {
  const stages = Object.fromEntries(
    question.stages.map((stage) => [
      stage.id,
      Object.fromEntries(
        stage.fields.flatMap((field) => {
          const answer = answers[stage.id]?.[field.id]
          return answer === undefined ? [] : [[field.id, answer]]
        }),
      ),
    ]),
  )

  return {
    formatVersion: 1,
    questionId: question.id,
    variantId: question.variantId,
    stages,
  }
}

export function getCorrectCodeLabSubmission(
  question: CodeLabQuestionCore,
): CodeLabSubmission {
  return makeCodeLabSubmission(
    question,
    Object.fromEntries(
      question.stages.map((stage) => [
        stage.id,
        Object.fromEntries(stage.fields.map((field) => [field.id, field.correctOptionId])),
      ]),
    ),
  )
}

export function validateCodeLabStage(
  stage: CodeLabStageSpec,
  answers: Record<string, string | undefined> | undefined,
): CodeLabStageValidation {
  const missingFieldIds = stage.fields
    .filter((field) => answers?.[field.id] === undefined)
    .map((field) => field.id)
  const incorrectFieldIds = stage.fields
    .filter(
      (field) =>
        answers?.[field.id] !== undefined && answers[field.id] !== field.correctOptionId,
    )
    .map((field) => field.id)

  return {
    correct: missingFieldIds.length === 0 && incorrectFieldIds.length === 0,
    missingFieldIds,
    incorrectFieldIds,
  }
}

export function validateCodeLabSubmission(
  question: CodeLabQuestionCore,
  submission: unknown,
): ValidationResult {
  if (!submission || typeof submission !== 'object') {
    return { correct: false, message: 'No structured Code Lab submission was provided.' }
  }

  const payload = submission as Partial<CodeLabSubmission>
  if (
    payload.formatVersion !== 1 ||
    payload.questionId !== question.id ||
    payload.variantId !== question.variantId ||
    !payload.stages
  ) {
    return {
      correct: false,
      message: 'The Code Lab question ID, variant ID, or format version does not match.',
    }
  }

  const failedStage = question.stages.find(
    (stage) => !validateCodeLabStage(stage, payload.stages?.[stage.id]).correct,
  )

  return failedStage
    ? {
        correct: false,
        message: `At least one answer in ${failedStage.id} is missing or incorrect.`,
      }
    : { correct: true, message: 'Every Code Lab stage is correct.' }
}

type CodeLabQuestionInput = Omit<CodeLabQuestionSpec, 'validator'>

export function defineCodeLabQuestion(input: CodeLabQuestionInput): CodeLabQuestionSpec {
  return {
    ...input,
    validator: (submission) => validateCodeLabSubmission(input, submission),
  }
}
