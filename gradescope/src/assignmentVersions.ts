import type { AssignmentSpec } from '../../src/types'
import { kmeansAssignment, validateLloydIteration } from '../../src/data/kmeansAssignment'

// Versions 6 and 7 differ only in the first question's initial centroids and
// derived Lloyd sequence. Preserve that key instead of ignoring version checks.
export const kmeansVersion6Assignment: AssignmentSpec = {
  ...kmeansAssignment,
  version: 6,
  questions: kmeansAssignment.questions.map((question) =>
    question.kind === 'kmeans2d' && question.id === 'kmeans-two-iterations'
      ? {
          ...question,
          datasetId: 'fullIterationV6',
          validator: validateLloydIteration('fullIterationV6', question.centroidTolerance),
        }
      : question,
  ),
}

export function assignmentForSubmissionVersion(
  current: AssignmentSpec,
  version: unknown,
): AssignmentSpec {
  if (current.id === 'kmeans' && current.version === 7 && version === 6) {
    return kmeansVersion6Assignment
  }
  return current
}
