import type { AssignmentSpec } from '../types'
import { withBalancedChoiceOrder } from '../lib/choiceOrder'
import { decisionTreeAssignment } from './decisionTreeAssignment'
import { kmeansAssignment } from './kmeansAssignment'
import { knnAssignment } from './knnAssignment'
import { numpyAssignment } from './numpyAssignment'
import { pcaAssignment } from './pcaAssignment'
import { randomForestAssignment } from './randomForestAssignment'
import { notebookAssignments } from './notebookAssignments'
import { advancedAssignments } from './advancedAssignments'

export const assignmentRegistry: AssignmentSpec[] = [
  numpyAssignment,
  pcaAssignment,
  kmeansAssignment,
  knnAssignment,
  decisionTreeAssignment,
  randomForestAssignment,
  ...notebookAssignments,
  ...advancedAssignments,
].map(withBalancedChoiceOrder)

export const publishedAssignments = assignmentRegistry.filter((assignment) => assignment.published)

export const assignmentsById = Object.fromEntries(
  assignmentRegistry.map((assignment) => [assignment.id, assignment]),
) as Record<string, AssignmentSpec>
