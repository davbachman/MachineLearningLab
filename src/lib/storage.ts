import { ASSIGNMENT_STATE_VERSION, createInitialAssignmentState, hydrateAssignmentState } from './assignmentState'
import type { AssignmentSpec, AssignmentState } from '../types'

function storageKey(assignmentId: string, assignmentVersion: number) {
  return `cs158-ml-assignment-apps::${assignmentId}::content-${assignmentVersion}::state-${ASSIGNMENT_STATE_VERSION}`
}

export function loadAssignmentState(assignment: AssignmentSpec): AssignmentState {
  if (typeof window === 'undefined') {
    return createInitialAssignmentState(assignment)
  }

  const raw = window.localStorage.getItem(storageKey(assignment.id, assignment.version))
  if (!raw) {
    return createInitialAssignmentState(assignment)
  }

  try {
    const parsed = JSON.parse(raw) as AssignmentState
    return hydrateAssignmentState(assignment, parsed)
  } catch {
    return createInitialAssignmentState(assignment)
  }
}

export function saveAssignmentState(assignment: AssignmentSpec, state: AssignmentState) {
  if (typeof window === 'undefined') {
    return
  }

  window.localStorage.setItem(storageKey(assignment.id, assignment.version), JSON.stringify(state))
}

export function clearAssignmentState(assignmentId: string) {
  if (typeof window === 'undefined') {
    return
  }

  const prefix = `cs158-ml-assignment-apps::${assignmentId}::`
  const matchingKeys = Array.from({ length: window.localStorage.length }, (_, index) =>
    window.localStorage.key(index),
  ).filter((key): key is string => Boolean(key?.startsWith(prefix)))

  for (const key of matchingKeys) {
    window.localStorage.removeItem(key)
  }
}
