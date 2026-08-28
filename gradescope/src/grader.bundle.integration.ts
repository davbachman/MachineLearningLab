import { spawnSync } from 'child_process'
import { createRequire } from 'module'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import { join, resolve } from 'path'
import { describe, expect, it } from 'vitest'
import { publishedAssignments } from '../../src/data/assignments'
import type { GradescopeResults } from './grader'
import { createReferenceSubmission } from './referenceSubmissions'

interface BundledGrader {
  gradeSubmission: (config: unknown, submission: unknown) => GradescopeResults
}

const bundledGrader = createRequire(import.meta.url)('../build/grader.cjs') as BundledGrader
const repositoryRoot = resolve(import.meta.dirname, '../..')

describe('bundled Gradescope grader', () => {
  it('awards exactly 100 points to every canonical assignment export', () => {
    for (const assignment of publishedAssignments) {
      const results = bundledGrader.gradeSubmission(
        {
          assignmentId: assignment.id,
          assignmentVersion: assignment.version,
        },
        createReferenceSubmission(assignment),
      )

      expect(results.score, assignment.id).toBe(100)
      expect(results.tests.every((test) => test.status === 'passed'), assignment.id).toBe(true)
    }
  })

  it('grades a valid export through every extracted package runner', () => {
    for (const assignment of publishedAssignments) {
      const temporaryRoot = mkdtempSync(join(tmpdir(), `gradescope-${assignment.id}-`))
      try {
        const sourceDirectory = join(temporaryRoot, 'source')
        const submissionDirectory = join(temporaryRoot, 'submission')
        const resultsDirectory = join(temporaryRoot, 'results')
        mkdirSync(sourceDirectory)
        mkdirSync(submissionDirectory)

        const archivePath = join(repositoryRoot, 'gradescope', 'dist', `${assignment.id}.zip`)
        const extraction = spawnSync('unzip', ['-q', archivePath, '-d', sourceDirectory], {
          encoding: 'utf8',
        })
        expect(extraction.status, extraction.stderr).toBe(0)

        const submissionPath = join(
          submissionDirectory,
          `${assignment.id}-submission.json`,
        )
        writeFileSync(
          submissionPath,
          `${JSON.stringify(createReferenceSubmission(assignment), null, 2)}\n`,
          'utf8',
        )

        const execution = spawnSync('bash', [join(sourceDirectory, 'run_autograder')], {
          encoding: 'utf8',
          env: {
            ...process.env,
            AUTOGRADER_SOURCE_DIR: sourceDirectory,
            AUTOGRADER_SUBMISSION_DIR: submissionDirectory,
            AUTOGRADER_RESULTS_DIR: resultsDirectory,
            AUTOGRADER_NODE_BIN: process.execPath,
          },
        })
        expect(execution.status, execution.stderr).toBe(0)

        const resultsPath = join(resultsDirectory, 'results.json')
        expect(existsSync(resultsPath), assignment.id).toBe(true)
        const results = JSON.parse(readFileSync(resultsPath, 'utf8')) as GradescopeResults
        expect(results.score, assignment.id).toBe(100)
        expect(results.tests.every((test) => test.status === 'passed'), assignment.id).toBe(true)
      } finally {
        rmSync(temporaryRoot, { recursive: true, force: true })
      }
    }
  })
})
