import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'node',
    include: [
      'gradescope/src/grader.test.ts',
      'gradescope/src/grader.bundle.integration.ts',
    ],
  },
})
