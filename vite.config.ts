import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/MachineLearningLab/',
  test: {
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
    css: true,
    testTimeout: 15_000,
  },
})
