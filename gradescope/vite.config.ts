import { resolve } from 'path'
import { fileURLToPath } from 'url'
import { defineConfig } from 'vite'

const gradescopeRoot = fileURLToPath(new URL('.', import.meta.url))

export default defineConfig({
  build: {
    ssr: resolve(gradescopeRoot, 'src/grader.ts'),
    outDir: resolve(gradescopeRoot, 'build'),
    emptyOutDir: true,
    copyPublicDir: false,
    target: 'node12',
    minify: false,
    rollupOptions: {
      output: {
        format: 'cjs',
        entryFileNames: 'grader.cjs',
        codeSplitting: false,
      },
    },
  },
})
