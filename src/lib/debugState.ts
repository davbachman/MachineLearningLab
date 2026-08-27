import { useEffect } from 'react'

declare global {
  interface Window {
    __cs158DebugSnapshot?: unknown
  }
}

export function useDebugState(snapshot: unknown) {
  useEffect(() => {
    window.__cs158DebugSnapshot = snapshot
    return () => {
      window.__cs158DebugSnapshot = undefined
    }
  }, [snapshot])
}
