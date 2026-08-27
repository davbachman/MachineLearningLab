import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

declare global {
  interface Window {
    __cs158DebugSnapshot?: unknown
    render_game_to_text?: () => string
    advanceTime?: (ms: number) => void
  }
}

window.render_game_to_text = () =>
  JSON.stringify(window.__cs158DebugSnapshot ?? { route: 'unknown' }, null, 2)

window.advanceTime = () => {}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
