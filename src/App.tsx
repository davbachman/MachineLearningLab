import { HashRouter, Navigate, Route, Routes, useParams } from 'react-router-dom'
import './App.css'
import { AssignmentPage } from './components/AssignmentPage'
import { HomePage } from './components/HomePage'
import { assignmentsById, publishedAssignments } from './data/assignments'

function AssignmentRoute() {
  const params = useParams()
  const assignment = params.assignmentId ? assignmentsById[params.assignmentId] : undefined

  if (!assignment || !assignment.published) {
    return <Navigate replace to="/" />
  }

  return <AssignmentPage key={assignment.id} assignment={assignment} />
}

function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<HomePage assignments={publishedAssignments} />} />
        <Route path="/assignment/:assignmentId" element={<AssignmentRoute />} />
        <Route path="*" element={<Navigate replace to="/" />} />
      </Routes>
    </HashRouter>
  )
}

export default App
