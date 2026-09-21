// Execute the Python students actually see, independently of the JS answer validators.
// Usage: node scripts/verify_advanced_app_labs.mjs /path/to/python-with-numpy-and-torch
import { spawnSync } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

process.chdir(resolve(dirname(fileURLToPath(import.meta.url)), '..'))
const server = await createServer({
  logLevel: 'error',
  server: { middlewareMode: true, watch: null, hmr: false },
})
try {
  const { advancedAssignments } = await server.ssrLoadModule('/src/data/advancedAssignments.ts')
  const labs = advancedAssignments.flatMap(a => a.questions.filter(q => q.kind === 'codeLab').map(q => ({
    id: q.id, code: q.code, fixture: q.fixture, invocation: q.invocation,
    fields: q.stages.flatMap(s => s.fields.map(f => ({
      stage: s.id, expression: f.label,
      expected: f.options.find(o => o.id === f.correctOptionId).label,
    }))),
  })))
  const python = String.raw`
import ast, contextlib, io, json, math, sys
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
sys.path.insert(0, str(Path('Homework2026').resolve()))
labs = json.load(sys.stdin)
total = 0
conceptual = []

def comparable(value):
    if isinstance(value, torch.Size):
        return tuple(value)
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

def matches(actual, expected):
    actual = comparable(actual)
    if isinstance(expected, (list, tuple)):
        return type(actual) is type(expected) and len(actual) == len(expected) and all(matches(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        return actual == expected
    if isinstance(expected, (float, int)):
        return isinstance(actual, (float, int)) and math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)
    return actual == expected

for lab in labs:
    env = {'__name__': '__main__'}
    with contextlib.redirect_stdout(io.StringIO()):
        for part in ('code', 'fixture', 'invocation'):
            exec(compile(lab[part], lab['id'] + ':' + part, 'exec'), env)
    checked = 0
    for field in lab['fields']:
        try:
            expression = ast.parse(field['expression'], mode='eval')
            expected = ast.literal_eval(field['expected'])
        except (SyntaxError, ValueError):
            conceptual.append(lab['id'] + '/' + field['stage'])
            continue
        actual = eval(compile(expression, lab['id'] + ':' + field['stage'], 'eval'), env)
        assert matches(actual, expected), (lab['id'], field['expression'], actual, expected)
        checked += 1
    total += checked
    print(f"{lab['id']}: displayed Python executes; {checked} literal-valued checkpoints match")
print(f"Verified {len(labs)} executable labs and {total} literal-valued answers.")
if conceptual:
    print('Conceptual/nonliteral choices are checked by the app tests: ' + ', '.join(conceptual))
`
  const result = spawnSync(process.argv[2] ?? 'python3', ['-c', python], {
    input: JSON.stringify(labs), encoding: 'utf8', maxBuffer: 10 * 1024 * 1024,
  })
  if (result.stdout) process.stdout.write(result.stdout)
  if (result.stderr) process.stderr.write(result.stderr)
  if (result.error) throw result.error
  process.exitCode = result.status ?? 1
} finally {
  await server.close()
}
