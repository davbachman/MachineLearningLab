import { describe, expect, it } from 'vitest'
import { codeLabQuestions } from '../data/codeLabQuestions'
import {
  tokenizePythonLine,
  type PythonSyntaxTokenKind,
} from '../lib/pythonSyntax'

function tokenTexts(line: string, kind: PythonSyntaxTokenKind) {
  return tokenizePythonLine(line)
    .filter((token) => token.kind === kind)
    .map((token) => token.text)
}

describe('tokenizePythonLine', () => {
  it('preserves every character in every displayed Code Lab Python block', () => {
    for (const question of Object.values(codeLabQuestions)) {
      for (const block of [question.code, question.fixture, question.invocation]) {
        for (const line of block.split('\n')) {
          const tokens = tokenizePythonLine(line)
          expect(tokens.map((token) => token.text).join('')).toBe(line)
          expect(tokens.every((token) => token.text.length > 0)).toBe(true)
        }
      }
    }
  })

  it('classifies keywords, definition names, builtins, numbers, and operators', () => {
    const functionLine = 'def score_split(rows, threshold=2.5):  # weighted score'
    const classLine = 'class FixedTree(object):'

    expect(tokenTexts(functionLine, 'keyword')).toEqual(['def'])
    expect(tokenTexts(functionLine, 'function')).toEqual(['score_split'])
    expect(tokenTexts(functionLine, 'number')).toEqual(['2.5'])
    expect(tokenTexts(functionLine, 'operator').join('')).toBe('(,=):')
    expect(tokenTexts(functionLine, 'comment')).toEqual(['# weighted score'])

    expect(tokenTexts(classLine, 'keyword')).toEqual(['class'])
    expect(tokenTexts(classLine, 'class')).toEqual(['FixedTree'])
    expect(tokenTexts(classLine, 'builtin')).toEqual(['object'])
  })

  it('recognizes strings with prefixes and escaped quotes without starting comments inside them', () => {
    const line = String.raw`message = f"say \"hi\" # text"  # actual comment`

    expect(tokenTexts(line, 'string')).toEqual([String.raw`f"say \"hi\" # text"`])
    expect(tokenTexts(line, 'comment')).toEqual(['# actual comment'])
    expect(tokenizePythonLine(line).map((token) => token.text).join('')).toBe(line)
  })

  it('classifies builtins and keywords in representative comprehension code', () => {
    const line = 'return min(set(votes), key=lambda value: sum(value for value in votes))'

    expect(tokenTexts(line, 'keyword')).toEqual(['return', 'lambda', 'for', 'in'])
    expect(tokenTexts(line, 'builtin')).toEqual(['min', 'set', 'sum'])
    expect(tokenTexts(line, 'plain')).toContain('votes')
  })

  it('recognizes common Python numeric literal forms', () => {
    const line = 'values = [0x2A, 0b101, 0o17, 1_000, .5, 2., 6.02e23, 3j]'

    expect(tokenTexts(line, 'number')).toEqual([
      '0x2A',
      '0b101',
      '0o17',
      '1_000',
      '.5',
      '2.',
      '6.02e23',
      '3j',
    ])
  })
})
