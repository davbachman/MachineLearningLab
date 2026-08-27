export type PythonSyntaxTokenKind =
  | 'plain'
  | 'keyword'
  | 'builtin'
  | 'function'
  | 'class'
  | 'string'
  | 'number'
  | 'comment'
  | 'operator'

export interface PythonSyntaxToken {
  kind: PythonSyntaxTokenKind
  text: string
}

const PYTHON_KEYWORDS = new Set([
  'False',
  'None',
  'True',
  'and',
  'as',
  'assert',
  'async',
  'await',
  'break',
  'case',
  'class',
  'continue',
  'def',
  'del',
  'elif',
  'else',
  'except',
  'finally',
  'for',
  'from',
  'global',
  'if',
  'import',
  'in',
  'is',
  'lambda',
  'match',
  'nonlocal',
  'not',
  'or',
  'pass',
  'raise',
  'return',
  'try',
  'while',
  'with',
  'yield',
])

const PYTHON_BUILTINS = new Set([
  'Exception',
  'NotImplementedError',
  'RuntimeError',
  'TypeError',
  'ValueError',
  '__import__',
  'abs',
  'all',
  'any',
  'bin',
  'bool',
  'bytearray',
  'bytes',
  'callable',
  'chr',
  'classmethod',
  'compile',
  'complex',
  'delattr',
  'dict',
  'dir',
  'divmod',
  'enumerate',
  'eval',
  'exec',
  'filter',
  'float',
  'format',
  'frozenset',
  'getattr',
  'globals',
  'hasattr',
  'hash',
  'help',
  'hex',
  'id',
  'input',
  'int',
  'isinstance',
  'issubclass',
  'iter',
  'len',
  'list',
  'locals',
  'map',
  'max',
  'memoryview',
  'min',
  'next',
  'object',
  'oct',
  'open',
  'ord',
  'pow',
  'print',
  'property',
  'range',
  'repr',
  'reversed',
  'round',
  'set',
  'setattr',
  'slice',
  'sorted',
  'staticmethod',
  'str',
  'sum',
  'super',
  'tuple',
  'type',
  'vars',
  'zip',
])

const IDENTIFIER = /^[A-Za-z_]\w*/
const WHITESPACE = /^\s+/
const STRING_PREFIX = /^(?:br|rb|fr|rf|r|u|b|f)(?=['"])/i
const NUMBER = /^(?:0[xX][0-9a-fA-F](?:_?[0-9a-fA-F])*|0[bB][01](?:_?[01])*|0[oO][0-7](?:_?[0-7])*|(?:(?:\d(?:_?\d)*)?\.\d(?:_?\d)*|\d(?:_?\d)*\.)(?:[eE][+-]?\d(?:_?\d)*)?|\d(?:_?\d)*(?:[eE][+-]?\d(?:_?\d)*)?)(?:[jJ])?/

const MULTI_CHARACTER_OPERATORS = [
  '**=',
  '//=',
  '<<=',
  '>>=',
  '...',
  ':=',
  '==',
  '!=',
  '<=',
  '>=',
  '->',
  '+=',
  '-=',
  '*=',
  '/=',
  '%=',
  '@=',
  '&=',
  '|=',
  '^=',
  '**',
  '//',
  '<<',
  '>>',
]

const SINGLE_CHARACTER_OPERATORS = new Set('+-*/%@&|^~<>=:.,;()[]{}')

interface StringStart {
  prefixLength: number
  quote: "'" | '"'
}

function findStringStart(source: string): StringStart | undefined {
  if (source[0] === "'" || source[0] === '"') {
    return { prefixLength: 0, quote: source[0] }
  }

  const prefix = STRING_PREFIX.exec(source)?.[0]
  const quote = source[prefix?.length ?? -1]
  if (prefix && (quote === "'" || quote === '"')) {
    return { prefixLength: prefix.length, quote }
  }

  return undefined
}

function readString(line: string, start: number, stringStart: StringStart): number {
  const quoteIndex = start + stringStart.prefixLength
  const tripleQuoted = line.startsWith(stringStart.quote.repeat(3), quoteIndex)
  const delimiter = tripleQuoted ? stringStart.quote.repeat(3) : stringStart.quote
  let cursor = quoteIndex + delimiter.length

  while (cursor < line.length) {
    if (line.startsWith(delimiter, cursor)) {
      return cursor + delimiter.length
    }

    if (line[cursor] === '\\') {
      cursor = Math.min(cursor + 2, line.length)
    } else {
      cursor += 1
    }
  }

  return line.length
}

function readOperator(line: string, start: number): string | undefined {
  return (
    MULTI_CHARACTER_OPERATORS.find((operator) => line.startsWith(operator, start)) ??
    (SINGLE_CHARACTER_OPERATORS.has(line[start]) ? line[start] : undefined)
  )
}

/**
 * Tokenize one display line of Python without changing, dropping, or escaping
 * any source characters. The returned text fragments can be rendered directly
 * as React children; no HTML parsing is needed.
 */
export function tokenizePythonLine(line: string): PythonSyntaxToken[] {
  const tokens: PythonSyntaxToken[] = []
  let cursor = 0
  let pendingDefinition: 'function' | 'class' | undefined

  const append = (kind: PythonSyntaxTokenKind, text: string) => {
    const previous = tokens.at(-1)
    if (previous?.kind === kind) {
      previous.text += text
    } else {
      tokens.push({ kind, text })
    }
  }

  while (cursor < line.length) {
    const source = line.slice(cursor)
    const whitespace = WHITESPACE.exec(source)?.[0]
    if (whitespace) {
      append('plain', whitespace)
      cursor += whitespace.length
      continue
    }

    if (line[cursor] === '#') {
      append('comment', line.slice(cursor))
      break
    }

    const stringStart = findStringStart(source)
    if (stringStart) {
      const end = readString(line, cursor, stringStart)
      append('string', line.slice(cursor, end))
      cursor = end
      pendingDefinition = undefined
      continue
    }

    const number = NUMBER.exec(source)?.[0]
    if (number) {
      append('number', number)
      cursor += number.length
      pendingDefinition = undefined
      continue
    }

    const identifier = IDENTIFIER.exec(source)?.[0]
    if (identifier) {
      let kind: PythonSyntaxTokenKind = 'plain'
      if (pendingDefinition) {
        kind = pendingDefinition
        pendingDefinition = undefined
      } else if (PYTHON_KEYWORDS.has(identifier)) {
        kind = 'keyword'
        if (identifier === 'def') pendingDefinition = 'function'
        if (identifier === 'class') pendingDefinition = 'class'
      } else if (PYTHON_BUILTINS.has(identifier)) {
        kind = 'builtin'
      }

      append(kind, identifier)
      cursor += identifier.length
      continue
    }

    const operator = readOperator(line, cursor)
    if (operator) {
      append('operator', operator)
      cursor += operator.length
      pendingDefinition = undefined
      continue
    }

    append('plain', line[cursor])
    cursor += 1
    pendingDefinition = undefined
  }

  return tokens
}
