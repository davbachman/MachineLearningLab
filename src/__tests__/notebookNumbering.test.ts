/// <reference types="node" />
import { readFileSync, readdirSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import { publishedAssignments } from '../data/assignments'

const directory = resolve('Homework2026')
const filenames = readdirSync(directory).filter(f => f.endsWith('.ipynb'))
interface Cell { cell_type: string; source: string[] }
function cells(filename: string): Cell[] { return JSON.parse(readFileSync(resolve(directory, filename), 'utf8')).cells }

describe('homework numbering', () => {
  it('has consecutive homework numbers, with two parts only for Homework 3', () => {
    const numbers = filenames.map(f => parseInt(f)).sort((a,b) => a-b)
    expect(numbers).toEqual([0,1,2,3,3, ...Array.from({length:22}, (_,i) => i+4)])
    expect(filenames).toContain('3aKNN.ipynb')
    expect(filenames).toContain('3bKNN.ipynb')
    for (const filename of filenames) {
      const title = cells(filename).find(c => c.cell_type === 'markdown')!.source.join('')
      expect(title.match(/Homework (\d+)/)?.[1], filename).toBe(String(parseInt(filename)))
    }
  })

  it('references existing notebook filenames and valid homework numbers', () => {
    for (const filename of filenames) {
      const source = cells(filename).flatMap(c => c.source).join('')
      for (const match of source.matchAll(/\b\d+[a-z]?[A-Z][A-Za-z0-9]*\.ipynb\b/g)) {
        expect(filenames, `${filename} references ${match[0]}`).toContain(match[0])
      }
      for (const match of source.matchAll(/\bHomework (\d+)\b/g)) expect(+match[1],filename).toBeLessThanOrEqual(25)
    }
    expect(publishedAssignments.map(a => a.displayNumber)).toEqual(Array.from({length:26}, (_,i)=>i))
    for (const assignment of publishedAssignments) {
      for (const q of assignment.questions.filter(q => q.kind === 'codeLab')) {
        for (const match of q.prompt.matchAll(/\b\d+[a-z]?[A-Z][A-Za-z0-9]*\.ipynb\b/g)) {
          expect(filenames).toContain(match[0])
          expect(parseInt(match[0])).toBe(assignment.displayNumber)
        }
      }
    }
  })
})
