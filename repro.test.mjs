// Reproducibility guard for the two pure-JS (offline, deterministic) rows of
// the Butterfly study. The LLM-tagger rows (adversarial/taggers/longmemeval)
// are not deterministic and are intentionally not covered here.
//
//   node --test        (or: npm test)
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'

const run = (file) =>
  execFileSync('node', [file], { encoding: 'utf8', cwd: import.meta.dirname })

// P-20260515-06: the confirmed hard regime — 4/4 transcripts, mean Δ = 100pp.
test('confirm: butterfly-purecode-hard reproduces CONFIRMED @ Δ=100pp', () => {
  const out = run('butterfly-purecode-hard.mjs')
  assert.match(out, /PRE-REGISTERED VERDICT \(P-20260515-06\): CONFIRMED/)
  assert.match(out, /All 4 transcripts: bfly > lastN\. Mean Δ = 100pp/)
})

// The 960-cell phase sweep — clear-win region grows monotonically with gens.
test('sweep: phase-diagram win-region cell counts are stable', () => {
  const out = run('butterfly-sweep-phasediagram.mjs')
  const expected = { 1: 30, 2: 34, 3: 36, 4: 37, 5: 39 }
  for (const [gens, cells] of Object.entries(expected)) {
    assert.match(
      out,
      new RegExp(`gens=${gens}: ${cells} cells`),
      `gens=${gens} should be ${cells} cells`,
    )
  }
})
