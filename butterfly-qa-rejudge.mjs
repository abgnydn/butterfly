#!/usr/bin/env node
// Re-judge an existing butterfly-qa-eval result file with a different judge model.
// Eliminates same-model bias by getting a second independent judge's verdict
// on the SAME answer text, without re-running the (expensive) answerer.
//
// Usage:
//   JUDGE=google/gemma-4-e4b node tools/butterfly-qa-rejudge.mjs <result.json>
//   JUDGE=google/gemma-4-e4b node tools/butterfly-qa-rejudge.mjs    # latest

import { readFileSync, writeFileSync } from 'node:fs'
import { join, dirname, basename } from 'node:path'
import { fileURLToPath } from 'node:url'
import { readdirSync, statSync } from 'node:fs'

const ROOT = dirname(fileURLToPath(import.meta.url))
const LMS_BASE = process.env.LMS_BASE || 'http://localhost:1234/v1'
const JUDGE = process.env.JUDGE || 'google/gemma-4-e4b'

function latestResultFile() {
  const dir = join(ROOT, 'test-results', 'butterfly-sweep')
  const files = readdirSync(dir)
    .filter(f => f.startsWith('butterfly-qa-') && f.endsWith('.json') && !f.includes('rejudged'))
    .map(f => ({ f, mtime: statSync(join(dir, f)).mtimeMs }))
    .sort((a, b) => b.mtime - a.mtime)
  if (!files.length) throw new Error('no butterfly-qa-*.json files found')
  return join(dir, files[0].f)
}

const inPath = process.argv[2] || latestResultFile()
console.log(`[rejudge] input:  ${inPath}`)
console.log(`[rejudge] judge:  ${JUDGE}`)

const data = JSON.parse(readFileSync(inPath, 'utf8'))
const rows = data.rows
const cfg = data.config

const dataFile = cfg.data_file || 'longmemeval_oracle.json'
const datasetPath = join(ROOT, 'test-results', 'longmemeval', dataFile)
console.log(`[rejudge] dataset: ${datasetPath} (for question lookup)`)

const dataset = JSON.parse(readFileSync(datasetPath, 'utf8'))
const qidToQ = new Map(dataset.map(x => [x.question_id, x.question]))
console.log(`[rejudge] loaded ${qidToQ.size} qid→question mappings`)

const THINKING_MODELS = /(qwen3|deepseek-r1|r1-)/i
function applyNoThink(model, user) {
  return THINKING_MODELS.test(model) ? `/no_think ${user}` : user
}
function stripThink(s) { return (s || '').replace(/<think>[\s\S]*?<\/think>/g, '').trim() }

async function lmsChat(model, system, user, max_tokens, temperature = 0.0) {
  const res = await fetch(`${LMS_BASE}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model, messages: [
        { role: 'system', content: system },
        { role: 'user', content: applyNoThink(model, user) },
      ],
      max_tokens, temperature,
    }),
  })
  if (!res.ok) throw new Error(`LMS ${res.status}: ${await res.text()}`)
  const j = await res.json()
  return stripThink(j.choices?.[0]?.message?.content || '')
}

async function judge(question, expectedAnswer, gotAnswer) {
  const expStr = Array.isArray(expectedAnswer) ? expectedAnswer[0] : expectedAnswer
  const sys = `You grade whether an answer correctly conveys an expected fact.

Return STRICTLY a single digit as the LAST CHARACTER of your reply:
  2 = HIT — the answer accurately conveys the expected fact (paraphrasing fine)
  1 = PARTIAL — captures the load-bearing identification (file/function/mechanism) but mis-states or omits a detail
  0 = MISS — the fact is missing, vague, or invented

The LAST CHARACTER must be 0, 1, or 2.`
  const user = `QUESTION: ${question}\nEXPECTED: ${expStr}\nANSWER: ${gotAnswer}\n\nGrade. End with 0, 1, or 2.`
  const raw = await lmsChat(JUDGE, sys, user, 300, 0.0)
  const m = raw.match(/[012](?!.*[012])/s)
  return m ? Number(m[0]) : 0
}

const t0 = Date.now()
const outRows = []
let i = 0
for (const row of rows) {
  i++
  const q = qidToQ.get(row.qid) || ''
  if (!q) {
    console.warn(`  [warn] no question for qid=${row.qid}, skipping judge`)
    outRows.push({ ...row, judge_score_alt: 0 })
    continue
  }
  const score = await judge(q, row.gold, row.answer)
  outRows.push({ ...row, judge_score_alt: score })
  if (i % 20 === 0 || i === rows.length) {
    const elapsed = ((Date.now() - t0) / 1000).toFixed(0)
    console.log(`  [${i}/${rows.length}] ${elapsed}s elapsed`)
  }
}

// summary
const strategies = [...new Set(outRows.map(r => r.strategy))]
console.log('\n── per-strategy: original judge vs alt judge ──')
console.log('strategy             N    judge-hit (orig)  judge-hit (alt)   agreement')
console.log('─'.repeat(80))
for (const s of strategies) {
  const sub = outRows.filter(r => r.strategy === s)
  const N = sub.length
  const origHit = sub.filter(r => r.judge_score >= 2).length / N
  const altHit = sub.filter(r => r.judge_score_alt >= 2).length / N
  const agree = sub.filter(r => (r.judge_score >= 2) === (r.judge_score_alt >= 2)).length / N
  console.log(`${s.padEnd(20)} ${String(N).padStart(3)}    ${(origHit*100).toFixed(0).padStart(8)}%        ${(altHit*100).toFixed(0).padStart(6)}%      ${(agree*100).toFixed(0).padStart(6)}%`)
}

const outPath = inPath.replace(/\.json$/, '_rejudged.json')
writeFileSync(outPath, JSON.stringify({
  ...data,
  judge_alt: JUDGE,
  rejudged_at: new Date().toISOString(),
  rows: outRows,
}, null, 2))
console.log(`\nwrote: ${outPath}`)
