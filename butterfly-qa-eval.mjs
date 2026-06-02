#!/usr/bin/env node
// Butterfly v4.0 — downstream QA evaluation.
//
// The criticism: "you measured whether the literal turn text survived
// the compaction. That's not the same as 'the model can actually answer
// the question from the compacted memory.'" Fair.
//
// This script runs the FULL pipeline:
//   1. Compact LongMemEval's haystack_sessions via each strategy
//      (regex, longmem-trained, longmem-hybrid, lastN)
//   2. Feed (compacted memory + question) to a local LLM
//   3. Score the LLM's answer two ways:
//      a) substring match against gold answer (strict, biased toward
//         verbatim — under-reports paraphrased correct answers)
//      b) LLM-as-judge (qwen3-14b grades on a 0/1/2 rubric — biased
//         in the other direction, may inflate)
//   4. Report both scores per strategy, so the reader can decide which
//      to trust.
//
// Caveat — same-model bias: we use qwen3-14b for BOTH the answer and
// the judge calls. LongMemEval's official evaluator uses GPT-4 for
// both, which has the same problem; this is fine for a within-script
// comparison (all strategies graded the same way) but doesn't give
// an absolute accuracy number for the field.
//
// Usage:
//   node tools/butterfly-qa-eval.mjs                          # 20-example smoke
//   N=100 node tools/butterfly-qa-eval.mjs                    # bigger sample
//   DATA_FILE=longmemeval_s.json node tools/butterfly-qa-eval.mjs

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = dirname(fileURLToPath(import.meta.url))
const RESULTS_DIR = join(ROOT, 'test-results', 'butterfly-sweep')
const DATA_FILE = process.env.DATA_FILE || 'longmemeval_oracle.json'
const DATA_PATH = join(ROOT, 'test-results', 'longmemeval', DATA_FILE)
const N_EXAMPLES = parseInt(process.env.N || '20', 10)
const BUDGET = parseInt(process.env.BUDGET || '2048', 10)
const STRATEGIES = (process.env.STRATEGIES || 'regex,longmem-trained,longmem-hybrid,lastn').split(',')
const LMS_BASE = process.env.LMS_BASE || 'http://localhost:1234/v1'
const ANSWERER = process.env.ANSWERER || 'qwen3-14b-mlx'
const JUDGE = process.env.JUDGE || 'qwen3-14b-mlx'

const tokens = (s) => Math.ceil(s.length / 4)
const sigmoid = (z) => 1 / (1 + Math.exp(-z))
const softmax = (z) => { const m = Math.max(...z); const e = z.map(v => Math.exp(v-m)); const s = e.reduce((a,b)=>a+b,0); return e.map(v=>v/s) }

// ─── feature extractors (same as butterfly-longmemeval.mjs) ─────
const FEAT = [
  t => /\b\w+\/[\w\-./]+\.(ts|js|py|md|sql|yaml|toml|json|tsx|jsx|go|rs|html|css)\b/.test(t) ? 1 : 0,
  t => /TICKET-\d+/i.test(t) ? 1 : 0,
  t => /#[\w-]+/.test(t) ? 1 : 0,
  t => /@[\w-]+\/[\w-]+/.test(t) ? 1 : 0,
  t => /\blines?\s+\d+(\s*[-–]\s*\d+)?\b/i.test(t) ? 1 : 0,
  t => /\b(Decision|Root cause|Confirmed|Found it):/i.test(t) ? 1 : 0,
  t => /\b\w+\.\w+\(\)/.test(t) ? 1 : 0,
  t => /\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/.test(t) ? 1 : 0,
  t => /\b\d+\s*(req\/min|ms|s|min|gb|mb|kb|tokens?|seconds?|minutes?)\b/i.test(t) ? 1 : 0,
  t => /`[^`\n]{2,}`/.test(t) ? 1 : 0,
  t => /^\s*(ok|lgtm|sure|thx|noted|got it|will do|cool|sweet|alright|yep|yes|no|nope)[\s,.!]*$/i.test(t) ? 1 : 0,
  t => /(while you're (here|at it)|btw unrelated|side q|while we're here|unrelated|aside|off-topic)/i.test(t) ? 1 : 0,
  t => (t.endsWith('?') && t.length < 50 && !/\w+\/[\w.]+/.test(t)) ? 1 : 0,
  t => Math.min(1.0, Math.log(t.length + 1) / 6.5),
]

const longmemW = JSON.parse(readFileSync(join(ROOT, 'tools', 'butterfly-longmem-weights.json'), 'utf8'))

function regexTag(text) {
  let score = 0
  if (/\b\w+\/[\w\-./]+\.(ts|js|py|md|sql|yaml|toml|json|tsx|jsx|go|rs|html|css)\b/.test(text)) score += 3
  if (/TICKET-\d+/i.test(text)) score += 3
  if (/#[\w-]+/.test(text)) score += 3
  if (/@[\w-]+\/[\w-]+/.test(text)) score += 3
  if (/\blines?\s+\d+(\s*[-–]\s*\d+)?\b/i.test(text)) score += 3
  if (/\b(Decision|Root cause|Confirmed|Found it):/i.test(text)) score += 3
  if (/\b\w+\.\w+\(\)/.test(text)) score += 2
  if (/\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/.test(text)) score += 2
  if (/\b\d+\s*(req\/min|ms|s|min|gb|mb|kb|tokens?|seconds?|minutes?)\b/i.test(text)) score += 2
  if (/`[^`\n]{2,}`/.test(text)) score += 1
  if (/^\s*(ok|lgtm|sure|thx|noted|got it|will do|cool|sweet|alright|yep|yes|no|nope)[\s,.!]*$/i.test(text)) score -= 4
  if (/^\s*(ok|lgtm|sure|thx|noted|got it)\b.{0,30}$/i.test(text) && text.length < 50) score -= 2
  if (/(while you're (here|at it)|btw unrelated|side q|while we're here|unrelated|aside|off-topic)/i.test(text)) score -= 1
  if (text.endsWith('?') && text.length < 50 && !/\w+\/[\w.]+/.test(text)) score -= 1
  if (text.length < 40) score -= 1
  return score >= 3 ? 'keep' : score >= 1 ? 'summarize' : 'melt'
}

function longmemTrainedTag(text) {
  const x = FEAT.map(f => f(text))
  const z = longmemW.W.reduce((s, wv, j) => s + wv * x[j], longmemW.b)
  if (sigmoid(z) >= longmemW.threshold) return 'keep'
  const rt = regexTag(text)
  return rt === 'keep' ? 'summarize' : rt
}

// ─── chrysalis + lastN ────────────────────────────────────────────
function firstSentence(s) { const m = s.match(/^[^.!?]*[.!?]/); return (m ? m[0] : s).trim() }
function chrysalis(turns, tags, budget) {
  const parts = []
  for (let i = 0; i < turns.length; i++) {
    const t = tags[i], m = turns[i]
    if (t === 'keep') parts.push(`[${m.role}] ${m.content}`)
    else if (t === 'summarize') parts.push(`[${m.role}] ${firstSentence(m.content)}`)
  }
  let text = parts.join('\n')
  const maxChars = budget * 4
  if (text.length > maxChars) text = text.slice(0, maxChars).replace(/\s\S*$/, '') + '…'
  return text
}
function pickLastN(turns, budget) {
  const out = []; let acc = 0
  for (let i = turns.length - 1; i >= 0; i--) {
    const t = tokens(turns[i].content)
    if (acc + t > budget && out.length > 0) break
    out.unshift(turns[i]); acc += t
  }
  return out
}
function lastnMemory(turns, budget) {
  return pickLastN(turns, budget).map(m => `[${m.role}] ${m.content}`).join('\n')
}
function pickLastNIndices(turns, budget) {
  const idxs = []; let acc = 0
  for (let i = turns.length - 1; i >= 0; i--) {
    const t = tokens(turns[i].content)
    if (acc + t > budget && idxs.length > 0) break
    idxs.unshift(i); acc += t
  }
  return new Set(idxs)
}
function hybridMemory(turns, tags, totalBudget, lastnFrac = 0.4) {
  const lastnBudget = Math.floor(totalBudget * lastnFrac)
  const chrysBudget = totalBudget - lastnBudget
  const lastnIdx = pickLastNIndices(turns, lastnBudget)
  const chrysTags = tags.map((t, i) => lastnIdx.has(i) ? 'melt' : t)
  const chrys = chrysalis(turns, chrysTags, chrysBudget)
  const lastnPart = turns.filter((_, i) => lastnIdx.has(i)).map(m => `[${m.role}] ${m.content}`).join('\n')
  return chrys + (chrys && lastnPart ? '\n\n[recent ↓]\n' : '') + lastnPart
}

function makeMemory(turns, strategy, budget) {
  if (strategy === 'lastn') return lastnMemory(turns, budget)
  let tags
  if (strategy === 'regex') tags = turns.map(t => regexTag(t.content))
  else if (strategy === 'longmem-trained') tags = turns.map(t => longmemTrainedTag(t.content))
  else if (strategy === 'longmem-hybrid') tags = turns.map(t => longmemTrainedTag(t.content))
  else throw new Error(`unknown strategy: ${strategy}`)
  if (strategy === 'longmem-hybrid') return hybridMemory(turns, tags, budget)
  return chrysalis(turns, tags, budget)
}

// ─── LLM answerer + judge ───────────────────────────────────────
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
      model,
      messages: [
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

async function answer(memory, question) {
  const sys = `You answer questions using ONLY the prior conversation memory provided. If the memory doesn't contain the answer, say so plainly. Be concise — one sentence when possible.`
  const user = `PRIOR CONVERSATION MEMORY:\n\n${memory}\n\nQUESTION: ${question}\n\nAnswer:`
  return lmsChat(ANSWERER, sys, user, 150, 0.0)
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
  const raw = await lmsChat(JUDGE, sys, user, 60, 0.0)
  const m = raw.match(/[012](?!.*[012])/s)
  return m ? Number(m[0]) : 0
}

function substringMatch(gold, got) {
  if (!gold) return false
  const lower = got.toLowerCase()
  if (typeof gold === 'string') return lower.includes(gold.toLowerCase())
  if (Array.isArray(gold)) return gold.some(g => typeof g === 'string' && lower.includes(g.toLowerCase()))
  return false
}

// ─── streaming JSON array reader (for m support) ──────────────────
async function readJsonArray(path, maxExamples) {
  const { createReadStream } = await import('node:fs')
  return new Promise((resolve, reject) => {
    const stream = createReadStream(path, { encoding: 'utf8', highWaterMark: 1 << 20 })
    const examples = []
    let pending = []
    let depth = 0, inString = false, escape = false, seenOuter = false
    stream.on('data', (chunk) => {
      if (examples.length >= maxExamples) { stream.destroy(); return }
      for (let i = 0; i < chunk.length; i++) {
        const c = chunk[i]
        if (escape) { escape = false; continue }
        if (c === '\\') { escape = true; continue }
        if (c === '"') { inString = !inString; continue }
        if (inString) continue
        if (!seenOuter) { if (c === '[') seenOuter = true; continue }
        if (c === '{') depth++
        else if (c === '}') {
          depth--
          if (depth === 0) {
            const piece = pending.join('') + chunk.slice(0, i + 1)
            const start = piece.indexOf('{')
            try { examples.push(JSON.parse(piece.slice(start))) } catch (e) { reject(e); return }
            pending = []
            chunk = chunk.slice(i + 1)
            i = -1
            if (examples.length >= maxExamples) { stream.destroy(); return }
          }
        }
      }
      if (chunk.length > 0) pending.push(chunk)
    })
    stream.on('end', () => resolve(examples))
    stream.on('close', () => resolve(examples))
    stream.on('error', reject)
  })
}

// ─── main ────────────────────────────────────────────────────────
async function main() {
  console.log(`[qa-eval] config: dataset=${DATA_FILE} · N=${N_EXAMPLES} · budget=${BUDGET}`)
  console.log(`[qa-eval] strategies=${STRATEGIES.join(',')}  answerer=${ANSWERER}  judge=${JUDGE}`)

  const examples = await readJsonArray(DATA_PATH, N_EXAMPLES)
  console.log(`[qa-eval] loaded ${examples.length} examples`)

  const rows = []
  const t0 = Date.now()
  for (let i = 0; i < examples.length; i++) {
    const ex = examples[i]
    const turns = ex.haystack_sessions.flat()
    const question = ex.question
    const gold = ex.answer
    console.log(`\n[qa-eval] ${i+1}/${examples.length}  Q: ${question.slice(0, 80)}`)

    for (const s of STRATEGIES) {
      const memory = makeMemory(turns, s, BUDGET)
      const memTokens = tokens(memory)
      try {
        const ans = await answer(memory, question)
        const sub = substringMatch(gold, ans)
        const j = await judge(question, gold, ans)
        rows.push({
          idx: i, qid: ex.question_id, qtype: ex.question_type,
          strategy: s, budget: BUDGET,
          memory_tokens: memTokens,
          answer: ans.slice(0, 200),
          gold: Array.isArray(gold) ? gold[0] : gold,
          substring_match: sub,
          judge_score: j,
        })
        console.log(`  ${s.padEnd(18)} mem=${memTokens}t  judge=${j}  sub=${sub ? '✓' : '✗'}  ans="${ans.slice(0, 80).replace(/\n/g, ' ')}"`)
      } catch (e) {
        console.error(`  ${s.padEnd(18)} ERROR: ${e.message}`)
        rows.push({ idx: i, qid: ex.question_id, qtype: ex.question_type, strategy: s, budget: BUDGET, error: e.message })
      }
    }
  }

  // ── aggregate ────────────────────────────────────────────────
  console.log(`\n[qa-eval] swept ${rows.length} rows in ${((Date.now()-t0)/1000/60).toFixed(1)} min`)
  console.log(`\n── per-strategy summary ──`)
  console.log('strategy             N    sub-match     mean-judge    judge-hit (≥2)   judge-partial-or-hit (≥1)')
  console.log('─'.repeat(105))
  for (const s of STRATEGIES) {
    const subset = rows.filter(r => r.strategy === s && !r.error)
    const N = subset.length
    if (N === 0) continue
    const subMatch = subset.filter(r => r.substring_match).length / N
    const meanJudge = subset.reduce((acc, r) => acc + r.judge_score, 0) / N
    const judgeHit = subset.filter(r => r.judge_score === 2).length / N
    const judgePartialOrHit = subset.filter(r => r.judge_score >= 1).length / N
    console.log(
      `${s.padEnd(20)}${N.toString().padStart(3)}${(subMatch*100).toFixed(0).padStart(11)}%` +
      `${meanJudge.toFixed(2).padStart(14)}` +
      `${(judgeHit*100).toFixed(0).padStart(15)}%` +
      `${(judgePartialOrHit*100).toFixed(0).padStart(22)}%`
    )
  }

  mkdirSync(RESULTS_DIR, { recursive: true })
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const out = join(RESULTS_DIR, `butterfly-qa-${stamp}.json`)
  writeFileSync(out, JSON.stringify({
    started_at: new Date(t0).toISOString(),
    finished_at: new Date().toISOString(),
    runtime_ms: Date.now() - t0,
    config: { data_file: DATA_FILE, n_examples: N_EXAMPLES, budget: BUDGET, strategies: STRATEGIES, answerer: ANSWERER, judge: JUDGE },
    rows,
  }, null, 2))
  console.log(`\nwrote: ${out}`)
}

main().catch(e => { console.error('FATAL:', e); process.exit(1) })
