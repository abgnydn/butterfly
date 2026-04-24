// Hybrid: recursive summarization → chrysalis.
//
// Hypothesis: per-message tagging breaks coreference because the tagger sees
// each message in isolation ("she said it's fine" — who's "she"?). Recursive
// summarization summarizes CHUNKS, so within a chunk the antecedent + pronoun
// are co-located and the summarizer can resolve them.
//
// Pipeline:
//   1. Chunk transcript into N pieces.
//   2. Summarize each chunk with an explicit "resolve all pronouns" instruction.
//   3. Run chrysalis on the chunk-summaries (instead of raw messages).
//   4. Run needles through baseline (full) + hybrid (rebuilt) arms.
//
// Falsifier: N5 (coreference, "Did Sarah approve...") was 0/3 under butterfly.
// If hybrid scores 1/1 on N5 in a single seed, the hypothesis has legs.

import OpenAI from "openai";
import { writeFileSync } from "node:fs";
import { transcript, needles } from "./transcript.mjs";

// Local Ollama by default for everything.
const localClient = new OpenAI({
  baseURL: process.env.BASE_URL || "http://localhost:11434/v1",
  apiKey:  process.env.API_KEY  || "ollama",
});

// Optional bridge: an OpenAI-compatible HTTP server that fronts `claude -p`
// via your Max-plan auth. When BRIDGE_URL is set, summarizer + chrysalis use it.
const bridgeClient = process.env.BRIDGE_URL
  ? new OpenAI({ baseURL: process.env.BRIDGE_URL, apiKey: "bridge" })
  : null;

const MODELS = {
  summarizer: process.env.SUMM_MODEL    || (bridgeClient ? "claude-via-cli" : "gemma4:latest"),
  rebuild:    process.env.REBUILD_MODEL || (bridgeClient ? "claude-via-cli" : "gemma4:latest"),
  answer:     process.env.ANSWER_MODEL  || "gemma4:latest",
  judge:      process.env.JUDGE_MODEL   || "qwen3:0.6b",
};

// Pick client by model name: anything called "claude-*" routes to the bridge.
const clientFor = (model) => (model.startsWith("claude") && bridgeClient) ? bridgeClient : localClient;

const CLIENTS = {
  summarizer: clientFor(MODELS.summarizer),
  rebuild:    clientFor(MODELS.rebuild),
  answer:     clientFor(MODELS.answer),
  judge:      clientFor(MODELS.judge),
};

const NO_THINK    = "/no_think";
const RUN_SEED    = Number(process.env.SEED)            || 1;
const TIMEOUT_MS  = Number(process.env.CALL_TIMEOUT_MS) || 240000;
const N_CHUNKS    = Number(process.env.N_CHUNKS)        || 4;

const ts = () => new Date().toISOString().slice(11, 19);
const tokens = (s) => Math.ceil(s.length / 4);
const stripThink = (s) => s.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();

async function chat({ model, system, user, max_tokens = 800, label, client = localClient }) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), TIMEOUT_MS);
  const t0 = Date.now();
  console.log(`[${ts()}] ${label} start (${model})`);
  try {
    const r = await client.chat.completions.create({
      model,
      max_tokens,
      temperature: 0.2,
      seed: RUN_SEED,
      messages: [
        { role: "system", content: `${NO_THINK}\n\n${system}` },
        { role: "user",   content: user },
      ],
    }, { signal: ac.signal });
    clearTimeout(timer);
    const msg = r.choices[0].message;
    const out = stripThink(msg.content?.trim() ? msg.content : (msg.reasoning || ""));
    console.log(`[${ts()}] ${label} done in ${((Date.now() - t0) / 1000).toFixed(1)}s (${out.length} chars)`);
    return out;
  } catch (e) {
    clearTimeout(timer);
    console.log(`[${ts()}] ${label} FAILED: ${e.message?.slice(0, 100) || e}`);
    throw e;
  }
}

const msgsAsText = (msgs) =>
  msgs.map(m => `[${m.role}]\n${m.content}`).join("\n\n");

function chunkMessages(msgs, n) {
  const sz = Math.ceil(msgs.length / n);
  return Array.from({ length: n }, (_, i) => msgs.slice(i * sz, (i + 1) * sz)).filter(c => c.length > 0);
}

async function summarizeChunk(chunkMsgs, idx) {
  return chat({
    model: MODELS.summarizer,
    client: CLIENTS.summarizer,
    max_tokens: 600,
    label: `chunk-summ #${idx}`,
    system: `Summarize this conversation chunk in 4-7 terse bullets.

CRITICAL rules:
- Preserve every NAMED entity by name (people, files, packages, channels, identifiers, error messages).
- Resolve every pronoun to the named referent. NEVER use "she/he/they/it/her/him" without naming who/what — write the name in place of the pronoun.
- Keep specific facts: file:line, exact identifiers, decisions, code snippets.
- Drop pleasantries, acknowledgments, dead-end tangents.
- Output bullets only. No preamble. No meta-commentary.`,
    user: msgsAsText(chunkMsgs),
  });
}

async function rebuildFromSummaries(summaries, liveGoal) {
  const summBlock = summaries.map((s, i) => `=== chunk ${i + 1} ===\n${s}`).join("\n\n");
  return chat({
    model: MODELS.rebuild,
    client: CLIENTS.rebuild,
    max_tokens: 2000,
    label: "chrysalis (hybrid)",
    system: `You rebuild a sequence of chunk summaries into ONE coherent context the agent will resume from.

Rules:
- Lead with the live goal.
- Preserve every named entity, file:line, and decision from the summaries.
- Drop redundancy across chunks.
- Plain prose with light structure (short sections, terse bullets).
- Output ONLY the rebuild. No preamble, no meta-commentary about what you compressed.`,
    user: `LIVE GOAL: ${liveGoal}\n\nCHUNK SUMMARIES:\n\n${summBlock}`,
  });
}

async function answer(ctx, q, arm) {
  return chat({
    model: MODELS.answer,
    client: CLIENTS.answer,
    max_tokens: 400,
    label: `answer.${arm}`,
    system: `You continue a prior conversation. Answer the follow-up using ONLY the prior context. If a fact isn't there, say so plainly — do not invent. Be concise.`,
    user: `PRIOR CONTEXT:\n\n${ctx}\n\n---\n\nFOLLOW-UP QUESTION: ${q}`,
  });
}

async function judge(q, fact, ans, arm) {
  const raw = await chat({
    model: MODELS.judge,
    client: CLIENTS.judge,
    max_tokens: 250,
    label: `judge.${arm}`,
    system: `You grade whether an answer preserves a specific expected fact.

Score 1 if the answer accurately conveys the expected fact (paraphrasing fine).
Score 0 if missing, vague, contradictory, or invented.

The LAST CHARACTER of your response MUST be a single digit: 1 or 0. Nothing after.`,
    user: `QUESTION: ${q}\nEXPECTED FACT: ${fact}\nANSWER:\n${ans}\n\nGrade. End with 1 or 0.`,
  });
  const m = stripThink(raw).trim().match(/[01](?!.*[01])/s);
  if (!m) {
    console.warn(`      ⚠ judge no digit; defaulting 0. raw: ${stripThink(raw).slice(-100)}`);
    return 0;
  }
  return Number(m[0]);
}

async function run() {
  console.log(`\n=== HYBRID (recursive-summ → chrysalis) seed=${RUN_SEED} ===\n`);
  const fullText = msgsAsText(transcript);
  const fullTokens = tokens(fullText);
  const chunks = chunkMessages(transcript, N_CHUNKS);
  console.log(`Transcript: ${transcript.length} msgs, ~${fullTokens} tokens. Chunks: ${chunks.map(c => c.length).join("/")}.\n`);

  console.log(`[1/2] Summarizing ${chunks.length} chunks (${MODELS.summarizer})`);
  const summaries = [];
  for (let i = 0; i < chunks.length; i++) {
    summaries.push(await summarizeChunk(chunks[i], i));
  }
  const summBlock = summaries.join("\n\n");
  console.log(`      chunk-summaries total: ~${tokens(summBlock)} tokens\n`);

  console.log(`[2/2] Rebuilding from summaries (${MODELS.rebuild})`);
  const liveGoal = "Just shipped a fix for a flaky JWT-expiration test; user may follow up about the change or related items.";
  const rebuilt = await rebuildFromSummaries(summaries, liveGoal);
  const rebuiltTokens = tokens(rebuilt);
  const shrink = ((1 - rebuiltTokens / fullTokens) * 100).toFixed(1);
  console.log(`      rebuilt: ~${rebuiltTokens} tokens (${shrink}% smaller)\n`);

  writeFileSync("./hybrid-rebuilt.txt", rebuilt);
  writeFileSync("./hybrid-summaries.txt", summaries.map((s, i) => `=== chunk ${i + 1} ===\n${s}`).join("\n\n"));

  console.log(`Running ${needles.length} needles through baseline (full) + hybrid (rebuilt) arms\n`);
  const rows = [];
  for (const n of needles) {
    console.log(`--- ${n.id} [${n.category}] (${n.importance}) ---`);
    console.log(`Q: ${n.question}`);
    const ansBaseline = await answer(fullText, n.question, "baseline");
    const ansHybrid   = await answer(rebuilt,  n.question, "hybrid");
    console.log(`BASELINE: ${ansBaseline.replace(/\n/g, " ").slice(0, 200)}${ansBaseline.length > 200 ? "..." : ""}`);
    console.log(`HYBRID  : ${ansHybrid.replace(/\n/g, " ").slice(0, 200)}${ansHybrid.length > 200 ? "..." : ""}`);
    const sB = await judge(n.question, n.fact, ansBaseline, "baseline");
    const sH = await judge(n.question, n.fact, ansHybrid,   "hybrid");
    console.log(`SCORE: baseline=${sB}  hybrid=${sH}\n`);
    rows.push({ id: n.id, category: n.category, importance: n.importance, baseline: sB, hybrid: sH });
  }

  console.log(`\n=== HYBRID RESULTS (seed ${RUN_SEED}) ===\n`);
  console.log(`Needle  Category       Imp    Baseline  Hybrid`);
  console.log(`------  -------------  -----  --------  ------`);
  for (const r of rows) {
    console.log(`${r.id.padEnd(7)} ${r.category.padEnd(14)} ${r.importance.padEnd(6)} ${String(r.baseline).padEnd(9)} ${r.hybrid}`);
  }
  const bSum = rows.reduce((a, r) => a + r.baseline, 0);
  const hSum = rows.reduce((a, r) => a + r.hybrid,   0);
  console.log(`\nTotal: baseline ${bSum}/${rows.length}  hybrid ${hSum}/${rows.length}`);
  console.log(`Tokens: full ${fullTokens}  ->  hybrid ${rebuiltTokens}  (${shrink}% smaller)`);

  // Headline comparison: did N5 (coreference) get rescued?
  const n5 = rows.find(r => r.id === "N5");
  if (n5) {
    console.log(`\n*** Coreference test (N5):  butterfly was 0/3 across seeds.  hybrid is ${n5.hybrid}/1 this seed. ***`);
  }

  writeFileSync(`./hybrid-results-seed-${RUN_SEED}.json`, JSON.stringify({
    seed: RUN_SEED,
    transcript_tokens: fullTokens,
    rebuilt_tokens: rebuiltTokens,
    shrink_pct: Number(shrink),
    needles: rows,
    totals: { baseline: bSum, hybrid: hSum, of: rows.length },
  }, null, 2));
  console.log(`      wrote ./hybrid-results-seed-${RUN_SEED}.json\n`);
}

run().catch(e => { console.error("\nFAILED:", e.message); process.exit(1); });
