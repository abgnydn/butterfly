// transgen: transgenerational metamorphosis test.
//
// Motivated by Jo Nagai's 2024-2026 butterfly-memory experiment: conditioned
// memories survive multiple generations after caterpillar→butterfly metamorphosis.
// The artificial-agent analog: importance-tagged context should survive multiple
// metamorphosis events in an agent loop, where naive truncation loses the needle
// after the first cocoon.
//
// Pipeline per task:
//   F0 = original 20-msg transcript (needle planted at msg 3-9)
//      → metamorphosis 1 (compress to TARGET_TOKENS) → F1 rebuild
//      → inject 6-8 noise messages of unrelated agent activity
//      → metamorphosis 2 → F2 rebuild
//      → more noise
//      → metamorphosis 3 → F3 rebuild
//      → ask the needle question on F3
//
// Arms: butterfly (tag + chrysalis) vs lastN (suffix truncation).
// Primary metric: F3 needle survival rate per method.

import OpenAI from "openai";
import { writeFileSync, readFileSync, mkdirSync, existsSync, readdirSync } from "node:fs";
import { join } from "node:path";

const localClient  = new OpenAI({ baseURL: "http://localhost:11434/v1", apiKey: "ollama" });
const bridgeClient = new OpenAI({ baseURL: process.env.BRIDGE_URL || "http://localhost:3002/v1", apiKey: "bridge" });

// Defaults: all-Haiku via arena bridge (Max-plan free, ~12s/turn). Tagger stays
// local by default because per-message tagging is high-frequency and qwen3:0.6b
// is fast enough even when noisy.
const MODELS = {
  generator: process.env.GEN_MODEL      || "haiku",
  tagger:    process.env.TAGGER_MODEL   || "qwen3:0.6b",
  rebuild:   process.env.REBUILD_MODEL  || "haiku",
  answer:    process.env.ANSWER_MODEL   || "haiku",
  judge:     process.env.JUDGE_MODEL    || "haiku",
};

// Route by model name: bridge for haiku/sonnet/opus/claude-*, local Ollama otherwise.
const clientFor = (model) =>
  (/^(haiku|sonnet|opus|claude[-_])/i.test(model) ? bridgeClient : localClient);
const CLIENTS = {
  generator: clientFor(MODELS.generator),
  tagger:    clientFor(MODELS.tagger),
  rebuild:   clientFor(MODELS.rebuild),
  answer:    clientFor(MODELS.answer),
  judge:     clientFor(MODELS.judge),
};

const NO_THINK       = "/no_think";
const TIMEOUT_MS     = Number(process.env.CALL_TIMEOUT_MS) || 240000;
const TARGET_TOKENS  = Number(process.env.TARGET_TOKENS)   || 400;
const N_GENERATIONS  = Number(process.env.N_GENERATIONS)   || 3;
const TASKS_FILE     = process.env.TASKS_FILE              || "./paper-test-tasks.json";
const CHECKPOINT_DIR = process.env.CHECKPOINT_DIR          || "./checkpoints";

// Per-task checkpointing — skip tasks with existing checkpoint. `rm -rf checkpoints/` to start fresh.
mkdirSync(CHECKPOINT_DIR, { recursive: true });
const checkpointPath = (id) => join(CHECKPOINT_DIR, `task-${id}.json`);
const loadCheckpoint = (id) => existsSync(checkpointPath(id)) ? JSON.parse(readFileSync(checkpointPath(id), "utf8")) : null;
const saveCheckpoint = (id, row) => writeFileSync(checkpointPath(id), JSON.stringify(row, null, 2));

const ts = () => new Date().toISOString().slice(11, 19);
const tokens = (s) => Math.ceil(s.length / 4);
const stripThink = (s) => s.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();

function extractJSON(raw) {
  let t = stripThink(raw).replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
  const firstObj = t.indexOf("{"), firstArr = t.indexOf("[");
  const startsWithObj = firstObj !== -1 && (firstArr === -1 || firstObj < firstArr);
  if (startsWithObj) {
    const end = t.lastIndexOf("}");
    if (end > firstObj) return JSON.parse(t.slice(firstObj, end + 1));
  }
  if (firstArr !== -1) {
    const end = t.lastIndexOf("]");
    if (end > firstArr) return JSON.parse(t.slice(firstArr, end + 1));
  }
  if (firstObj !== -1) {
    const end = t.lastIndexOf("}");
    if (end > firstObj) return JSON.parse(t.slice(firstObj, end + 1));
  }
  throw new Error(`no JSON in:\n${raw.slice(0, 300)}`);
}

// chat() with retry-on-any-error + exponential backoff. Survives transient
// bridge restarts, connection blips, timeouts.
async function chatOnce({ model, system, user, max_tokens, client, signal }) {
  const r = await client.chat.completions.create({
    model, max_tokens, temperature: 0.2,
    messages: [
      { role: "system", content: `${NO_THINK}\n\n${system}` },
      { role: "user",   content: user },
    ],
  }, { signal });
  const msg = r.choices[0].message;
  return stripThink(msg.content?.trim() ? msg.content : (msg.reasoning || ""));
}

async function chat({ model, system, user, max_tokens = 800, client = bridgeClient, label }) {
  const maxAttempts = 4;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), TIMEOUT_MS);
    const t0 = Date.now();
    console.log(`[${ts()}] ${label} start (${model}${attempt > 1 ? `, retry ${attempt}` : ""})`);
    try {
      const out = await chatOnce({ model, system, user, max_tokens, client, signal: ac.signal });
      clearTimeout(timer);
      console.log(`[${ts()}] ${label} done in ${((Date.now() - t0) / 1000).toFixed(1)}s (${out.length} chars, ~${tokens(out)} tok)`);
      return out;
    } catch (e) {
      clearTimeout(timer);
      console.log(`[${ts()}] ${label} attempt ${attempt} failed: ${(e.message || String(e)).slice(0, 120)}`);
      if (attempt >= maxAttempts) throw e;
      const delay = 3000 * Math.pow(2, attempt - 1);
      console.log(`[${ts()}] ${label} retrying in ${delay / 1000}s...`);
      await new Promise(r => setTimeout(r, delay));
    }
  }
}

const msgsAsText = (msgs) => msgs.map(m => `[${m.role}]\n${m.content}`).join("\n\n");
const sumTokens = (msgs) => msgs.reduce((n, m) => n + tokens(m.content), 0);

async function tagOne(message, idx, gen) {
  try {
    const raw = await chat({
      model: MODELS.tagger,
      client: CLIENTS.tagger,
      max_tokens: 200,
      label: `tag.G${gen} #${idx}`,
      system: `Tag ONE message for context compaction. Reply with one JSON object only:
{"action": "keep" | "summarize" | "melt", "importance": 0.0-1.0, "reason": "<10 words"}

keep = irreplaceable atom (root cause, owner, file:line, decision)
summarize = substantive but a one-line gist suffices
melt = greetings, acks, dead-end tangents, noise

Examples:
"Sure. Share the file." -> {"action":"melt","importance":0.05,"reason":"ack"}
"Root cause: Date.now() race." -> {"action":"keep","importance":0.95,"reason":"root cause"}`,
      user: `Message #${idx}:\n${message.content}\n---\nReturn JSON.`,
    });
    return extractJSON(raw);
  } catch (e) {
    return { action: "keep", importance: 0.5, reason: "tag-fail" };
  }
}

async function butterflyCompress(messages, gen) {
  const tags = [];
  for (let i = 0; i < messages.length; i++) tags.push(await tagOne(messages[i], i, gen));

  const tagged = messages.map((m, i) =>
    `[#${i} role=${m.role} action=${tags[i].action} imp=${tags[i].importance}]\n${m.content}`
  ).join("\n\n");

  return chat({
    model: MODELS.rebuild,
    client: CLIENTS.rebuild,
    max_tokens: Math.ceil(TARGET_TOKENS * 1.4),
    label: `chrysalis.G${gen}`,
    system: `You rebuild a tagged conversation transcript into a small coherent context the agent will resume from.

HARD CONSTRAINT: the rebuild MUST fit in approximately ${TARGET_TOKENS} tokens (~${TARGET_TOKENS * 4} characters). Be ruthless about cutting non-essential content. Brevity is REQUIRED, not optional.

Rules:
- KEEP messages: preserve every load-bearing fact, name, file:line, decision, code snippet — verbatim if short, terse paraphrase if not.
- SUMMARIZE messages: collapse to one phrase each.
- MELT messages: drop entirely.
- Output ONLY the rebuilt context. No preamble, no meta-commentary, no markdown headers heavier than ##.`,
    user: `TAGGED TRANSCRIPT (${tagged.length} chars):\n\n${tagged}`,
  });
}

function pickLastN(messages, targetTokens) {
  const out = [];
  let acc = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const t = tokens(messages[i].content);
    if (acc + t > targetTokens && out.length > 0) break;
    out.unshift(messages[i]);
    acc += t;
  }
  return out;
}

async function generateNoise(idx) {
  const raw = await chat({
    model: MODELS.generator,
    client: CLIENTS.generator,
    max_tokens: 1200,
    label: `noise #${idx}`,
    system: `Generate 6-8 realistic conversation messages between a developer (USER) and an AI assistant (ASSISTANT) — off-topic / refactor / style noise that happens during a long debugging session AFTER the main bug has been diagnosed.

Topics: code style debates, refactor discussions, tooling tangents, minor unrelated bugs.

CRITICAL: do NOT mention any specific bug root cause, ownership, file:line that resembles a "load-bearing fact." This is filler noise — tag-worthy as "melt".

Output STRICT JSON: a single array of message objects [{"role":"user|assistant","content":"..."}, ...]. No preamble, no fences. First char '['. Last char ']'.`,
    user: `Generate noise batch #${idx}.`,
  });
  return extractJSON(raw);
}

async function answerQuestion(ctx, q, label) {
  return chat({
    model: MODELS.answer,
    client: CLIENTS.answer,
    max_tokens: 300,
    label: `answer.${label}`,
    system: `You continue a prior conversation. Answer the follow-up using ONLY the prior context. If a fact isn't there, say so plainly — do not invent. Be concise.`,
    user: `PRIOR CONTEXT:\n\n${ctx}\n\n---\n\nFOLLOW-UP: ${q}`,
  });
}

async function judge(question, fact, ans, label) {
  const raw = await chat({
    model: MODELS.judge,
    client: CLIENTS.judge,
    max_tokens: 200,
    label: `judge.${label}`,
    system: `Grade whether an answer accurately conveys an expected fact.

Score 1 if the answer accurately conveys the EXPECTED FACT (paraphrasing fine).
Score 0 if missing, vague, contradictory, or invented.

Last character of your response MUST be a single digit: 1 or 0.`,
    user: `QUESTION: ${question}\nEXPECTED FACT: ${fact}\nANSWER:\n${ans}\n\nGrade. End with 1 or 0.`,
  });
  const m = stripThink(raw).trim().match(/[01](?!.*[01])/s);
  return m ? Number(m[0]) : 0;
}

async function runButterflyChain(initialMessages, noiseBatches) {
  let messages = initialMessages.slice();
  const trace = [{ gen: 0, kind: "initial", tokens: sumTokens(messages) }];

  for (let gen = 1; gen <= N_GENERATIONS; gen++) {
    const rebuilt = await butterflyCompress(messages, gen);
    messages = [{ role: "system", content: `[CONTEXT REBUILT AT GEN ${gen}]\n${rebuilt}` }];
    trace.push({ gen, kind: "rebuilt", tokens: tokens(rebuilt) });
    if (gen < N_GENERATIONS) {
      messages = [...messages, ...noiseBatches[gen - 1]];
      trace.push({ gen, kind: "after-noise", tokens: sumTokens(messages) });
    }
  }
  return { messages, trace };
}

async function runLastNChain(initialMessages, noiseBatches) {
  let messages = initialMessages.slice();
  const trace = [{ gen: 0, kind: "initial", tokens: sumTokens(messages) }];

  for (let gen = 1; gen <= N_GENERATIONS; gen++) {
    messages = pickLastN(messages, TARGET_TOKENS);
    trace.push({ gen, kind: "truncated", tokens: sumTokens(messages) });
    if (gen < N_GENERATIONS) {
      messages = [...messages, ...noiseBatches[gen - 1]];
      trace.push({ gen, kind: "after-noise", tokens: sumTokens(messages) });
    }
  }
  return { messages, trace };
}

async function run() {
  console.log(`\n=== TRANSGEN: butterfly vs lastN across ${N_GENERATIONS} metamorphoses at ${TARGET_TOKENS}-token budget ===\n`);
  console.log(`Models: gen=${MODELS.generator} tag=${MODELS.tagger} rebuild=${MODELS.rebuild} answer=${MODELS.answer} judge=${MODELS.judge}`);
  console.log(`Bridge: ${bridgeClient.baseURL}\n`);

  const tasks = JSON.parse(readFileSync(TASKS_FILE, "utf8"));
  console.log(`Loaded ${tasks.length} tasks from ${TASKS_FILE}\n`);

  const rows = [];
  for (let ti = 0; ti < tasks.length; ti++) {
    const t = tasks[ti];

    const existing = loadCheckpoint(t.id);
    if (existing) {
      console.log(`\n--- ${t.id} (resumed from checkpoint: butterfly=${existing.butterfly} lastN=${existing.lastN}) ---`);
      rows.push(existing);
      continue;
    }

    console.log(`\n--- ${t.id} ${t.topic.slice(0, 60)} ---`);
    console.log(`  initial transcript: ${t.messages.length} msgs, ~${sumTokens(t.messages)} tokens`);

    const noiseBatches = [];
    for (let g = 0; g < N_GENERATIONS - 1; g++) {
      try { noiseBatches.push(await generateNoise(`${t.id}-${g + 1}`)); }
      catch (e) { console.log(`  noise gen ${g + 1} failed: ${e.message?.slice(0, 80)}`); noiseBatches.push([]); }
    }

    let bResult, lResult, bScore = null, lScore = null, bAns, lAns;
    try {
      console.log(`  -- butterfly chain --`);
      bResult = await runButterflyChain(t.messages, noiseBatches);
      console.log(`     trace: ${bResult.trace.map(s => `G${s.gen}:${s.kind}=${s.tokens}t`).join(" → ")}`);

      console.log(`  -- lastN chain --`);
      lResult = await runLastNChain(t.messages, noiseBatches);
      console.log(`     trace: ${lResult.trace.map(s => `G${s.gen}:${s.kind}=${s.tokens}t`).join(" → ")}`);

      bAns = await answerQuestion(msgsAsText(bResult.messages), t.question, "butterfly");
      lAns = await answerQuestion(msgsAsText(lResult.messages), t.question, "lastN");
      console.log(`  BUTTERFLY ANS: ${bAns.replace(/\n/g, " ").slice(0, 180)}`);
      console.log(`  LASTN     ANS: ${lAns.replace(/\n/g, " ").slice(0, 180)}`);

      bScore = await judge(t.question, t.needle_fact, bAns, "butterfly");
      lScore = await judge(t.question, t.needle_fact, lAns, "lastN");
      console.log(`  SCORE: butterfly=${bScore}  lastN=${lScore}`);

      const row = {
        id: t.id, topic: t.topic, question: t.question,
        butterfly: bScore, lastN: lScore,
        butterfly_trace: bResult.trace, lastN_trace: lResult.trace,
        butterfly_answer: bAns?.slice(0, 300),
        lastN_answer: lAns?.slice(0, 300),
        needle_fact: t.needle_fact,
        completed_at: new Date().toISOString(),
      };
      saveCheckpoint(t.id, row);
      rows.push(row);
    } catch (e) {
      console.log(`  TASK FAILED: ${e.message}`);
      const row = { id: t.id, topic: t.topic, error: e.message, completed_at: new Date().toISOString() };
      saveCheckpoint(t.id, row);
      rows.push(row);
    }
  }

  console.log(`\n\n=== TRANSGEN RESULTS (${N_GENERATIONS} generations, ${TARGET_TOKENS}-token budget) ===\n`);
  console.log(`Task    Topic                                          Butterfly  LastN`);
  console.log(`------  -------------------------------------------    ---------  -----`);
  for (const r of rows) {
    const topic = (r.topic || "").slice(0, 44).padEnd(45);
    console.log(`${r.id.padEnd(7)} ${topic} ${String(r.butterfly).padEnd(10)} ${r.lastN}`);
  }
  const valid = rows.filter(r => r.butterfly !== null && r.lastN !== null && !r.error);
  const bSum = valid.reduce((a, r) => a + r.butterfly, 0);
  const lSum = valid.reduce((a, r) => a + r.lastN, 0);
  const wins = valid.filter(r => r.butterfly > r.lastN).length;
  const losses = valid.filter(r => r.butterfly < r.lastN).length;
  const ties = valid.filter(r => r.butterfly === r.lastN).length;
  console.log(`\nValid runs: ${valid.length}/${rows.length}`);
  console.log(`Survival across ${N_GENERATIONS} metamorphoses: butterfly ${bSum}/${valid.length} (${(100 * bSum / valid.length).toFixed(0)}%)   lastN ${lSum}/${valid.length} (${(100 * lSum / valid.length).toFixed(0)}%)`);
  console.log(`Head-to-head: butterfly wins ${wins}, loses ${losses}, ties ${ties}`);

  writeFileSync("./transgen-results.json", JSON.stringify({
    n_generations: N_GENERATIONS,
    target_tokens: TARGET_TOKENS,
    models: MODELS,
    butterfly_survival: bSum / valid.length,
    lastN_survival: lSum / valid.length,
    head_to_head: { wins, losses, ties },
    rows,
  }, null, 2));
  console.log(`\nWrote ./transgen-results.json`);
}

run().catch(e => { console.error("FAILED:", e.message); process.exit(1); });
