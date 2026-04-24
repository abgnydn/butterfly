// Butterfly falsifier: does a metamorphosed (rebuilt) context preserve facts
// as well as the original full transcript?
//
// Pipeline:
//   transcript ─► tagger (per-message imaginal-cell tags)
//                  └─► chrysalis (one rebuild call) ─► rebuilt context
//
// Then for each test needle:
//   BASELINE  : answer(full transcript, question)
//   BUTTERFLY : answer(rebuilt context, question)
//   judge both against the expected fact -> 0/1
//
// Local mode: talks to Ollama via its OpenAI-compatible endpoint.
// Override any model with env vars: TAGGER_MODEL, REBUILD_MODEL, ANSWER_MODEL, JUDGE_MODEL.

import OpenAI from "openai";
import { writeFileSync } from "node:fs";
import { transcript, needles } from "./transcript.mjs";

// Default endpoint = Ollama. The tagger can be split off to a different host
// (e.g. bitnet.cpp's llama-server on :8080) by setting BASE_URL_TAGGER.
const mainClient = new OpenAI({
  baseURL: process.env.BASE_URL || "http://localhost:11434/v1",
  apiKey:  process.env.API_KEY  || "ollama",  // ignored by Ollama, required by SDK
});

const taggerClient = process.env.BASE_URL_TAGGER
  ? new OpenAI({
      baseURL: process.env.BASE_URL_TAGGER,
      apiKey:  process.env.API_KEY_TAGGER || "bitnet",
    })
  : mainClient;

// Empirical: gemma4 doesn't have thinking mode and emits clean answers in
// `content`. The qwen3.5 variants on this box dump CoT and time out before
// producing the rebuild. qwen3:0.6b is fast enough for per-message tagging
// even when half its outputs need a fallback. BitNet b1.58 2B is the next
// candidate for the tagger slot — set BASE_URL_TAGGER + TAGGER_MODEL.
const MODELS = {
  tagger:  process.env.TAGGER_MODEL  || "qwen3:0.6b",
  rebuild: process.env.REBUILD_MODEL || "gemma4:latest",
  answer:  process.env.ANSWER_MODEL  || "gemma4:latest",
  judge:   process.env.JUDGE_MODEL   || "gemma4:latest",
};

// Rough token estimate. char/4 is the usual rule of thumb.
const tokens = (s) => Math.ceil(s.length / 4);
const transcriptTokens = transcript.reduce((n, m) => n + tokens(m.content), 0);

// Robust JSON extractor: small open models love to emit <think>...</think> blocks,
// markdown fences, and stray prose. Pull out the first balanced {...} we see.
function extractJSON(raw) {
  let t = raw.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  t = t.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
  const start = t.indexOf("{");
  const end   = t.lastIndexOf("}");
  if (start === -1 || end === -1 || end < start) {
    throw new Error(`no JSON object in model output:\n${raw.slice(0, 300)}`);
  }
  return JSON.parse(t.slice(start, end + 1));
}

// Strip <think> from prose outputs too (chrysalis, answer).
function stripThink(raw) {
  return raw.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
}

// Local models often dump chain-of-thought into `content` despite /no_think.
// We instruct them to wrap the real answer between markers, then extract.
// If markers are absent, fall back to the whole text (warn-only).
function extractMarked(raw, openTag, closeTag) {
  const open = raw.lastIndexOf(openTag);
  if (open === -1) return raw.trim();
  const after = raw.slice(open + openTag.length);
  const close = after.indexOf(closeTag);
  return (close === -1 ? after : after.slice(0, close)).trim();
}

// Qwen3 family ships with thinking-mode ON; it emits <think>...</think> before
// the real answer and burns the token budget. /no_think disables it.
const NO_THINK = "/no_think";

const ts = () => new Date().toISOString().slice(11, 19);
const TIMEOUT_MS = Number(process.env.CALL_TIMEOUT_MS) || 60000;
const RUN_SEED = Number(process.env.SEED) || 1;

async function chatOnce({ model, system, user, max_tokens, client, jsonOnly, signal }) {
  const r = await client.chat.completions.create({
    model,
    max_tokens,
    temperature: 0.2,
    seed: RUN_SEED,  // Ollama honors this for reproducibility across runs.
    messages: [
      { role: "system", content: `${NO_THINK}\n\n${system}` },
      { role: "user",   content: user   },
    ],
    ...(jsonOnly ? { response_format: { type: "json_object" } } : {}),
  }, { signal });
  const msg = r.choices[0].message;
  const content = msg.content ?? "";
  if (!content.trim() && msg.reasoning) return msg.reasoning;
  return content;
}

async function chat({ model, system, user, max_tokens = 600, client = mainClient, jsonOnly = false, label = "call" }) {
  for (let attempt = 1; attempt <= 2; attempt++) {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), TIMEOUT_MS);
    const t0 = Date.now();
    console.log(`[${ts()}] ${label} start (model=${model}, attempt ${attempt})`);
    try {
      const out = await chatOnce({ model, system, user, max_tokens, client, jsonOnly, signal: ac.signal });
      clearTimeout(timer);
      const dt = ((Date.now() - t0) / 1000).toFixed(1);
      console.log(`[${ts()}] ${label} done in ${dt}s (${out.length} chars)`);
      return out;
    } catch (e) {
      clearTimeout(timer);
      const dt = ((Date.now() - t0) / 1000).toFixed(1);
      const why = ac.signal.aborted ? `TIMEOUT after ${dt}s` : `ERR ${e.message?.slice(0, 80) || e}`;
      console.log(`[${ts()}] ${label} ${why} (attempt ${attempt})`);
      if (attempt === 2) throw e;
    }
  }
}

// ---------- Tagger ----------
async function tagOne(message, idx) {
  const raw = await chat({
    model: MODELS.tagger,
    client: taggerClient,
    max_tokens: 400,
    jsonOnly: true,
    label: `tagger #${idx}`,
    system: `Tag ONE message for context compaction. Reply with one JSON object only:
{"action": "keep" | "summarize" | "melt", "importance": 0.0-1.0, "reason": "<10 words"}

Aim across many messages: ~20% keep, ~40% summarize, ~40% melt. Use SUMMARIZE for substance-with-flab — under-using it forces everything into binary keep-or-lose.

keep = irreplaceable atom (root cause, owner, file:line, decision, committed code)
summarize = substantive but a one-line gist suffices (tool output, multi-step explanation, plan)
melt = greetings, acks, restatements, dead-end tangents, superseded scratch

Examples:
"Sure. Share the file." -> {"action":"melt","importance":0.05,"reason":"ack"}
"lgtm. pushing." -> {"action":"melt","importance":0.05,"reason":"ack"}
"Root cause: Date.now() called twice; second drift." -> {"action":"keep","importance":0.95,"reason":"root cause"}
"Sarah owns @company/jwt-utils, #auth-platform." -> {"action":"keep","importance":0.85,"reason":"ownership"}
"Read 87 lines, confirmed issueToken uses Date.now()." -> {"action":"summarize","importance":0.4,"reason":"verbose tool output, one fact"}
"Two reasons CI fails more: shared CPU stalls; narrow race window." -> {"action":"summarize","importance":0.5,"reason":"multi-part explanation"}`,
    user: `Message #${idx} (role=${message.role}):\n---\n${message.content}\n---\nReturn the JSON object only.`,
  });
  try {
    return extractJSON(raw);
  } catch (e) {
    // Even with response_format json_object, parsing can fail on rare edge cases.
    // Default to "keep" so we don't lose information; chrysalis can still compress.
    console.warn(`\n      ⚠ msg #${idx} tag failed (${e.message.split("\n")[0]}); defaulting to keep`);
    return { action: "keep", importance: 0.5, reason: "tag-parse-failed" };
  }
}

async function tagAll(messages) {
  // Sequential — Ollama serves one request at a time per model anyway.
  const tags = [];
  for (let i = 0; i < messages.length; i++) {
    tags.push(await tagOne(messages[i], i));
  }
  return tags;
}

// ---------- Chrysalis ----------
async function chrysalis(messages, tags, liveGoal) {
  const tagged = messages.map((m, i) => {
    const t = tags[i];
    return `[#${i} role=${m.role} action=${t.action} importance=${t.importance} reason="${t.reason}"]\n${m.content}`;
  }).join("\n\n");

  const raw = await chat({
    model: MODELS.rebuild,
    max_tokens: 2000,
    label: "chrysalis",
    system: `You rebuild a tagged conversation transcript into a small coherent context the agent will resume from.

Each input message has action=keep|summarize|melt and importance.

Rules:
- KEEP messages: preserve every load-bearing fact, name, file:line, decision, code snippet. Paraphrase only when needed for flow.
- SUMMARIZE messages: condense to one terse line each.
- MELT messages: do not include or mention.
- Lead with the live goal.
- Plain prose with light structure (short sections, terse bullets).
- Output ONLY the rebuilt context. No preamble, no meta-commentary about what you compressed.`,
    user: `LIVE GOAL: ${liveGoal}\n\nTAGGED TRANSCRIPT:\n\n${tagged}`,
  });
  return stripThink(raw);
}

// ---------- Answering (both arms use the SAME model and prompt) ----------
async function answer(contextText, question, arm) {
  const raw = await chat({
    model: MODELS.answer,
    max_tokens: 400,
    label: `answer.${arm}`,
    system: `You continue a prior conversation. Answer the follow-up using ONLY the prior context. If a fact isn't there, say so plainly — do not invent. Be concise.`,
    user: `PRIOR CONTEXT:\n\n${contextText}\n\n---\n\nFOLLOW-UP QUESTION: ${question}`,
  });
  return stripThink(raw);
}

const transcriptAsText = (msgs) =>
  msgs.map(m => `[${m.role}]\n${m.content}`).join("\n\n");

// Last-N baseline: keep the suffix of messages whose combined token count fits
// within `targetTokens`. Compares butterfly's semantic-compression against naive
// recency truncation at the same budget.
function lastNMessages(messages, targetTokens) {
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

// ---------- Judge ----------
// Small models choke on JSON. Ask for a single digit at the end — much easier
// to hit reliably, and we can grep it out regardless of any prose preamble.
async function judge(question, expectedFact, answerText, arm) {
  const raw = await chat({
    model: MODELS.judge,
    max_tokens: 250,
    label: `judge.${arm}`,
    system: `You grade whether an answer preserves a specific expected fact.

Score 1 if the answer accurately conveys the expected fact (paraphrasing is fine).
Score 0 if the fact is missing, vague, contradictory, or invented.

You may briefly explain your reasoning, but the LAST CHARACTER of your response MUST be a single digit: 1 or 0. Nothing after the digit.`,
    user: `QUESTION: ${question}\nEXPECTED FACT: ${expectedFact}\nANSWER:\n${answerText}\n\nGrade it. End with the digit 1 or 0.`,
  });
  const cleaned = stripThink(raw).trim();
  // Find the last 0 or 1 in the response.
  const m = cleaned.match(/[01](?!.*[01])/s);
  if (!m) {
    console.warn(`      ⚠ judge gave no digit; defaulting to 0. raw: ${cleaned.slice(-120)}`);
    return { score: 0, reason: "judge-parse-failed" };
  }
  return { score: Number(m[0]), reason: cleaned.slice(0, 120).replace(/\n/g, " ") };
}

// ---------- Experiment ----------
async function run() {
  console.log(`\n=== BUTTERFLY FALSIFIER (local) ===\n`);
  console.log(`Main endpoint  : ${mainClient.baseURL}`);
  if (taggerClient !== mainClient) {
    console.log(`Tagger endpoint: ${taggerClient.baseURL}  (split off)`);
  }
  console.log(`Models  : tagger=${MODELS.tagger}  rebuild=${MODELS.rebuild}  answer=${MODELS.answer}  judge=${MODELS.judge}`);
  console.log(`Transcript: ${transcript.length} messages, ~${transcriptTokens} tokens.\n`);

  console.log(`[1/3] Tagging (${MODELS.tagger})`);
  const tags = await tagAll(transcript);
  const counts = tags.reduce((acc, t) => ((acc[t.action] = (acc[t.action] || 0) + 1), acc), {});
  console.log(`      keep:${counts.keep || 0}  summarize:${counts.summarize || 0}  melt:${counts.melt || 0}\n`);

  console.log(`[2/3] Building chrysalis (${MODELS.rebuild})...`);
  const liveGoal = "Just shipped a fix for a flaky JWT-expiration test; user may follow up about the change or related items.";
  const rebuilt = await chrysalis(transcript, tags, liveGoal);
  const rebuiltTokens = tokens(rebuilt);
  const shrink = ((1 - rebuiltTokens / transcriptTokens) * 100).toFixed(1);
  console.log(`      rebuilt: ~${rebuiltTokens} tokens  (${shrink}% smaller than original)`);

  writeFileSync("./rebuilt-context.txt", rebuilt);
  writeFileSync("./tags.json", JSON.stringify(tags, null, 2));
  console.log(`      wrote ./rebuilt-context.txt and ./tags.json\n`);

  // Last-N baseline: truncate to match butterfly's token budget, keep the suffix.
  const lastNMsgs = lastNMessages(transcript, rebuiltTokens);
  const lastNText = transcriptAsText(lastNMsgs);
  const lastNTokens = tokens(lastNText);
  console.log(`      last-N baseline: keep final ${lastNMsgs.length}/${transcript.length} msgs = ~${lastNTokens} tokens\n`);

  console.log(`[3/3] Running ${needles.length} needle questions through 3 arms (baseline / butterfly / lastN)\n`);
  const fullText = transcriptAsText(transcript);
  const rows = [];

  for (const n of needles) {
    console.log(`--- ${n.id} [${n.category}] (${n.importance}) ---`);
    console.log(`Q: ${n.question}`);

    const ansBaseline  = await answer(fullText,  n.question, "baseline");
    const ansButterfly = await answer(rebuilt,   n.question, "butterfly");
    const ansLastN     = await answer(lastNText, n.question, "lastN");
    console.log(`\nBASELINE : ${ansBaseline.replace(/\n/g, " ").slice(0, 200)}${ansBaseline.length > 200 ? "..." : ""}`);
    console.log(`BUTTERFLY: ${ansButterfly.replace(/\n/g, " ").slice(0, 200)}${ansButterfly.length > 200 ? "..." : ""}`);
    console.log(`LAST-N   : ${ansLastN.replace(/\n/g, " ").slice(0, 200)}${ansLastN.length > 200 ? "..." : ""}`);

    const sB = await judge(n.question, n.fact, ansBaseline,  "baseline");
    const sF = await judge(n.question, n.fact, ansButterfly, "butterfly");
    const sL = await judge(n.question, n.fact, ansLastN,     "lastN");
    console.log(`SCORE: baseline=${sB.score}  butterfly=${sF.score}  lastN=${sL.score}\n`);
    rows.push({
      id: n.id, category: n.category, importance: n.importance,
      baseline: sB.score, butterfly: sF.score, lastN: sL.score,
    });
  }

  console.log(`\n=== RESULTS ===\n`);
  console.log(`Needle  Category       Imp    Baseline  Butterfly  LastN`);
  console.log(`------  -------------  -----  --------  ---------  -----`);
  for (const r of rows) {
    console.log(`${r.id.padEnd(7)} ${r.category.padEnd(14)} ${r.importance.padEnd(6)} ${String(r.baseline).padEnd(9)} ${String(r.butterfly).padEnd(10)} ${r.lastN}`);
  }

  // Per-category aggregation.
  const byCat = {};
  for (const r of rows) {
    (byCat[r.category] ??= { baseline: 0, butterfly: 0, lastN: 0, n: 0 });
    byCat[r.category].baseline  += r.baseline;
    byCat[r.category].butterfly += r.butterfly;
    byCat[r.category].lastN     += r.lastN;
    byCat[r.category].n         += 1;
  }
  console.log(`\nPer category:`);
  for (const [cat, s] of Object.entries(byCat)) {
    console.log(`  ${cat.padEnd(14)} baseline=${s.baseline}/${s.n}  butterfly=${s.butterfly}/${s.n}  lastN=${s.lastN}/${s.n}`);
  }

  const bSum = rows.reduce((n, r) => n + r.baseline,  0);
  const fSum = rows.reduce((n, r) => n + r.butterfly, 0);
  const lSum = rows.reduce((n, r) => n + r.lastN,     0);
  console.log(`\nTotal:  baseline ${bSum}/${rows.length}   butterfly ${fSum}/${rows.length}   lastN ${lSum}/${rows.length}`);
  console.log(`Tokens: full ${transcriptTokens} | butterfly ${rebuiltTokens} (${shrink}%) | lastN ${lastNTokens}\n`);

  const resultPath = `./results-seed-${RUN_SEED}.json`;
  writeFileSync(resultPath, JSON.stringify({
    seed: RUN_SEED,
    transcript_messages: transcript.length,
    transcript_tokens: transcriptTokens,
    rebuilt_tokens: rebuiltTokens,
    lastN_tokens: lastNTokens,
    shrink_pct: Number(shrink),
    tag_distribution: counts,
    needles: rows,
    totals: { baseline: bSum, butterfly: fSum, lastN: lSum, of: rows.length },
    per_category: byCat,
  }, null, 2));
  console.log(`      wrote ${resultPath}\n`);
}

run().catch(e => { console.error("\nFAILED:", e.message); process.exit(1); });
