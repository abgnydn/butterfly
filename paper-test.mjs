// paper-test: the one direct experiment to prove butterfly.
//
// Design: 10 long debugging conversations, each with a buried root-cause
// "needle" near the start and a final "what was the root cause?" question.
// Compress the conversation two ways, ask the question, judge the answer.
//
//   butterfly = qwen3:0.6b per-message tagger + Claude chrysalis
//   lastN     = keep the suffix of messages that fits in butterfly's token budget
//
// Primary metric: task completion rate per arm. The claim under test:
// "online importance-aware compaction beats naive recency truncation at
// matched compression budget on tasks where the needle lives early."
//
// If butterfly clearly wins → expand to 30+ tasks across domains.
// If it loses → diagnose before building more.

import OpenAI from "openai";
import { writeFileSync } from "node:fs";

const localClient  = new OpenAI({ baseURL: "http://localhost:11434/v1", apiKey: "ollama" });
const bridgeClient = new OpenAI({ baseURL: "http://localhost:3001/v1",  apiKey: "bridge" });

const MODELS = {
  generator: "claude-via-cli",
  tagger:    "qwen3:0.6b",
  rebuild:   "claude-via-cli",
  answer:    "claude-via-cli",
  judge:     "claude-via-cli",
};

const NO_THINK    = "/no_think";
const TIMEOUT_MS  = Number(process.env.CALL_TIMEOUT_MS) || 240000;
const N_TASKS     = Number(process.env.N_TASKS)         || 10;

const ts = () => new Date().toISOString().slice(11, 19);
const tokens = (s) => Math.ceil(s.length / 4);
const stripThink = (s) => s.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();

function extractJSON(raw) {
  let t = stripThink(raw).replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/i, "").trim();
  // Prefer whichever outer bracket comes FIRST — don't let a nested array
  // inside an object (e.g. "messages":[...]) get picked over the outer object.
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

async function chat({ model, system, user, max_tokens = 800, client = bridgeClient, label }) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), TIMEOUT_MS);
  const t0 = Date.now();
  console.log(`[${ts()}] ${label} start (${model})`);
  try {
    const r = await client.chat.completions.create({
      model, max_tokens, temperature: 0.2,
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

const msgsAsText = (msgs) => msgs.map(m => `[${m.role}]\n${m.content}`).join("\n\n");

// ---------- Task generation ----------

// Pre-defined bug topics so tasks are varied without relying on Claude
// to diversify on its own. Add/edit to change the eval mix.
const BUG_TOPICS = [
  "timing race — same clock function called twice across an assertion boundary",
  "off-by-one error in a loop bound (using <= where < was intended, or vice versa)",
  "missing await on an async function call — promise resolves after assertion",
  "shared mutable state — module-level singleton accumulating across tests",
  "unicode normalization mismatch — NFC vs NFD comparison failing",
  "timezone offset bug — date constructed in local time but compared in UTC",
  "memory leak — closure capturing large object inside a setInterval that's never cleared",
  "event listener never removed on component unmount, firing on stale instance",
  "null vs undefined check confusion — `if (x !== null)` letting undefined through",
  "circular import causing temporal dead zone — module A imports B which imports A",
];

async function generateOneTask(idx, topic) {
  const id = `T${String(idx + 1).padStart(2, "0")}`;
  const raw = await chat({
    model: MODELS.generator,
    client: bridgeClient,
    max_tokens: 3000,
    label: `gen ${id}`,
    system: `You generate ONE synthetic debugging conversation transcript.

Output STRICT JSON with these exact top-level keys — nothing else, no preamble, no markdown fences, no explanation:
{"id":"${id}","topic":"...","messages":[{"role":"user|assistant","content":"..."},...],"needle_fact":"...","question":"Quick reminder — what was the actual root cause?"}

Constraints:
- 18-22 messages. Plant the specific root cause at message index 4-9.
- Concrete file/function/mechanism in needle_fact (not generic).
- Realistic code snippets where useful.
- First character of your response MUST be '{'. Last character MUST be '}'.`,
    user: `Bug topic: ${topic}\n\nGenerate.`,
  });
  try {
    const task = extractJSON(raw);
    if (!task.messages || !task.needle_fact || !task.question) {
      writeFileSync(`/tmp/debug-raw-${id}.txt`, raw);
      throw new Error(`malformed task ${id}: missing fields (saved raw to /tmp/debug-raw-${id}.txt)`);
    }
    task.id = id;
    return task;
  } catch (e) {
    if (!e.message.includes("saved raw")) {
      writeFileSync(`/tmp/debug-raw-${id}.txt`, raw);
      throw new Error(`${e.message} — saved raw to /tmp/debug-raw-${id}.txt`);
    }
    throw e;
  }
}

async function generateTasks(n) {
  const tasks = [];
  for (let i = 0; i < n; i++) {
    const topic = BUG_TOPICS[i % BUG_TOPICS.length];
    try {
      tasks.push(await generateOneTask(i, topic));
    } catch (e) {
      console.log(`  ⚠ task ${i + 1} generation failed: ${e.message?.slice(0, 100)}`);
    }
  }
  if (tasks.length === 0) throw new Error("no tasks generated");
  return tasks;
}

// ---------- Butterfly compression ----------

async function tagOne(message, idx) {
  try {
    const raw = await chat({
      model: MODELS.tagger,
      client: localClient,
      max_tokens: 250,
      label: `tag #${idx}`,
      system: `Tag ONE message for context compaction. Reply with one JSON object only:
{"action": "keep" | "summarize" | "melt", "importance": 0.0-1.0, "reason": "<10 words"}

keep = irreplaceable atom (root cause, owner, file:line, decision)
summarize = substantive but a one-line gist suffices
melt = greetings, acks, dead-end tangents

Examples:
"Sure. Share the file." -> {"action":"melt","importance":0.05,"reason":"ack"}
"Root cause: Date.now() race." -> {"action":"keep","importance":0.95,"reason":"root cause"}
"Read 87 lines, confirmed bug." -> {"action":"summarize","importance":0.4,"reason":"verbose, one fact"}`,
      user: `Message #${idx} (role=${message.role}):\n---\n${message.content}\n---\nReturn JSON only.`,
    });
    return extractJSON(raw);
  } catch (e) {
    return { action: "keep", importance: 0.5, reason: "tag-parse-failed" };
  }
}

async function butterflyCompress(messages) {
  const tags = [];
  for (let i = 0; i < messages.length; i++) tags.push(await tagOne(messages[i], i));
  const tagged = messages.map((m, i) => `[#${i} role=${m.role} action=${tags[i].action} importance=${tags[i].importance}]\n${m.content}`).join("\n\n");

  const rebuilt = await chat({
    model: MODELS.rebuild,
    client: bridgeClient,
    max_tokens: 1500,
    label: "chrysalis",
    system: `You rebuild a tagged conversation transcript into a small coherent context the agent will resume from.

Rules:
- KEEP messages: preserve every load-bearing fact, name, file:line, decision, code snippet.
- SUMMARIZE messages: condense to one terse line each.
- MELT messages: drop entirely — do not include or mention.
- Plain prose with light structure.
- Output ONLY the rebuilt context. No preamble, no meta-commentary.`,
    user: `TAGGED TRANSCRIPT:\n\n${tagged}`,
  });
  return rebuilt;
}

// ---------- LastN compression ----------

function lastNToBudget(messages, targetTokens) {
  const out = [];
  let acc = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const t = tokens(messages[i].content);
    if (acc + t > targetTokens && out.length > 0) break;
    out.unshift(messages[i]);
    acc += t;
  }
  return msgsAsText(out);
}

// ---------- Answer + judge ----------

async function answerQuestion(ctx, q, arm) {
  return chat({
    model: MODELS.answer,
    client: bridgeClient,
    max_tokens: 300,
    label: `answer.${arm}`,
    system: `You continue a prior conversation. Answer the follow-up using ONLY the prior context. If a fact isn't there, say so plainly — do not invent. Be concise.`,
    user: `PRIOR CONTEXT:\n\n${ctx}\n\n---\n\nFOLLOW-UP: ${q}`,
  });
}

async function judge(question, fact, ans, arm) {
  const raw = await chat({
    model: MODELS.judge,
    client: bridgeClient,
    max_tokens: 200,
    label: `judge.${arm}`,
    system: `You grade whether an answer accurately conveys an expected fact.

Score 1 if the answer accurately conveys the EXPECTED FACT (paraphrasing fine, exact wording not required).
Score 0 if missing, vague, contradictory, or invented.

The LAST CHARACTER of your response MUST be a single digit: 1 or 0. Nothing after.`,
    user: `QUESTION: ${question}\nEXPECTED FACT: ${fact}\nANSWER:\n${ans}\n\nGrade. End with 1 or 0.`,
  });
  const m = stripThink(raw).trim().match(/[01](?!.*[01])/s);
  return m ? Number(m[0]) : 0;
}

// ---------- Main ----------

async function run() {
  console.log(`\n=== PAPER TEST: butterfly vs lastN on agent task completion ===\n`);
  console.log(`Generating ${N_TASKS} tasks via Claude...`);
  const tasks = await generateTasks(N_TASKS);
  writeFileSync("./paper-test-tasks.json", JSON.stringify(tasks, null, 2));
  console.log(`  -> wrote ${tasks.length} tasks to ./paper-test-tasks.json\n`);
  if (process.env.GEN_ONLY) {
    console.log(`GEN_ONLY=1 — exiting after task generation.`);
    return;
  }

  const rows = [];
  for (let i = 0; i < tasks.length; i++) {
    const t = tasks[i];
    console.log(`\n--- ${t.id} ${t.topic} ---`);
    const fullText = msgsAsText(t.messages);
    const fullTokens = tokens(fullText);

    let butterflyCtx, lastNCtx, butterflyTokens, lastNTokens;
    try {
      butterflyCtx = await butterflyCompress(t.messages);
      butterflyTokens = tokens(butterflyCtx);
      lastNCtx = lastNToBudget(t.messages, butterflyTokens);
      lastNTokens = tokens(lastNCtx);
      console.log(`  full=${fullTokens}t  butterfly=${butterflyTokens}t  lastN=${lastNTokens}t`);
    } catch (e) {
      console.log(`  COMPRESS FAILED: ${e.message}`);
      rows.push({ id: t.id, topic: t.topic, butterfly: null, lastN: null, error: e.message });
      continue;
    }

    let bAns, lAns, bScore = null, lScore = null;
    try {
      bAns = await answerQuestion(butterflyCtx, t.question, "butterfly");
      lAns = await answerQuestion(lastNCtx,     t.question, "lastN");
      bScore = await judge(t.question, t.needle_fact, bAns, "butterfly");
      lScore = await judge(t.question, t.needle_fact, lAns, "lastN");
    } catch (e) {
      console.log(`  ANSWER/JUDGE FAILED: ${e.message}`);
    }
    console.log(`  SCORE: butterfly=${bScore}  lastN=${lScore}`);
    rows.push({
      id: t.id, topic: t.topic,
      full_tokens: fullTokens, butterfly_tokens: butterflyTokens, lastN_tokens: lastNTokens,
      butterfly: bScore, lastN: lScore,
      butterfly_answer: bAns?.slice(0, 200), lastN_answer: lAns?.slice(0, 200),
      needle_fact: t.needle_fact,
    });
  }

  console.log(`\n\n=== RESULTS ===\n`);
  console.log(`Task    Topic                                          Butterfly  LastN`);
  console.log(`------  -------------------------------------------    ---------  -----`);
  for (const r of rows) {
    const topic = (r.topic || "").slice(0, 44).padEnd(45);
    console.log(`${r.id.padEnd(7)} ${topic} ${String(r.butterfly).padEnd(10)} ${r.lastN}`);
  }
  const valid = rows.filter(r => r.butterfly !== null && r.lastN !== null);
  const bSum = valid.reduce((a, r) => a + r.butterfly, 0);
  const lSum = valid.reduce((a, r) => a + r.lastN, 0);
  console.log(`\nTotal valid runs: ${valid.length}/${rows.length}`);
  console.log(`Task completion: butterfly ${bSum}/${valid.length} (${(100*bSum/valid.length).toFixed(0)}%)   lastN ${lSum}/${valid.length} (${(100*lSum/valid.length).toFixed(0)}%)`);

  const wins = valid.filter(r => r.butterfly > r.lastN).length;
  const losses = valid.filter(r => r.butterfly < r.lastN).length;
  const ties = valid.filter(r => r.butterfly === r.lastN).length;
  console.log(`Head-to-head: butterfly wins ${wins}, loses ${losses}, ties ${ties}`);

  writeFileSync("./paper-test-results.json", JSON.stringify({
    n_tasks: rows.length,
    valid: valid.length,
    butterfly_completion: bSum / valid.length,
    lastN_completion: lSum / valid.length,
    head_to_head: { wins, losses, ties },
    rows,
  }, null, 2));
  console.log(`\nWrote ./paper-test-results.json`);
}

run().catch(e => { console.error("FAILED:", e.message); process.exit(1); });
