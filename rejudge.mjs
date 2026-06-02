// Re-judge transgen results with a 3-tier rubric instead of binary 0/1.
// The binary judge undercounts wins where butterfly preserved the load-bearing
// bug identification but mis-stated a consequence. Rubric:
//   2 = HIT     — answer accurately conveys the full expected fact
//   1 = PARTIAL — answer preserves the load-bearing identification (file,
//                 function, code, mechanism) but mis-states a detail or consequence
//   0 = MISS    — answer doesn't identify the bug at all (says "no info" or invents)

import OpenAI from "openai";
import { readFileSync, writeFileSync } from "node:fs";

const client = new OpenAI({ baseURL: process.env.BRIDGE_URL || "http://localhost:3002/v1", apiKey: "bridge" });
const MODEL = process.env.JUDGE_MODEL || "sonnet";
const NO_THINK = "/no_think";
const TIMEOUT_MS = Number(process.env.CALL_TIMEOUT_MS) || 120000;

const stripThink = (s) => s.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();

async function chat({ system, user, label }) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), TIMEOUT_MS);
  const t0 = Date.now();
  console.log(`[${new Date().toISOString().slice(11,19)}] ${label} start`);
  try {
    const r = await client.chat.completions.create({
      model: MODEL, max_tokens: 250, temperature: 0,
      messages: [
        { role: "system", content: `${NO_THINK}\n\n${system}` },
        { role: "user", content: user },
      ],
    }, { signal: ac.signal });
    clearTimeout(timer);
    const out = stripThink(r.choices[0].message.content || "");
    console.log(`[${new Date().toISOString().slice(11,19)}] ${label} done in ${((Date.now()-t0)/1000).toFixed(1)}s -> ${out.slice(-80).replace(/\n/g," ")}`);
    return out;
  } catch (e) {
    clearTimeout(timer);
    console.log(`[${new Date().toISOString().slice(11,19)}] ${label} FAILED: ${e.message?.slice(0,80)}`);
    throw e;
  }
}

async function judgeRubric(question, fact, answer, label) {
  const raw = await chat({
    label,
    system: `You grade an LLM's answer against an expected fact using a 3-tier rubric.

Score 2 (HIT): the answer accurately conveys the full expected fact. Paraphrasing fine.
Score 1 (PARTIAL): the answer preserves the load-bearing identification (file, function, exact code, mechanism, owner — the parts an engineer would need to act) but mis-states or omits a detail/consequence.
Score 0 (MISS): the answer fails to identify the bug — it says "no info in context", invents something wrong, or is too vague to be actionable.

You may briefly explain. The LAST CHARACTER of your response MUST be a single digit: 0, 1, or 2. Nothing after.`,
    user: `QUESTION: ${question}\nEXPECTED FACT: ${fact}\nANSWER:\n${answer}\n\nGrade. End with 0, 1, or 2.`,
  });
  const cleaned = stripThink(raw).trim();
  const m = cleaned.match(/[012](?!.*[012])/s);
  return m ? Number(m[0]) : 0;
}

async function run() {
  const data = JSON.parse(readFileSync("./transgen-results.json", "utf8"));
  console.log(`\n=== RE-JUDGE with 3-tier rubric ===\n`);
  console.log(`Tasks to re-judge: ${data.rows.length}\n`);

  const newRows = [];
  for (const r of data.rows) {
    if (!r.butterfly_answer || !r.lastN_answer) {
      newRows.push({ ...r, butterfly_rubric: null, lastN_rubric: null });
      continue;
    }
    console.log(`\n--- ${r.id} ${r.topic.slice(0,55)} ---`);
    const bScore = await judgeRubric(r.id, r.needle_fact, r.butterfly_answer, `${r.id}.butterfly`);
    const lScore = await judgeRubric(r.id, r.needle_fact, r.lastN_answer,     `${r.id}.lastN`);
    console.log(`  RUBRIC: butterfly=${bScore} (was ${r.butterfly})  lastN=${lScore} (was ${r.lastN})`);
    newRows.push({ ...r, butterfly_rubric: bScore, lastN_rubric: lScore });
  }

  console.log(`\n\n=== RUBRIC RESULTS ===\n`);
  console.log(`Task    Topic                                          Binary B/L   Rubric B/L`);
  console.log(`------  -------------------------------------------    ----------   ----------`);
  for (const r of newRows) {
    const topic = (r.topic || "").slice(0,44).padEnd(45);
    const bin   = `${r.butterfly}/${r.lastN}`.padEnd(11);
    const rub   = `${r.butterfly_rubric}/${r.lastN_rubric}`;
    console.log(`${r.id.padEnd(7)} ${topic} ${bin}  ${rub}`);
  }
  const valid = newRows.filter(r => r.butterfly_rubric !== null && r.lastN_rubric !== null);
  const bSum = valid.reduce((a,r) => a + r.butterfly_rubric, 0);
  const lSum = valid.reduce((a,r) => a + r.lastN_rubric, 0);
  const maxScore = valid.length * 2;
  const bAny = valid.filter(r => r.butterfly_rubric > 0).length;
  const lAny = valid.filter(r => r.lastN_rubric > 0).length;
  console.log(`\nValid: ${valid.length}/${newRows.length}`);
  console.log(`Sum (max ${maxScore}): butterfly ${bSum}  lastN ${lSum}`);
  console.log(`Any preservation (>0): butterfly ${bAny}/${valid.length}  lastN ${lAny}/${valid.length}`);
  console.log(`Mean rubric score:  butterfly ${(bSum/valid.length).toFixed(2)}/2  lastN ${(lSum/valid.length).toFixed(2)}/2`);

  writeFileSync("./transgen-rubric-results.json", JSON.stringify({
    n_generations: data.n_generations,
    target_tokens: data.target_tokens,
    binary: { butterfly: data.butterfly_survival, lastN: data.lastN_survival },
    rubric_mean: { butterfly: bSum/valid.length, lastN: lSum/valid.length },
    rubric_any_preservation: { butterfly: bAny/valid.length, lastN: lAny/valid.length },
    rows: newRows,
  }, null, 2));
  console.log(`\nWrote ./transgen-rubric-results.json`);
}

run().catch(e => { console.error("FAILED:", e.message); process.exit(1); });
