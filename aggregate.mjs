// Roll up results-seed-*.json into a mean / per-seed / per-category view.
// Run after `SEED=1,2,3 node experiment.mjs` has produced the per-seed files.

import { readdirSync, readFileSync } from "node:fs";

const files = readdirSync(".").filter(f => /^results-seed-\d+\.json$/.test(f)).sort();
if (files.length === 0) {
  console.error("no results-seed-*.json found");
  process.exit(1);
}

const runs = files.map(f => JSON.parse(readFileSync(f, "utf8")));
const arms = ["baseline", "butterfly", "lastN"];
const seeds = runs.map(r => r.seed);

console.log(`\n=== AGGREGATE (${runs.length} seed${runs.length > 1 ? "s" : ""}: ${seeds.join(", ")}) ===\n`);

// Per-seed totals.
console.log(`Per-seed totals (preserved / total):`);
console.log(`seed  baseline  butterfly  lastN  | compression`);
console.log(`----  --------  ---------  -----  | -----------`);
for (const r of runs) {
  const t = r.totals;
  const comp = `${r.rebuilt_tokens}/${r.transcript_tokens} (${r.shrink_pct}%↓)`;
  console.log(`${String(r.seed).padEnd(5)} ${`${t.baseline}/${t.of}`.padEnd(9)} ${`${t.butterfly}/${t.of}`.padEnd(10)} ${`${t.lastN}/${t.of}`.padEnd(6)} | ${comp}`);
}

// Mean preservation rate per arm.
console.log(`\nMean preservation rate across seeds:`);
for (const arm of arms) {
  const scores = runs.map(r => r.totals[arm] / r.totals.of);
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  console.log(`  ${arm.padEnd(10)} ${(mean * 100).toFixed(1)}%   (min ${(min * 100).toFixed(0)}%, max ${(max * 100).toFixed(0)}%)`);
}

// Per-category preservation, averaged over seeds.
// Needle-level: for each needle id, count hits per arm across seeds.
const byNeedle = {};
for (const r of runs) {
  for (const n of r.needles) {
    const key = `${n.id}|${n.category}|${n.importance}`;
    (byNeedle[key] ??= { id: n.id, category: n.category, importance: n.importance, baseline: 0, butterfly: 0, lastN: 0, seen: 0 });
    byNeedle[key].baseline  += n.baseline;
    byNeedle[key].butterfly += n.butterfly;
    byNeedle[key].lastN     += n.lastN;
    byNeedle[key].seen      += 1;
  }
}

console.log(`\nPer-needle preservation (hits / seeds):`);
console.log(`Needle  Category       Imp    Baseline  Butterfly  LastN`);
console.log(`------  -------------  -----  --------  ---------  -----`);
for (const k of Object.keys(byNeedle)) {
  const r = byNeedle[k];
  console.log(`${r.id.padEnd(7)} ${r.category.padEnd(14)} ${r.importance.padEnd(6)} ${`${r.baseline}/${r.seen}`.padEnd(9)} ${`${r.butterfly}/${r.seen}`.padEnd(10)} ${`${r.lastN}/${r.seen}`}`);
}

// Per-category rollup.
const byCat = {};
for (const n of Object.values(byNeedle)) {
  (byCat[n.category] ??= { baseline: 0, butterfly: 0, lastN: 0, seen: 0 });
  byCat[n.category].baseline  += n.baseline;
  byCat[n.category].butterfly += n.butterfly;
  byCat[n.category].lastN     += n.lastN;
  byCat[n.category].seen      += n.seen;
}
console.log(`\nPer-category (hits / total evaluations):`);
for (const [cat, s] of Object.entries(byCat)) {
  console.log(`  ${cat.padEnd(14)} baseline=${s.baseline}/${s.seen}  butterfly=${s.butterfly}/${s.seen}  lastN=${s.lastN}/${s.seen}`);
}

console.log();
