// Synthetic debugging session with 3 planted "needles" of varying importance.
// The experiment will check whether a metamorphosed (rebuilt) context preserves
// each needle as well as the full transcript does.

export const transcript = [
  { role: "user", content: "Hey, our CI keeps failing on auth/session.test.ts but it passes locally. Can you help me figure out why?" },
  { role: "assistant", content: "Sure. Can you share the test file and the CI failure output?" },

  { role: "user", content: "Here's the test:\n\n```ts\nimport { issueToken, decodeToken } from '@/lib/jwt'\n\ntest('token expires after 1h', () => {\n  const t = issueToken({ sub: 'u_1' })\n  const decoded = decodeToken(t)\n  expect(decoded.exp).toBe(Math.floor(Date.now() / 1000) + 3600)\n})\n```\n\nAnd the CI log says:\n`Expected: 1735689600  Received: 1735689599`" },
  { role: "assistant", content: "The numbers are off by one second. That smells like a timing race rather than a logic bug — `Date.now()` is read twice (once inside `issueToken`, once in the assertion), and on a slow CI machine the second read can land in the next second. Want me to look at `issueToken` to confirm?" },

  { role: "user", content: "yeah it's at lib/jwt.ts" },
  { role: "assistant", content: "Read lib/jwt.ts (87 lines). Confirmed: `issueToken` calls `Math.floor(Date.now()/1000) + 3600` and stores it as `exp`. So the assertion re-computes `Date.now()` in a different millisecond. That's the bug — the test is racing the clock." },

  { role: "user", content: "ok but why does it only fail in CI and never locally?" },
  { role: "assistant", content: "Two reasons make CI more likely to lose the race: (1) CI runners share CPU and stall mid-test more often than your laptop, and (2) the failure window is narrow — only when the two `Date.now()` calls straddle a second boundary. On a fast local machine you almost never cross one." },

  { role: "user", content: "huh ok. before I fix it — I want to also note something I just spotted. In next.config.mjs line 47 someone wrote `experiemental:` (typo, should be `experimental:`). Whole config block is being silently ignored. Not related to this bug but I'll fix it in the same PR." },
  { role: "assistant", content: "Good catch. That typo means whatever flags are in that block are no-ops. Worth a separate commit in the same PR so the diff is clear." },

  { role: "user", content: "right. back to the test — what's the fix?" },
  { role: "assistant", content: "Two options. Cheap fix: capture `Date.now()` once before calling `issueToken`, then assert against that captured value. Better fix: have `issueToken` accept an optional `now` parameter so tests can inject a fixed clock. The second is the proper one — it makes the function deterministic under test." },

  { role: "user", content: "let's do the proper fix. show me the change to lib/jwt.ts" },
  { role: "assistant", content: "Edit lib/jwt.ts: change `export function issueToken(payload)` to `export function issueToken(payload, now = Date.now())` and replace the internal `Date.now()` with `now`. Then in the test: `const now = Date.now(); const t = issueToken({sub:'u_1'}, now); expect(decoded.exp).toBe(Math.floor(now/1000)+3600)`." },

  { role: "user", content: "applied. running the test... passes 50/50 in a loop. nice." },
  { role: "assistant", content: "Great. Worth adding the loop itself as a quick sanity test — `for (let i=0;i<50;i++)` around it — so we'd catch a regression of this exact race in CI." },

  { role: "user", content: "yeah added. one more thing — `decodeToken` is in the shared @company/jwt-utils package, not in our repo. If we ever want to change the exp semantics we need to coordinate with the auth team. Sarah owns that package, she's in #auth-platform on slack. Worth noting since 4 other services depend on it." },
  { role: "assistant", content: "Noted. So `decodeToken` is owned externally by Sarah / #auth-platform, and shape changes need their sign-off. For this PR we're not touching it, just `issueToken`, so we're fine." },

  { role: "user", content: "while we're here — should we be using vitest instead of jest? the rest of the company moved over last quarter" },
  { role: "assistant", content: "Probably yes long-term, but it's a separate migration and would balloon this PR. Worth a follow-up ticket. For now keep jest since the test infra is already wired for it." },

  { role: "user", content: "oh wait — I just remembered. Dan emailed the team yesterday: all new tests MUST be vitest starting next sprint, it's mandatory company-wide now. So vitest isn't optional anymore, it's required. I was wrong before." },
  { role: "assistant", content: "Thanks for the correction — so vitest is mandatory from next sprint, not a nice-to-have. Do you want to expand this PR to switch session.test.ts to vitest, or do it as an immediate follow-up PR?" },

  { role: "user", content: "follow-up PR, right after this one merges. keeping this one focused on the flake fix." },
  { role: "assistant", content: "Sounds right. Tracking: (a) this PR lands the race fix on jest, (b) follow-up PR migrates session.test.ts to vitest before next sprint per the mandate." },

  { role: "user", content: "agreed. ok let me also look at the snapshot tests in the same folder to make sure none of them have similar timing issues" },
  { role: "assistant", content: "Read auth/__snapshots__/. Three snapshot files, none reference `Date.now()` or any other dynamic source. They're keyed off static fixture inputs. Safe." },

  { role: "user", content: "also — I pinged her about the exp change. She said it's fine since we're only touching issueToken, not decodeToken semantics. So we're clear on that side." },
  { role: "assistant", content: "Great, that's sign-off from the jwt-utils owner. No coordination blockers for this PR." },

  { role: "user", content: "ok. and let me re-read the final diff." },
  { role: "assistant", content: "Diff summary:\n- lib/jwt.ts: `issueToken(payload, now = Date.now())` — accepts injected clock\n- auth/session.test.ts: capture `now` before calling, assert against it, wrap in 50-iteration loop\n- next.config.mjs: fix `experiemental` -> `experimental` typo on line 47\nThree files, ~12 lines changed." },

  { role: "user", content: "lgtm. pushing." },
  { role: "assistant", content: "Sounds good. If CI flakes again on something similar, the pattern to look for is any `Date.now()` (or `performance.now()`, `new Date()`) called twice across an assertion boundary." },
];

// Ground-truth needles — never shown to the tagger or chrysalis.
// Used only by the experiment to grade what got preserved.
//
// Categories stress different failure modes:
//   - simple:        fact stated once, plain text
//   - contradiction: claim asserted, later corrected; correction is ground truth
//   - coreference:   referent far from antecedent (pronoun chain); needs resolution
export const needles = [
  {
    id: "N1",
    category: "simple",
    importance: "HIGH",
    fact: "The root cause was that Date.now() was being called twice (once inside issueToken, once in the assertion), and on slow CI the two calls could land in different seconds — a clock race, not a logic bug.",
    question: "Quick reminder — what was the actual root cause of the flaky session test?",
  },
  {
    id: "N2",
    category: "simple",
    importance: "MEDIUM",
    fact: "decodeToken lives in the shared @company/jwt-utils package, owned by Sarah on the auth team, reachable in #auth-platform on Slack. Changes to it require coordinating with her because 4 other services depend on it.",
    question: "If I need to change decodeToken's behavior later, who do I coordinate with and where?",
  },
  {
    id: "N3",
    category: "simple",
    importance: "LOW",
    fact: "There was an unrelated typo in next.config.mjs at line 47: `experiemental:` should be `experimental:`. The whole config block under it was being silently ignored.",
    question: "Was there anything unrelated to the main bug I noted I should also clean up?",
  },
  {
    id: "N4",
    category: "contradiction",
    importance: "HIGH",
    fact: "Vitest migration is MANDATORY (not optional) — per Dan's email, all new tests must be vitest starting next sprint, company-wide. The initial assumption that vitest was a long-term nice-to-have was corrected and is no longer accurate.",
    question: "Is the vitest migration optional or required, and on what timeline?",
  },
  {
    id: "N5",
    category: "coreference",
    importance: "MEDIUM",
    fact: "Yes, we have sign-off on the expiration change. Sarah (owner of @company/jwt-utils, on #auth-platform) confirmed it's fine since we're only touching issueToken, not decodeToken semantics. Note: the user referred to Sarah only as 'her'/'she' in the later message, so answering correctly requires linking the pronoun back to Sarah introduced earlier.",
    question: "Did we get sign-off from the jwt-utils owner on the expiration change, and on what condition?",
  },
];
