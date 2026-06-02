// claude-bridge: HTTP shim exposing the `claude -p` CLI as an OpenAI-compatible
// /v1/chat/completions endpoint. Uses your Max-plan auth via subprocess; no
// ANTHROPIC_API_KEY needed.
//
// Start:  node claude-bridge.mjs   (defaults to port 3001)
// Use:    BASE_URL=http://localhost:3001/v1  REBUILD_MODEL=claude-via-cli  ...

import http from "node:http";
import { spawn, execSync } from "node:child_process";

const PORT       = Number(process.env.BRIDGE_PORT)       || 3001;
const TIMEOUT_MS = Number(process.env.BRIDGE_TIMEOUT_MS) || 180000;

// Resolve absolute path once at startup. spawn() doesn't always find PATH-only
// commands reliably under nvm; absolute path avoids surprises.
const CLAUDE_BIN = process.env.CLAUDE_BIN || (() => {
  try { return execSync("which claude", { encoding: "utf8" }).trim(); }
  catch { return "claude"; }
})();
console.log(`Using claude binary: ${CLAUDE_BIN}`);

// Intermittent ENOENT on rapid-fire spawns — first call works, second fails,
// fourth succeeds again. Looks like a brief window where the nvm-shimmed
// binary becomes unreachable. Retry with backoff fixes it cleanly.
function spawnClaudeOnce(prompt) {
  return new Promise((resolve, reject) => {
    const proc = spawn(CLAUDE_BIN, ["-p", "--output-format", "text"], {
      stdio: ["pipe", "pipe", "pipe"],
      env: process.env,
    });
    let out = "";
    let err = "";
    const killer = setTimeout(() => {
      proc.kill("SIGTERM");
      reject(new Error(`claude CLI timeout after ${TIMEOUT_MS}ms`));
    }, TIMEOUT_MS);
    proc.stdout.on("data", c => (out += c));
    proc.stderr.on("data", c => (err += c));
    proc.on("error", reject);
    proc.on("close", code => {
      clearTimeout(killer);
      if (code === 0) resolve(out.trim());
      else reject(new Error(`claude CLI exit ${code}: ${err.slice(0, 300)}`));
    });
    proc.stdin.end(prompt);
  });
}

async function callClaude(prompt) {
  const maxAttempts = 4;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await spawnClaudeOnce(prompt);
    } catch (e) {
      const isENOENT = e.code === "ENOENT" || /ENOENT/.test(e.message || "");
      if (isENOENT && attempt < maxAttempts) {
        const delay = 500 * attempt;
        console.log(`[bridge] ENOENT on attempt ${attempt}, retrying in ${delay}ms`);
        await new Promise(r => setTimeout(r, delay));
        continue;
      }
      throw e;
    }
  }
  throw new Error("unreachable");
}

const server = http.createServer(async (req, res) => {
  if (req.method === "GET" && req.url === "/v1/models") {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ object: "list", data: [{ id: "claude-via-cli", object: "model", owned_by: "anthropic-cli" }] }));
    return;
  }
  if (req.method !== "POST" || !req.url.includes("/v1/chat/completions")) {
    res.writeHead(404); res.end("not found"); return;
  }
  const t0 = Date.now();
  let body = "";
  for await (const chunk of req) body += chunk;
  try {
    const { messages = [] } = JSON.parse(body);
    // Flatten messages into a single prompt. Claude CLI takes one stdin blob.
    const sys = messages.filter(m => m.role === "system").map(m => m.content).join("\n\n");
    const usr = messages.filter(m => m.role !== "system").map(m => `[${m.role}]\n${m.content}`).join("\n\n");
    const prompt = sys ? `${sys}\n\n---\n\n${usr}` : usr;
    const text = await callClaude(prompt);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`[bridge] ${prompt.length}c in -> ${text.length}c out in ${elapsed}s`);
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({
      id: `chatcmpl-bridge-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: "claude-via-cli",
      choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
      usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    }));
  } catch (e) {
    console.error(`[bridge] ERROR: ${e.message}`);
    res.writeHead(500, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: { message: e.message, type: "bridge_error" } }));
  }
});

server.listen(PORT, () => {
  console.log(`Claude bridge listening on http://localhost:${PORT}`);
  console.log(`Test:  curl -s http://localhost:${PORT}/v1/chat/completions -H 'content-type: application/json' \\`);
  console.log(`         -d '{"model":"claude-via-cli","messages":[{"role":"user","content":"say OK"}]}' | jq -r .choices[0].message.content`);
});
