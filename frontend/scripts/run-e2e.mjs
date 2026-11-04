import { spawn } from "node:child_process";
import http from "node:http";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const port = 3100;
const apiPort = 8107;
const root = path.resolve("..");
const data = mkdtempSync(path.join(tmpdir(), "aipulse-browser-e2e-"));
const env = {
  ...process.env,
  PYTHONPATH: path.join(root, "backend"),
  AIPULSE_DATABASE_URL: `sqlite:///${path.join(data, "test.db")}`,
  AIPULSE_DATA_DIR: data,
  AIPULSE_OBJECT_STORE_DIR: path.join(data, "files"),
  AIPULSE_VECTOR_STORE_DIR: path.join(data, "vectors"),
  AIPULSE_LOG_PATH: path.join(data, "backend.jsonl"),
  AIPULSE_TRACE_PATH: path.join(data, "traces.jsonl"),
  AIPULSE_SCHEDULER_ENABLED: "false",
  AIPULSE_E2E_API_TARGET: `http://127.0.0.1:${apiPort}`,
};
const bootstrap = `from app.main import app
from app.db.session import SessionLocal
from app.db.init_data import initialize_defaults
with SessionLocal() as db: initialize_defaults(db)
import uvicorn
uvicorn.run(app, host='127.0.0.1', port=${apiPort})`;
const backend = spawn("python3", ["-c", bootstrap], {
  cwd: root,
  env,
  stdio: "inherit",
});
const server = spawn(
  "npx",
  [
    "webpack",
    "serve",
    "--mode",
    "development",
    "--host",
    "127.0.0.1",
    "--port",
    String(port),
  ],
  { env, stdio: "inherit", shell: false },
);
function waitForServer(url) {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const tick = () => {
      const req = http.get(url, (res) => {
        res.resume();
        if (res.statusCode === 200) resolve();
        else retry();
      });
      req.on("error", retry);
      function retry() {
        if (Date.now() - started > 120000)
          reject(new Error(`${url} not ready`));
        else setTimeout(tick, 500);
      }
    };
    tick();
  });
}
console.log("Isolated browser evidence:", path.join(data, "browser-results"));
const shutdown = () => {
  server.kill("SIGTERM");
  backend.kill("SIGTERM");
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
try {
  await Promise.all([
    waitForServer(`http://127.0.0.1:${port}/`),
    waitForServer(`http://127.0.0.1:${apiPort}/health`),
  ]);
  const code = await new Promise((resolve) => {
    const child = spawn(
      "npx",
      [
        "playwright",
        "test",
        "--config",
        "playwright.config.ts",
        ...process.argv.slice(2),
      ],
      {
        env: {
          ...env,
          AIPULSE_E2E_OUTPUT_DIR: path.join(data, "browser-results"),
        },
        stdio: "inherit",
        shell: false,
      },
    );
    child.on("exit", (code) => resolve(code ?? 1));
  });
  shutdown();
  process.exit(Number(code));
} catch (error) {
  console.error(error);
  shutdown();
  process.exit(1);
}
