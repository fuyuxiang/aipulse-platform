import { spawn } from 'node:child_process';
import http from 'node:http';

const port = 3100;

function waitForServer() {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const tick = () => {
      const req = http.get(`http://127.0.0.1:${port}/login`, (res) => {
        res.resume();
        resolve();
      });
      req.on('error', () => {
        if (Date.now() - started > 120000) {
          reject(new Error('frontend dev server did not become ready'));
          return;
        }
        setTimeout(tick, 500);
      });
    };
    tick();
  });
}

const server = spawn('npx', ['webpack', 'serve', '--mode', 'development', '--host', '127.0.0.1', '--port', String(port)], {
  stdio: ['ignore', 'inherit', 'inherit'],
  shell: false
});

const shutdown = () => {
  if (!server.killed) server.kill('SIGTERM');
};

try {
  await waitForServer();
  const result = await new Promise((resolve) => {
    const child = spawn('npx', ['playwright', 'test', '--config', 'playwright.config.ts'], {
      stdio: 'inherit',
      shell: false
    });
    child.on('exit', (code) => resolve(code ?? 1));
  });
  shutdown();
  process.exit(Number(result));
} catch (error) {
  console.error(error);
  shutdown();
  process.exit(1);
}

