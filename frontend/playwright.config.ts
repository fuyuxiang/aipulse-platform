import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  outputDir: process.env.AIPULSE_E2E_OUTPUT_DIR || "test-results",
  timeout: 45000,
  workers: 2,
  use: {
    baseURL: "http://127.0.0.1:3100",
    screenshot: "only-on-failure",
    launchOptions: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH }
      : undefined,
  },
});
