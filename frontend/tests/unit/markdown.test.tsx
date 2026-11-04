import React from "react";
import { render, screen, cleanup } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { MarkdownContent } from "../../src/components/MarkdownContent";
afterEach(cleanup);
describe("assistant response rendering", () => {
  it("renders tables and code without executing HTML or loading remote images", () => {
    const content =
      "| Source | Value |\n| --- | --- |\n| S1 | 42 |\n\n```python\nprint(42)\n```\n\n<img src=x onerror=alert(1)>\n\n![chart](https://example.com/tracker.png)";
    const { container } = render(<MarkdownContent>{content}</MarkdownContent>);
    expect(screen.getByRole("table")).toBeTruthy();
    expect(container.querySelector("pre code")?.textContent).toContain(
      "print(42)",
    );
    expect(container.querySelector("img")).toBeNull();
    expect(screen.getByRole("link").getAttribute("rel")).toContain("noopener");
  });
});
