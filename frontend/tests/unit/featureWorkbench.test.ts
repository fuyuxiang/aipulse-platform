import { describe, expect, it } from "vitest";
import { featureWorkbenchInternals } from "../../src/components/FeatureWorkbench";

describe("FeatureWorkbench", () => {
  it("maps knowledge pages to real operation endpoints", () => {
    const operations = featureWorkbenchInternals.operationsFor({
      path: "/knowledge",
      title: "知识库",
      api: "/knowledge-bases",
      group: "知识库",
    });
    expect(operations.map((item) => item.key)).toContain("retrieve");
    expect(
      operations.find((item) => item.key === "upload")?.buildPath("kb1"),
    ).toBe("/knowledge-bases/kb1/documents");
  });

  it("converts user input into execution contracts", () => {
    expect(
      featureWorkbenchInternals.buildPayload(
        "run",
        { prompt: "sum 17 and 25" },
        "{}",
      ),
    ).toEqual({ context: { input: "sum 17 and 25" } });
    expect(
      featureWorkbenchInternals.buildPayload(
        "evaluation-run",
        { input: "q", expected: "42", actual: "40" },
        "{}",
      ),
    ).toEqual({ cases: [{ input: "q", expected: "42", actual: "40" }] });
    expect(
      featureWorkbenchInternals.buildPayload(
        "test-rerank",
        { query: "a", documents: "one\ntwo" },
        "{}",
      ),
    ).toEqual({ query: "a", documents: ["one", "two"] });
  });
  it("parses object JSON payloads", () => {
    expect(featureWorkbenchInternals.parsePayload('{"query":"agent"}')).toEqual(
      { query: "agent" },
    );
    expect(() => featureWorkbenchInternals.parsePayload("[]")).toThrow(
      "JSON 必须是对象",
    );
  });
});
