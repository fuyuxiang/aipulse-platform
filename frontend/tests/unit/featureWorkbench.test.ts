import { describe, expect, it } from 'vitest';
import { featureWorkbenchInternals } from '../../src/components/FeatureWorkbench';

describe('FeatureWorkbench', () => {
  it('maps knowledge pages to real operation endpoints', () => {
    const operations = featureWorkbenchInternals.operationsFor({ path: '/knowledge', title: '知识库', api: '/knowledge-bases', group: '知识库' });
    expect(operations.map((item) => item.key)).toContain('retrieve');
    expect(operations.find((item) => item.key === 'upload')?.buildPath('kb1')).toBe('/knowledge-bases/kb1/documents');
  });

  it('parses object JSON payloads', () => {
    expect(featureWorkbenchInternals.parsePayload('{"query":"agent"}')).toEqual({ query: 'agent' });
    expect(() => featureWorkbenchInternals.parsePayload('[]')).toThrow('JSON 必须是对象');
  });
});
