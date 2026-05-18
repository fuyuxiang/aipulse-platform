import { describe, expect, it } from 'vitest';
import { pageConfigs } from '../../src/routes/pageConfig';

describe('pageConfigs', () => {
  it('covers the enterprise console pages', () => {
    expect(pageConfigs.length).toBeGreaterThanOrEqual(100);
    expect(pageConfigs.some((page) => page.designer === 'workflow')).toBe(true);
    expect(pageConfigs.find((page) => page.title === '模型 Provider 管理页')?.api).toBe('/model-providers');
  });
});

