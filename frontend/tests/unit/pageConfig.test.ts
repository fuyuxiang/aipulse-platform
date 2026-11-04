import { describe, expect, it } from 'vitest';
import { pageConfigs } from '../../src/routes/pageConfig';

describe('pageConfigs', () => {
  it('covers the consolidated navigation groups', () => {
    expect(pageConfigs.length).toBeGreaterThanOrEqual(20);
    expect(pageConfigs.some((page) => page.designer === 'workflow')).toBe(true);
    const groups = [...new Set(pageConfigs.map((p) => p.group))];
    expect(groups).toEqual(['工作台', '构建', '使用', '评测', '观测', '发布', '设置']);
    expect(pageConfigs.find(p => p.path === '/playground/multi')?.api).toBe('/agent-teams');
  });
});
