import { describe, expect, it } from 'vitest';
import { pageConfigs } from '../../src/routes/pageConfig';

describe('pageConfigs', () => {
  it('covers the consolidated navigation groups', () => {
    expect(pageConfigs.length).toBeGreaterThanOrEqual(20);
    expect(pageConfigs.some((page) => page.designer === 'workflow')).toBe(true);
    const groups = [...new Set(pageConfigs.map((p) => p.group))];
    expect(groups).toContain('首页');
    expect(groups).toContain('开发');
    expect(groups).toContain('运行');
    expect(groups).toContain('治理');
    expect(groups).toContain('设置');
    expect(groups.length).toBe(5);
  });
});
