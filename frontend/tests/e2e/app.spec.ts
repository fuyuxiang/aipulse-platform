import { expect, test } from '@playwright/test';

test('login page renders', async ({ page }) => {
  await page.goto('/login');
  await expect(page.getByText('AIPulse')).toBeVisible();
  await expect(page.getByLabel('用户名')).toBeVisible();
});

test('dashboard requires authentication', async ({ page }) => {
  await page.goto('/dashboard');
  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByText('登录控制台')).toBeVisible();
});
