import { describe, expect, it } from 'vitest';
import { getToken } from '../../src/services/http';

describe('http token helper', () => {
  it('returns an empty token when storage has no token', () => {
    localStorage.removeItem('aipulse_access_token');
    expect(getToken()).toBe('');
  });
});

