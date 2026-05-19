import type { ListResponse, ResourceRecord } from '../models/types';

const API_PREFIX = '/api/v1';

export interface LoginPayload {
  tenant: string;
  username: string;
  password: string;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  must_change_password: boolean;
}

export function getToken(): string {
  return localStorage.getItem('aipulse_access_token') || '';
}

export function setTokens(tokens: TokenPair): void {
  localStorage.setItem('aipulse_access_token', tokens.access_token);
  localStorage.setItem('aipulse_refresh_token', tokens.refresh_token);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set('Content-Type', 'application/json');
  const token = getToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const response = await fetch(`${API_PREFIX}${path}`, { ...init, headers });
  if (!response.ok) {
    const text = await response.text();
    let errorMessage = text || response.statusText;
    try {
      const payload = JSON.parse(text) as { message?: string; code?: string };
      errorMessage = payload.message || payload.code || errorMessage;
    } catch {
      /* keep server text */
    }
    throw new Error(errorMessage);
  }
  return (await response.json()) as T;
}

export const api = {
  login: (payload: LoginPayload) => request<TokenPair>('/auth/login', { method: 'POST', body: JSON.stringify(payload) }),
  me: () => request<Record<string, unknown>>('/auth/me'),
  list: (path: string, page = 1, pageSize = 20) => request<ListResponse<ResourceRecord>>(`${path}?page=${page}&page_size=${pageSize}`),
  create: (path: string, payload: Partial<ResourceRecord>) => request<ResourceRecord>(path, { method: 'POST', body: JSON.stringify(payload) }),
  update: (path: string, id: string, payload: Partial<ResourceRecord>) => request<ResourceRecord>(`${path}/${id}`, { method: 'PUT', body: JSON.stringify(payload) }),
  remove: (path: string, id: string) => request<Record<string, string>>(`${path}/${id}`, { method: 'DELETE' }),
  post: <T = Record<string, unknown>>(path: string, payload: Record<string, unknown>) => request<T>(path, { method: 'POST', body: JSON.stringify(payload) }),
  action: (path: string, payload: Record<string, unknown>) => request<Record<string, unknown>>(path, { method: 'POST', body: JSON.stringify({ payload }) }),
  get: <T = Record<string, unknown>>(path: string) => request<T>(path),
};

export function streamFetch(path: string, payload: Record<string, unknown>, signal?: AbortSignal): Promise<Response> {
  const headers = new Headers();
  headers.set('Content-Type', 'application/json');
  const token = getToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  return fetch(`${API_PREFIX}${path}`, { method: 'POST', headers, body: JSON.stringify(payload), signal });
}
