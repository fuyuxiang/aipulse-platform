import type { ListResponse, ResourceRecord } from "../models/types";

const API_PREFIX = "/api/v1";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

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
  return localStorage.getItem("aipulse_access_token") || "";
}

export function setTokens(tokens: TokenPair): void {
  localStorage.setItem("aipulse_access_token", tokens.access_token);
  localStorage.setItem("aipulse_refresh_token", tokens.refresh_token);
}

export async function checkedFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const response = await fetch(input, init);
  if (!response.ok) {
    const raw = await response.text();
    let detail = raw || response.statusText;
    try {
      const payload = JSON.parse(raw);
      detail =
        payload.message ||
        (typeof payload.detail === "string"
          ? payload.detail
          : Array.isArray(payload.detail)
            ? payload.detail.map((item: { msg: string }) => item.msg).join("；")
            : "") ||
        payload.code ||
        detail;
    } catch {
      /* retain non-JSON error text */
    }
    if (response.status === 401 && !String(input).includes("/auth/login")) {
      localStorage.removeItem("aipulse_access_token");
      localStorage.removeItem("aipulse_refresh_token");
      window.dispatchEvent(new Event("aipulse:unauthorized"));
    }
    throw new ApiError(
      response.status === 403 ? "当前账号没有执行此操作的权限" : detail,
      response.status,
    );
  }
  return response;
}
async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Content-Type", "application/json");
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const response = await checkedFetch(`${API_PREFIX}${path}`, {
    ...init,
    headers,
  });
  return (response.status === 204 ? {} : await response.json()) as T;
}

export const api = {
  login: (payload: LoginPayload) =>
    request<TokenPair>("/auth/login", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  me: () => request<Record<string, unknown>>("/auth/me"),
  list: <T = ResourceRecord>(
    path: string,
    page = 1,
    pageSize = 20,
    filters: Record<string, string> = {},
  ) =>
    request<ListResponse<T>>(
      `${path}${path.includes("?") ? "&" : "?"}${new URLSearchParams({ page: String(page), page_size: String(pageSize), ...filters })}`,
    ),
  create: (path: string, payload: Partial<ResourceRecord>) =>
    request<ResourceRecord>(path, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  update: (path: string, id: string, payload: Partial<ResourceRecord>) =>
    request<ResourceRecord>(`${path}/${id}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    }),
  remove: (path: string, id: string) =>
    request<Record<string, string>>(`${path}/${id}`, { method: "DELETE" }),
  post: <T = Record<string, unknown>>(
    path: string,
    payload: Record<string, unknown>,
  ) => request<T>(path, { method: "POST", body: JSON.stringify(payload) }),
  action: (path: string, payload: Record<string, unknown>) =>
    request<Record<string, unknown>>(path, {
      method: "POST",
      body: JSON.stringify({ payload }),
    }),
  get: <T = Record<string, unknown>>(path: string) => request<T>(path),
};

export function streamFetch(
  path: string,
  payload: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<Response> {
  const headers = new Headers();
  headers.set("Content-Type", "application/json");
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return fetch(`${API_PREFIX}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
    signal,
  });
}
