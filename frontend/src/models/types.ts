export interface ListResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface ResourceRecord {
  id: string;
  tenant_id: string;
  name: string;
  code: string;
  description: string;
  status: string;
  enabled: boolean;
  model_type?: string;
  provider_type?: string;
  provider_id?: string;
  model_id?: string;
  agent_id?: string;
  workflow_id?: string;
  session_id?: string;
  parent_id?: string;
  owner_id?: string;
  user_id?: string;
  trace_id?: string;
  config?: Record<string, unknown>;
  spec?: Record<string, unknown>;
  metadata_json?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface PageConfig {
  path: string;
  title: string;
  api: string;
  group: string;
  designer?: 'workflow' | 'agent';
}

