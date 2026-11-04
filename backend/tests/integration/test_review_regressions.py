"""Business-result regressions for the 2026-11-04 product/security review.

Provider stubs in this file prove local contracts, never real model quality.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import app
from app.db.session import SessionLocal
from app.services._shared.resource_service import ResourceService
from app.services.runtime.chat_service import ChatService
from app.services.runtime.agent_runner_service import AgentRunnerService
from app.services.settings.model_services import ModelInvocationService


def unique():
    return uuid.uuid4().hex[:12]


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def admin(client):
    response = client.post('/api/v1/auth/login', json={'tenant': 'default', 'username': 'admin', 'password': 'admin123456'})
    assert response.status_code == 200
    return {'Authorization': f"Bearer {response.json()['access_token']}"}


def create(client, admin, path, **payload):
    response = client.post('/api/v1' + path, headers=admin, json={'name': unique(), **payload})
    assert response.status_code == 200, response.text
    return response.json()


def member(client, admin, permissions):
    username = unique()
    user = create(client, admin, '/users', username=username, password='review-pass-123')
    role = create(client, admin, '/roles', code=unique(), config={'permissions': permissions})
    result = client.put(f"/api/v1/users/{user['id']}/roles", headers=admin, json=[role['id']])
    assert result.status_code == 200, result.text
    login = client.post('/api/v1/auth/login', json={'username': username, 'password': 'review-pass-123'})
    assert login.status_code == 200
    return user, role, {'Authorization': f"Bearer {login.json()['access_token']}"}


def sse_events(response):
    assert response.status_code == 200, response.text
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]


def test_ordinary_user_grant_revoke_and_no_superuser_escalation(client, admin):
    user, role, auth = member(client, admin, ['agents:read', 'chat:read', 'chat:write', 'users:write'])
    assert client.get('/api/v1/agents', headers=auth).status_code == 200
    assert client.post('/api/v1/agents', headers=auth, json={'name': 'denied'}).status_code == 403
    assert client.post('/api/v1/users', headers=auth, json={'username': unique(), 'password': 'pass', 'is_superuser': True}).status_code == 403
    changed = client.put(f"/api/v1/roles/{role['id']}", headers=admin, json={'config': {'permissions': ['chat:read']}})
    assert changed.status_code == 200
    assert client.get('/api/v1/agents', headers=auth).status_code == 403
    assert client.put(f"/api/v1/users/{user['id']}/roles", headers=admin, json=[]).status_code == 200
    assert client.get('/api/v1/chat/sessions', headers=auth).status_code == 403


def test_chat_history_regeneration_count_and_user_isolation(client, admin, monkeypatch):
    calls = []
    async def answer(self, tenant_id, user_id, agent_id, session_id, prompt, context, rag, tools, spec):
        calls.append((prompt, context))
        return 'answer:' + prompt
    monkeypatch.setattr(ChatService, '_call_agent', answer)
    agent = create(client, admin, '/agents')
    session = create(client, admin, '/chat/sessions', agent_id=agent['id'])
    path = f"/api/v1/chat/sessions/{session['id']}"
    assert client.get('/api/v1/chat/sessions', headers=admin).status_code == 200
    first = sse_events(client.post(path + '/stream', headers=admin, json={'content': 'first'}))
    sse_events(client.post(path + '/stream', headers=admin, json={'content': 'second'}))
    first_id = next(e['message_id'] for e in first if e['type'] == 'message_end')
    sse_events(client.post(path + f'/messages/{first_id}/regenerate', headers=admin))
    assert calls == [('first', []), ('second', [{'role': 'user', 'content': 'first'}, {'role': 'assistant', 'content': 'answer:first'}]), ('first', [])]
    messages = client.get(path + '/messages', headers=admin).json()
    assert messages['total'] == 5
    assert [m['spec']['role'] for m in messages['items']].count('user') == 2
    assert client.get(path, headers=admin).json()['spec']['message_count'] == 5
    _, _, other = member(client, admin, ['chat:*'])
    for method, suffix in [('get', ''), ('get', '/messages'), ('post', '/stream'), ('delete', '')]:
        assert getattr(client, method)(path + suffix, headers=other).status_code == 404
    assert client.post(f'/api/v1/chat/messages/{first_id}/feedback', headers=other, json={'rating': 'bad'}).status_code == 404


def test_chat_model_failure_is_persisted(client, admin, monkeypatch):
    async def fail(*args, **kwargs):
        raise TimeoutError('controlled provider timeout')
    monkeypatch.setattr(ChatService, '_call_agent', fail)
    session = create(client, admin, '/chat/sessions')
    path = f"/api/v1/chat/sessions/{session['id']}"
    events = sse_events(client.post(path + '/stream', headers=admin, json={'content': 'hello'}))
    assert events[-1]['type'] == 'error'
    messages = client.get(path + '/messages', headers=admin).json()['items']
    assert messages[-1]['status'] == 'failed'
    assert client.get(path, headers=admin).json()['spec']['message_count'] == 2


def test_tool_approval_is_bound_expiring_and_single_use(client, admin):
    tool = create(client, admin, '/tools', config={'type': 'calculator'}, spec={'risk_level': 'high'})
    invoke = f"/api/v1/tools/{tool['id']}/invoke"
    args = {'numbers': [2, 3]}
    waiting = client.post(invoke, headers=admin, json={'arguments': args, 'approved': True}).json()
    assert waiting['status'] == 'waiting_approval' and 'output' not in waiting
    assert client.post(invoke, headers=admin, json={'arguments': args, 'approval_id': 'forged'}).status_code == 404
    approval_id = waiting['approval_id']
    _, _, operator = member(client, admin, ['tools:write'])
    assert client.post(f'/api/v1/tool-approval-tasks/{approval_id}/approve', headers=operator).status_code == 403
    assert client.post(f'/api/v1/tool-approval-tasks/{approval_id}/approve', headers=admin).status_code == 200
    assert client.post(invoke, headers=admin, json={'arguments': {'numbers': [9]}, 'approval_id': approval_id}).status_code == 403
    assert client.post(invoke, headers=operator, json={'arguments': args, 'approval_id': approval_id}).status_code == 403
    approved = client.post(invoke, headers=admin, json={'arguments': args, 'approval_id': approval_id})
    assert approved.json()['output']['value'] == 5
    assert client.post(invoke, headers=admin, json={'arguments': args, 'approval_id': approval_id}).status_code == 403
    for decision in ['rejected', 'expired']:
        pending = client.post(invoke, headers=admin, json={'arguments': args}).json()['approval_id']
        action = 'reject' if decision == 'rejected' else 'approve'
        assert client.post(f'/api/v1/tool-approval-tasks/{pending}/{action}', headers=admin).status_code == 200
        if decision == 'expired':
            me = client.get('/api/v1/auth/me', headers=admin).json()
            with SessionLocal() as db:
                resources = ResourceService(db)
                approval = resources.get('tool_approval_tasks', me['tenant_id'], pending)
                resources.update('tool_approval_tasks', me['tenant_id'], me['id'], pending, {'spec': {**approval.spec, 'expires_at': (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}})
        assert client.post(invoke, headers=admin, json={'arguments': args, 'approval_id': pending}).status_code == 403


def test_workflow_approval_resumes_actual_downstream_once_and_rejects(client, admin):
    tool = create(client, admin, '/tools', config={'type': 'calculator'})
    nodes = [{'id': 's', 'type': 'start'}, {'id': 'approve', 'type': 'human_approval'},
             {'id': 'effect', 'type': 'tool', 'config': {'tool_id': tool['id'], 'arguments': {'numbers': [17, 25]}}}, {'id': 'e', 'type': 'end'}]
    edges = [{'source': a, 'target': b} for a, b in [('s', 'approve'), ('approve', 'effect'), ('effect', 'e')]]
    workflow = create(client, admin, '/workflows', spec={'nodes': nodes, 'edges': edges})
    run = client.post(f"/api/v1/workflows/{workflow['id']}/run", headers=admin).json()
    path = f"/api/v1/workflow-runs/{run['run_id']}"
    assert run['status'] == 'waiting_approval'
    assert client.post(path + '/resume', headers=admin).status_code == 403
    assert client.post(f"/api/v1/workflow-approvals/{run['waiting']['approval_id']}/approve", headers=admin).status_code == 200
    resumed = client.post(path + '/resume', headers=admin)
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()['status'] == 'success'
    assert resumed.json()['results']['effect']['output']['tool']['output']['value'] == 42
    assert client.post(path + '/resume', headers=admin).status_code == 409
    assert client.post(path + '/cancel', headers=admin).status_code == 409
    rejected = client.post(f"/api/v1/workflows/{workflow['id']}/run", headers=admin).json()
    assert client.post(f"/api/v1/workflow-approvals/{rejected['waiting']['approval_id']}/reject", headers=admin).status_code == 200
    assert client.post(f"/api/v1/workflow-runs/{rejected['run_id']}/resume", headers=admin).status_code == 409


def test_cross_tenant_runtime_checkpoints_resources_and_sockets(client, admin):
    agent = create(client, admin, '/agents')
    instance = create(client, admin, f"/runtime/agents/{agent['id']}/instances")
    tenant = create(client, admin, '/tenants', code=unique())
    other_admin = {**admin, 'X-Tenant-ID': tenant['id']}
    for method, suffix in [('post', '/start'), ('post', '/stop'), ('post', '/restart'), ('delete', ''), ('get', '/health')]:
        response = getattr(client, method)(f"/api/v1/runtime/instances/{instance['id']}" + suffix, headers=other_admin)
        assert response.status_code == 404, response.text
    assert client.get(f"/api/v1/runtime/instances/{instance['id']}/health", headers=admin).json()['status'] == 'created'
    assert client.get(f"/api/v1/agents/{agent['id']}", headers=other_admin).status_code == 404
    role = create(client, admin, '/roles', code=unique())
    foreign_user = create(client, other_admin, '/users', username=unique(), password='password')
    assert client.post('/api/v1/user-roles', headers=admin, json={'user_id': foreign_user['id'], 'role_id': role['id']}).status_code == 404
    for path in ['/api/v1/ws/runtime/runs/missing', '/api/v1/ws/workflow-runs/missing']:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(path):
                pass
    assert client.get('/api/v1/workflow-runs/missing/checkpoints', headers=other_admin).status_code == 404


def test_team_pipeline_passes_outputs_and_all_member_failure_is_failed(client, admin, monkeypatch):
    seen = []
    async def run(self, tenant_id, user_id, agent_id, payload, **kwargs):
        seen.append(payload)
        return {'response': payload['prompt'] + ':processed'}
    monkeypatch.setattr(AgentRunnerService, 'run', run)
    agents = [create(client, admin, '/agents') for _ in range(2)]
    team = create(client, admin, '/agent-teams', topology='pipeline', shared_memory_enabled=False,
                  enable_failover=False, members=[{'agent_id': a['id'], 'priority': i} for i, a in enumerate(agents)])
    response = client.post(f"/api/v1/agent-teams/{team['id']}/run", headers=admin, json={'prompt': 'source'})
    assert response.status_code == 200, response.text
    assert response.json()['response'] == 'source:processed:processed'
    assert [p['prompt'] for p in seen] == ['source', 'source:processed']
    assert all(p['memory_policy']['team_id'] == team['id'] and not p['memory_policy']['enabled'] for p in seen)
    async def fail(*args, **kwargs):
        raise RuntimeError('controlled member failure')
    monkeypatch.setattr(AgentRunnerService, 'run', fail)
    mesh = create(client, admin, '/agent-teams', topology='mesh', members=[{'agent_id': a['id']} for a in agents])
    failed = client.post(f"/api/v1/agent-teams/{mesh['id']}/run", headers=admin, json={'prompt': 'task'})
    assert failed.status_code == 502, failed.text
    history = client.get(f"/api/v1/agent-teams/{mesh['id']}/runs", headers=admin).json()
    assert history['items'][0]['status'] == 'failed'


def test_knowledge_uses_configured_embedding_for_document_and_query(client, admin, monkeypatch):
    seen = []
    async def invoke(self, tenant_id, user_id, model_id, model_type, payload):
        seen.append((model_type, payload))
        return {'result': {'embeddings': [[1.0, 0.0] for _ in payload.get('texts', [])]}}
    monkeypatch.setattr(ModelInvocationService, 'invoke', invoke)
    model = create(client, admin, '/models', model_type='embedding', provider_type='openai_compatible', model_id='fixture-embedding')
    kb = create(client, admin, '/knowledge-bases', config={'embedding_model_id': model['id'], 'embedding_dimensions': 2})
    doc = create(client, admin, f"/knowledge-bases/{kb['id']}/documents", filename='source.txt', content='document evidence: A=42')
    result = client.post(f"/api/v1/knowledge-bases/{kb['id']}/retrieve", headers=admin, json={'query': 'A', 'mode': 'vector'})
    assert result.status_code == 200, result.text
    assert result.json()['matches'][0]['metadata']['document_id'] == doc['id']
    assert result.json()['matches'][0]['text'] == 'document evidence: A=42'
    assert seen == [('embedding', {'texts': ['document evidence: A=42']}), ('embedding', {'texts': ['A']})]
    assert client.put(f"/api/v1/models/{model['id']}", headers=admin, json={'config': {'model_name': 'changed'}}).status_code == 200
    assert client.post(f"/api/v1/knowledge-bases/{kb['id']}/retrieve", headers=admin, json={'query': 'A'}).status_code == 409


def test_publication_pins_agent_config_and_enforces_key_lifecycle(client, admin, monkeypatch):
    seen = []
    async def invoke(self, tenant_id, user_id, model_id, model_type, payload):
        seen.append(payload['messages'])
        return {'result': {'content': '42', 'usage': {'input_tokens': 8, 'output_tokens': 1}}, 'cost': 0}
    monkeypatch.setattr(ModelInvocationService, 'invoke', invoke)
    model = create(client, admin, '/models', model_type='chat_llm', provider_type='local')
    agent = create(client, admin, '/agents', config={'model_id': model['id'], 'system_prompt': 'published v1'})
    publication = create(client, admin, f"/agents/{agent['id']}/publish-api", rate_limit={'requests_per_minute': 1})
    assert client.put(f"/api/v1/agents/{agent['id']}", headers=admin, json={'config': {'model_id': model['id'], 'system_prompt': 'draft v2'}}).status_code == 200
    path = f"/api/v1/published/{agent['id']}/chat"
    auth = {'X-API-Key': publication['api_key']}
    assert client.post(path, headers=auth, json={'content': 'sum'}).json()['response'] == '42'
    assert seen[0][0]['content'] == 'published v1'
    assert client.post(path, headers=auth, json={'content': 'sum'}).status_code == 429
    assert client.post(path, headers={'X-API-Key': 'forged'}, json={'content': 'sum'}).status_code == 401
    disabled = client.put(f"/api/v1/agent-publications/{publication['id']}", headers=admin, json={'status': 'disabled'})
    assert disabled.status_code == 200, disabled.text
    assert client.post(path, headers=auth, json={'content': 'sum'}).status_code == 403
    expired = create(client, admin, f"/agents/{agent['id']}/publish-api", expires_at='2000-01-01T00:00:00Z')
    assert client.post(path, headers={'X-API-Key': expired['api_key']}, json={'content': 'sum'}).status_code == 401


@pytest.mark.asyncio
async def test_workflow_cancel_terminates_active_local_handler_and_stops_successor(monkeypatch):
    import asyncio
    from app.services.build.workflow_service import WorkflowService
    from app.models.core import User
    from sqlalchemy import select
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    effects = []
    async def slow(config, context):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
    async def after(config, context):
        effects.append('should not execute')
    monkeypatch.setattr(WorkflowService, '_build_handlers', lambda *args: {'slow': slow, 'after': after})
    with SessionLocal() as db, SessionLocal() as cancellation_db:
        user = db.scalar(select(User).where(User.username == 'admin'))
        resources = ResourceService(db)
        workflow = resources.create('workflow_definitions', user.tenant_id, user.id, {'name': 'cancel flow', 'spec': {
            'nodes': [{'id': 'slow', 'type': 'slow'}, {'id': 'after', 'type': 'after'}], 'edges': [{'source': 'slow', 'target': 'after'}]}})
        task = asyncio.create_task(WorkflowService(db).run(user.tenant_id, user.id, workflow.id, {}))
        await asyncio.wait_for(entered.wait(), 2)
        runs, _ = resources.list('workflow_runs', user.tenant_id, 1, 1, {'workflow_id': workflow.id})
        result = await WorkflowService(cancellation_db).cancel(user.tenant_id, user.id, runs[0].id, {})
        assert result['status'] == 'cancelled'
        assert (await task)['status'] == 'cancelled'
        assert cancelled.is_set() and effects == []


def test_memory_subject_session_team_boundaries_and_no_scope_widening(client, admin):
    user, _, auth = member(client, admin, ['memory:*', 'agents:*'])
    other, _, other_auth = member(client, admin, ['memory:*', 'agents:*'])
    team = create(client, auth, '/agent-teams', topology='pipeline')
    remembered = client.post('/api/v1/memories/remember', headers=auth, json={'content': 'private fact', 'scope': 'session', 'session_id': 'same-session'})
    assert remembered.status_code == 200
    assert client.post('/api/v1/memories/search', headers=other_auth, json={'query': 'private', 'scope': 'session', 'session_id': 'same-session'}).json()['total'] == 0
    assert client.post('/api/v1/memories/search', headers=other_auth, json={'subject_user_id': user['id']}).status_code == 403
    assert client.post('/api/v1/memories/remember', headers=auth, json={'content': 'team', 'scope': 'team'}).status_code == 422
    assert client.post('/api/v1/memories/remember', headers=auth, json={'content': 'team fact', 'scope': 'team', 'team_id': team['id']}).status_code == 200
    assert client.post('/api/v1/memories/search', headers=other_auth, json={'team_id': team['id']}).status_code == 403
    assert client.get(f"/api/v1/agent-teams/{team['id']}", headers=other_auth).status_code == 404
    assert other['id'] != user['id']


def test_checkpoint_survives_new_python_process_without_repeating_side_effects(client, admin):
    import os
    import subprocess
    import sys
    tool = create(client, admin, '/tools', config={'type': 'calculator'})
    nodes = [{'id': 'pre', 'type': 'tool', 'config': {'tool_id': tool['id'], 'arguments': {'numbers': [11]}}},
             {'id': 'gate', 'type': 'human_approval'},
             {'id': 'post', 'type': 'tool', 'config': {'tool_id': tool['id'], 'arguments': {'numbers': [31]}}}]
    workflow = create(client, admin, '/workflows', spec={'nodes': nodes, 'edges': [{'source': 'pre', 'target': 'gate'}, {'source': 'gate', 'target': 'post'}]})
    run = client.post(f"/api/v1/workflows/{workflow['id']}/run", headers=admin).json()
    assert run['status'] == 'waiting_approval'
    assert client.post(f"/api/v1/workflow-approvals/{run['waiting']['approval_id']}/approve", headers=admin).status_code == 200
    me = client.get('/api/v1/auth/me', headers=admin).json()
    code = '''import asyncio,json,sys
from app.db.session import SessionLocal
from app.services.build.workflow_service import WorkflowService
with SessionLocal() as db:
 result = asyncio.run(WorkflowService(db).resume_from_checkpoint(sys.argv[1],sys.argv[2],sys.argv[3],{}))
 print(json.dumps(result))
'''
    completed = subprocess.run([sys.executable, '-c', code, me['tenant_id'], me['id'], run['run_id']], env=os.environ.copy(), capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result['status'] == 'success'
    logs = client.get('/api/v1/tool-call-logs?page_size=200', headers=admin).json()['items']
    effects = [row for row in logs if row['parent_id'] == tool['id'] and row['status'] == 'success']
    assert sorted(row['output_payload']['value'] for row in effects) == [11.0, 31.0]


def test_imported_evaluation_requires_actual_answers_and_records_method(client, admin):
    assert client.post('/api/v1/evaluation/runs', headers=admin, json={'cases': [{'input': '42', 'expected': '42'}]}).status_code == 422
    response = client.post('/api/v1/evaluation/runs', headers=admin, json={'cases': [{'input': 'sum', 'expected': '42', 'actual': '142', 'score': 1}]})
    assert response.status_code == 200
    assert response.json()['metrics']['accuracy'] == 0
    assert response.json()['execution_mode'] == 'imported_answers'
    assert response.json()['results'][0]['actual'] == '142'


def test_second_tenant_ordinary_login_isolation_and_disabled_tenant_revokes_access(client, admin):
    tenant = create(client, admin, '/tenants', code=unique())
    scoped_admin = {**admin, 'X-Tenant-ID': tenant['id']}
    user = create(client, scoped_admin, '/users', username=unique(), password='tenant-password')
    role = create(client, scoped_admin, '/roles', code=unique(), config={'permissions': ['agents:*', 'tenants:*', 'chat:*']})
    assert client.put(f"/api/v1/users/{user['id']}/roles", headers=scoped_admin, json=[role['id']]).status_code == 200
    login = client.post('/api/v1/auth/login', json={'tenant': tenant['code'], 'username': user['username'], 'password': 'tenant-password'})
    assert login.status_code == 200
    auth = {'Authorization': f"Bearer {login.json()['access_token']}"}
    own = create(client, auth, '/agents')
    foreign = create(client, admin, '/agents')
    assert client.get(f"/api/v1/agents/{own['id']}", headers=auth).status_code == 200
    assert client.get(f"/api/v1/agents/{foreign['id']}", headers=auth).status_code == 404
    assert client.get('/api/v1/agents', headers={**auth, 'X-Tenant-ID': foreign['tenant_id']}).status_code == 403
    assert client.get('/api/v1/tenants', headers=auth).status_code == 403
    assert client.post(f"/api/v1/tenants/{tenant['id']}/disable", headers=admin).status_code == 200
    assert client.get('/api/v1/agents', headers=auth).status_code == 401


def test_api_key_scopes_expiry_and_nested_credential_redaction(client, admin):
    from app.services._shared.auth_service import AuthService
    from app.models.core import APIKey
    me = client.get('/api/v1/auth/me', headers=admin).json()
    with SessionLocal() as db:
        key = AuthService(db).create_api_key(me['tenant_id'], me['id'], 'scoped review key', ['agents:read'])
    auth = {'X-API-Key': key['api_key']}
    assert client.get('/api/v1/agents', headers=auth).status_code == 200
    assert client.get('/api/v1/users', headers=auth).status_code == 403
    assert client.get('/api/v1/tenants', headers=auth).status_code == 403
    with SessionLocal() as db:
        db.get(APIKey, key['id']).expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.commit()
    assert client.get('/api/v1/agents', headers=auth).status_code == 401
    assert client.get('/api/v1/audit-integrity/verify', headers=admin).json()['valid'] is True
    model = create(client, admin, '/models', config={'model_config': {'api_key': 'review-secret-never-expose'}})
    assert model['config']['model_config']['api_key'] == '***'
    assert 'review-secret-never-expose' not in client.get('/api/v1/audit-logs?page_size=200', headers=admin).text


@pytest.mark.parametrize('failure, expected', [('rate_limit', 429), ('timeout', 502)])
def test_provider_failures_are_visible_and_persisted(client, admin, monkeypatch, failure, expected):
    import httpx
    async def fail(self, url, **kwargs):
        if failure == 'timeout':
            raise httpx.ReadTimeout('controlled provider timeout')
        return httpx.Response(429, json={'error': {'message': 'controlled quota exceeded'}}, request=httpx.Request('POST', url))
    monkeypatch.setattr(httpx.AsyncClient, 'post', fail)
    model = create(client, admin, '/models', provider_type='openai_compatible', model_type='chat_llm', model_id='controlled', config={'auth_scheme': 'none', 'base_url': 'http://controlled.invalid/v1'})
    agent = create(client, admin, '/agents', config={'model_id': model['id'], 'memory_policy': {'enabled': False}})
    response = client.post(f"/api/v1/agents/{agent['id']}/run", headers=admin, json={'prompt': 'synthetic failure request'})
    assert response.status_code == expected, response.text
    history = client.get(f"/api/v1/agents/{agent['id']}/runs", headers=admin).json()
    assert history['items'][0]['status'] == 'failed'
    assert 'controlled' in history['items'][0]['error_message']


def test_chat_client_cannot_set_internal_regeneration_or_override_agent_bindings(client, admin, monkeypatch):
    captured = []
    async def run(self, tenant_id, user_id, agent_id, payload, **kwargs):
        captured.append(payload)
        return {'response': 'safe response'}
    monkeypatch.setattr(AgentRunnerService, 'run', run)
    agent = create(client, admin, '/agents', config={'memory_policy': {'enabled': False}})
    own = create(client, admin, '/chat/sessions', agent_id=agent['id'], tool_ids=['unbound'], guardrail_policy_ids=['weaker'], model_id='unbound')
    foreign = create(client, admin, '/chat/sessions', agent_id=agent['id'])
    target = client.post(f"/api/v1/chat/sessions/{foreign['id']}/messages", headers=admin, json={'content': 'other input'}).json()
    events = sse_events(client.post(f"/api/v1/chat/sessions/{own['id']}/stream", headers=admin, json={
        'content': 'safe request', '_regenerate_id': target['id'], '_parent_id': target['id']}))
    assert events[-1]['type'] == 'message_end'
    assert captured == [{'prompt': 'safe request', 'session_id': own['id'], 'context': []}]
    assert client.get(f"/api/v1/chat/sessions/{foreign['id']}/messages", headers=admin).json()['items'][0]['status'] == 'sent'
    messages = client.get(f"/api/v1/chat/sessions/{own['id']}/messages", headers=admin).json()['items']
    assert len(messages) == 2 and messages[-1]['parent_id'] == messages[0]['id']


def test_required_tool_policy_rejects_textual_pretend_execution(client, admin, monkeypatch):
    async def invoke(*args, **kwargs):
        return {'result': {'content': 'I called the calculator: 42', 'tool_calls': []}}
    monkeypatch.setattr(ModelInvocationService, 'invoke', invoke)
    tool = create(client, admin, '/tools', code='review_sum', config={'type': 'calculator'})
    model = create(client, admin, '/models', model_type='chat_llm', provider_type='local')
    agent = create(client, admin, '/agents', config={'model_id': model['id'], 'tool_ids': [tool['id']], 'tool_choice': 'required'})
    response = client.post(f"/api/v1/agents/{agent['id']}/run", headers=admin, json={'prompt': 'sum'})
    assert response.status_code == 409
    logs = client.get('/api/v1/tool-call-logs?page_size=200', headers=admin).json()['items']
    assert not any(row['parent_id'] == tool['id'] for row in logs)


def test_runtime_list_and_operations_are_owned_for_ordinary_users(client, admin):
    agent = create(client, admin, '/agents')
    _, _, owner = member(client, admin, ['runtime:*'])
    _, _, other = member(client, admin, ['runtime:*'])
    instance = create(client, owner, f"/runtime/agents/{agent['id']}/instances", session_id='../../same')
    own = client.get('/api/v1/runtime/instances', headers=owner).json()
    assert [item['id'] for item in own] == [instance['id']]
    assert client.get('/api/v1/runtime/instances', headers=other).json() == []
    assert client.post(f"/api/v1/runtime/instances/{instance['id']}/stop", headers=other).status_code == 404
    assert '..' not in instance['workspace']


def test_agent_tool_approval_remains_waiting_and_does_not_generate_a_fake_answer(client, admin, monkeypatch):
    async def invoke(*args, **kwargs):
        return {'result': {'content': '', 'tool_calls': [{'id': 'controlled-call', 'function': {'name': 'high_risk_sum', 'arguments': '{"numbers":[17,25]}'}}]}}
    monkeypatch.setattr(ModelInvocationService, 'invoke', invoke)
    tool = create(client, admin, '/tools', code='high_risk_sum', config={'type': 'calculator'}, spec={'risk_level': 'high'})
    model = create(client, admin, '/models', model_type='chat_llm', provider_type='local')
    agent = create(client, admin, '/agents', config={'model_id': model['id'], 'tool_ids': [tool['id']]})
    response = client.post(f"/api/v1/agents/{agent['id']}/run", headers=admin, json={'prompt': 'sum'})
    assert response.status_code == 200
    assert response.json()['status'] == 'waiting_approval'
    assert response.json()['approval_id'] and response.json()['response'] == ''
    session = create(client, admin, '/chat/sessions', agent_id=agent['id'])
    events = sse_events(client.post(f"/api/v1/chat/sessions/{session['id']}/stream", headers=admin, json={'content': 'sum'}))
    assert events[-1]['type'] == 'approval_required'
    assert not any(event['type'] == 'message_end' for event in events)
    team = create(client, admin, '/agent-teams', topology='pipeline', members=[{'agent_id': agent['id']}])
    waiting = client.post(f"/api/v1/agent-teams/{team['id']}/run", headers=admin, json={'prompt': 'sum'})
    assert waiting.json()['status'] == 'waiting_approval'
    publication = create(client, admin, f"/agents/{agent['id']}/publish-api")
    public = client.post(f"/api/v1/published/{agent['id']}/chat", headers={'X-API-Key': publication['api_key']}, json={'content': 'sum'})
    assert public.json()['status'] == 'waiting_approval' and public.json()['response'] == ''
    workflow = create(client, admin, '/workflows', spec={'nodes': [{'id': 'agent', 'type': 'agent', 'config': {'agent_id': agent['id'], 'prompt': 'sum'}}], 'edges': []})
    workflow_run = client.post(f"/api/v1/workflows/{workflow['id']}/run", headers=admin).json()
    assert workflow_run['status'] == 'waiting_approval'
    assert client.post(f"/api/v1/workflow-runs/{workflow_run['run_id']}/resume", headers=admin).status_code == 409
    logs = client.get('/api/v1/tool-call-logs?page_size=200', headers=admin).json()['items']
    assert not any(row['parent_id'] == tool['id'] for row in logs)


def test_resource_editor_retains_masked_credentials(client, admin):
    model = create(client, admin, '/models', config={'base_url': 'http://test.invalid/v1', 'api_key': 'synthetic-ui-secret', 'nested': {'password': 'nested-ui-secret'}})
    assert model['config']['api_key'] == '***'
    edited = {**model['config'], 'temperature': 0.3}
    response = client.put(f"/api/v1/models/{model['id']}", headers=admin, json={'config': edited})
    assert response.status_code == 200
    assert response.json()['config']['api_key'] == '***'
    with SessionLocal() as db:
        stored = ResourceService(db).get('models', model['tenant_id'], model['id'])
        assert stored.config['api_key'] == 'synthetic-ui-secret'
        assert stored.config['nested']['password'] == 'nested-ui-secret'
        assert stored.config['temperature'] == 0.3


def test_team_edit_updates_members_together_and_rejects_invalid_reference(client, admin):
    first = create(client, admin, '/agents')
    second = create(client, admin, '/agents')
    team = create(client, admin, '/agent-teams', topology='pipeline', members=[{'agent_id': first['id'], 'description': 'old'}])
    path = f"/api/v1/agent-teams/{team['id']}"
    initial = client.get(path, headers=admin).json()
    rejected = client.put(path, headers=admin, json={'name': 'must-not-save', 'members': [{'agent_id': 'missing-agent'}]})
    assert rejected.status_code == 404
    unchanged = client.get(path, headers=admin).json()
    assert unchanged['name'] == initial['name']
    assert unchanged['members'][0]['id'] == initial['members'][0]['id']
    response = client.put(path, headers=admin, json={'name': 'edited-team', 'spec': {'topology': 'pipeline'}, 'members': [{'agent_id': second['id'], 'description': 'replacement'}]})
    assert response.status_code == 200, response.text
    edited = client.get(path, headers=admin).json()
    assert edited['name'] == 'edited-team'
    assert len(edited['members']) == 1
    assert edited['members'][0]['agent_id'] == second['id']
    assert edited['members'][0]['spec']['description'] == 'replacement'
    assert client.get('/api/v1/agent-teams?name=edited-team', headers=admin).json()['total'] >= 1


def test_version_list_and_release_respect_selected_agent(client, admin):
    first = create(client, admin, '/agents')
    second = create(client, admin, '/agents')
    version1 = client.post(f"/api/v1/agents/{first['id']}/versions", headers=admin, json={'version': 'ui-1'}).json()
    version2 = client.post(f"/api/v1/agents/{second['id']}/versions", headers=admin, json={'version': 'ui-2'}).json()
    listing = client.get(f"/api/v1/agents/{first['id']}/versions", headers=admin).json()
    assert listing['total'] == 1
    assert listing['items'][0]['id'] == version1['id']
    assert client.post(f"/api/v1/agents/{first['id']}/release", headers=admin, json={'version_id': version2['id']}).status_code == 422
    assert client.post(f"/api/v1/agents/{first['id']}/rollback", headers=admin, json={'target_version': 'does-not-exist'}).status_code == 404
    assert client.post(f"/api/v1/agents/{first['id']}/release", headers=admin, json={'version_id': version1['id']}).status_code == 200
