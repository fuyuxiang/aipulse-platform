"""Opt-in review smoke against an explicitly configured test model.

Uses synthetic data and an isolated temporary SQLite database. Never prints secrets.
Run from repo root: PYTHONPATH=backend python3 scripts/review_real_model_smoke.py
Requires LLM_API_KEY, LLM_BASE_URL, LLM_MODEL in the process environment.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

root = Path(tempfile.mkdtemp(prefix='aipulse-real-review-', dir='/private/tmp'))
for name, sub in [('DATA_DIR', 'data'), ('OBJECT_STORE_DIR', 'files'), ('VECTOR_STORE_DIR', 'vectors'), ('LOG_PATH', 'backend.jsonl'), ('TRACE_PATH', 'traces.jsonl')]:
    os.environ['AIPULSE_' + name] = str(root / sub)
os.environ['AIPULSE_DATABASE_URL'] = f'sqlite:///{root / "smoke.db"}'
os.environ['AIPULSE_SCHEDULER_ENABLED'] = 'false'

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.db.init_data import initialize_defaults  # noqa: E402

with SessionLocal() as db:
    initialize_defaults(db)

parser = argparse.ArgumentParser()
parser.add_argument('--base-url', default=os.getenv('LLM_BASE_URL'))
parser.add_argument('--model', default=os.getenv('LLM_MODEL'))
parser.add_argument('--tool-model', help='A model from the same endpoint with function calling support')
parser.add_argument('--extended', action='store_true', help='Also verify configured embedding/rerank, a real tool call, and a three-agent pipeline')
parser.add_argument('--no-auth', action='store_true', help='Use an explicitly authorized endpoint that requires no credential')
args = parser.parse_args()
if not args.base_url or not args.model or (not args.no_auth and not os.getenv('LLM_API_KEY')):
    raise SystemExit('base URL, model and credentials (or --no-auth) are required')

with TestClient(app) as client:
    login = client.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'admin123456'})
    login.raise_for_status()
    auth = {'Authorization': f"Bearer {login.json()['access_token']}"}
    def post(path, payload, headers=None):
        response = client.post('/api/v1' + path, headers=headers or auth, json=payload)
        if response.status_code != 200:
            # Error text can contain a provider response; preserve only the status in the public evidence.
            raise RuntimeError(f'{path}: HTTP {response.status_code}')
        return response.json()
    model = post('/models', {'name': 'Review real model', 'model_type': 'chat_llm', 'provider_type': 'openai_compatible',
        'model_id': args.model, 'config': {'api_key_env': '' if args.no_auth else 'LLM_API_KEY', 'auth_scheme': 'none' if args.no_auth else 'bearer', 'base_url': args.base_url, 'timeout_seconds': 60}})
    agent = post('/agents', {'name': 'Review evidence analyst', 'config': {'model_id': model['id'], 'system_prompt': 'Read only the supplied synthetic facts. Reply with the sum as digits only.', 'max_tokens': 512, 'memory_policy': {'enabled': False}}})
    role = post('/roles', {'name': 'Review caller', 'code': 'review-caller', 'config': {'permissions': ['agents:read', 'agents:write', 'chat:read', 'chat:write']}})
    user = post('/users', {'username': 'review-caller', 'password': 'review-local-password'})
    bound = client.put(f"/api/v1/users/{user['id']}/roles", headers=auth, json=[role['id']])
    bound.raise_for_status()
    ordinary = post('/auth/login', {'username': 'review-caller', 'password': 'review-local-password'})
    caller = {'Authorization': f"Bearer {ordinary['access_token']}"}
    session = post('/chat/sessions', {'agent_id': agent['id']}, caller)
    response = client.post(f"/api/v1/chat/sessions/{session['id']}/stream", headers=caller, json={'content': 'Synthetic ledger: project A has 17 units and project B has 25 units. What is the total?'})
    response.raise_for_status()
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert not any(e['type'] == 'error' for e in events), 'provider invocation failed'
    answer = ''.join(e.get('delta', '') for e in events)
    assert answer.strip() == '42', f'Unexpected answer: {answer[:100]}'
    saved = client.get(f"/api/v1/chat/sessions/{session['id']}/messages", headers=caller).json()
    assert saved['total'] == 2 and saved['items'][-1]['spec']['content'].strip() == '42'
    publication = post(f"/agents/{agent['id']}/publish-api", {'name': 'review-api'})
    consumed = post(f"/published/{agent['id']}/chat", {'content': 'Synthetic facts: A=19, B=23. Sum A and B.'}, {'X-API-Key': publication['api_key']})
    assert consumed['response'].strip() == '42'
    evidence = {'date': '2026-11-04', 'provider': 'openai_compatible', 'model': args.model, 'ordinary_user_chat': 'passed', 'saved_messages': saved['total'], 'answer': answer.strip(), 'published_api': 'passed', 'published_answer': consumed['response'].strip(), 'isolated_database': str(root / 'smoke.db')}
    if args.extended:
        endpoint_config = {'auth_scheme': 'none' if args.no_auth else 'bearer', 'api_key_env': '' if args.no_auth else 'LLM_API_KEY', 'base_url': args.base_url, 'timeout_seconds': 60}
        embedding = post('/models', {'name': 'Review embedding', 'provider_type': 'openai_compatible', 'model_type': 'embedding', 'model_id': 'embed-pro', 'config': endpoint_config})
        probe = post(f"/models/{embedding['id']}/test-embedding", {'texts': ['Synthetic budget evidence']})
        dimensions = len(probe['result']['embeddings'][0])
        rerank = post('/models', {'name': 'Review rerank', 'provider_type': 'openai_compatible', 'model_type': 'rerank', 'model_id': 'rerank-pro', 'config': {**endpoint_config, 'rerank_url': args.base_url.rstrip('/') + '/rerank'}})
        kb = post('/knowledge-bases', {'name': 'Review source ledger', 'config': {'embedding_model_id': embedding['id'], 'embedding_dimensions': dimensions, 'rerank_model_id': rerank['id']}})
        doc = post(f"/knowledge-bases/{kb['id']}/documents", {'filename': 'budget-source.txt', 'content': 'Source S1: The Bluebridge project has equipment budget 17 and training budget 25. Sum these to calculate its total budget.'})
        post(f"/knowledge-bases/{kb['id']}/documents", {'filename': 'unrelated.txt', 'content': 'Source S2: The unrelated Redbridge project has marketing budget 999. It is not the Bluebridge project.'})
        tool = post('/tools', {'name': 'review_sum', 'code': 'review_sum', 'config': {'type': 'calculator'}, 'spec': {'schema': {'type': 'object', 'required': ['numbers'], 'properties': {'numbers': {'type': 'array', 'items': {'type': 'number'}}}}}})
        tool_model = model
        if args.tool_model:
            tool_model = post('/models', {'name': 'Review function model', 'provider_type': 'openai_compatible', 'model_type': 'chat_llm', 'model_id': args.tool_model, 'config': endpoint_config})
        grounded = post('/agents', {'name': 'Review grounded calculator', 'config': {'model_id': tool_model['id'], 'tool_choice': 'required', 'system_prompt': 'Use retrieved evidence only. You MUST call review_sum to calculate the sum before answering. Cite the supplied knowledge source using [K1] and source S1. Never use the unrelated Redbridge budget.', 'knowledge_base_ids': [kb['id']], 'tool_ids': [tool['id']], 'memory_policy': {'enabled': False}}})
        answer = post(f"/agents/{grounded['id']}/run", {'prompt': 'What is the total Bluebridge budget? Show the source and calculated total.'}, caller)
        assert answer['tool_calls'] and answer['tool_calls'][0]['output']['value'] == 42
        assert any(source['document_id'] == doc['id'] for source in answer['rag']['sources'])
        assert '42' in answer['response'] and ('K1' in answer['response'] or 'S1' in answer['response'])
        evidence['rag_tool_task'] = {'status': 'passed', 'embedding_dimensions': dimensions, 'tool_model': args.tool_model or args.model, 'tool_value': answer['tool_calls'][0]['output']['value'], 'source_document_matched': True, 'response': answer['response']}
        members = []
        for index, instruction in enumerate([
            'Extract the synthetic facts and source IDs from the user input. Preserve all facts. Do not compute.',
            'Compute the sum of Bluebridge equipment and training budgets. Preserve the evidence and Source S1.',
            'Produce a concise comparison report. Include the Bluebridge total and source S1. Mention Redbridge is unrelated. Use only previous analysis.'
        ]):
            item = post('/agents', {'name': f'Review worker {index}', 'config': {'model_id': model['id'], 'system_prompt': instruction, 'memory_policy': {'enabled': False}}})
            members.append({'agent_id': item['id'], 'priority': index, 'description': instruction})
        team = post('/agent-teams', {'name': 'Review evidence pipeline', 'topology': 'pipeline', 'members': members, 'shared_memory_enabled': False, 'enable_failover': False, 'timeout_seconds': 240})
        team_result = post(f"/agent-teams/{team['id']}/run", {'prompt': 'Source S1: Bluebridge equipment budget is 17, training budget is 25. Source S2: Redbridge marketing budget is 999 and unrelated. Deliver a comparison report with sources.'})
        assert team_result['status'] == 'completed' and len(team_result['messages']) == 3
        assert '42' in team_result['response'] and 'S1' in team_result['response']
        evidence['team_pipeline'] = {'status': 'passed', 'member_messages': len(team_result['messages']), 'response': team_result['response'], 'acceptance_mechanism': 'not_implemented'}
    (root / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    print(json.dumps(evidence, ensure_ascii=False))
