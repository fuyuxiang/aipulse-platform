# AIPulse Backend

FastAPI control plane for the local AIPulse enterprise agent platform.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e ../echo-agent
pip install -e .
alembic upgrade head
python ../scripts/init_db.py
python ../scripts/init_admin.py
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Default local account:

- tenant: `default`
- username: `admin`
- password: `admin123456`

The admin user is created with `must_change_password=true` for the first login flow.

