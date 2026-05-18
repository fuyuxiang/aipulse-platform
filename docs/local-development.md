# 本地开发

## 后端

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

## 前端

```bash
cd frontend
npm install
npm run dev
```

前端默认访问 `http://127.0.0.1:3000`，后端默认访问 `http://127.0.0.1:8000`。

默认账号：

- tenant: `default`
- username: `admin`
- password: `admin123456`

首次登录后应修改默认密码，默认用户带 `must_change_password=true`。

