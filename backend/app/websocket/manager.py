from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from fastapi import WebSocket


class WebSocketManager:
    def __init__(self) -> None:
        self._clients: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, channel: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients[channel].add(websocket)
        await websocket.send_json({"type": "connected", "channel": channel})

    def disconnect(self, channel: str, websocket: WebSocket) -> None:
        clients = self._clients.get(channel)
        if clients:
            clients.discard(websocket)

    async def broadcast(self, channel: str, event: dict[str, Any]) -> None:
        clients = list(self._clients.get(channel, set()))
        for websocket in clients:
            await websocket.send_text(json.dumps(event, ensure_ascii=False))


websocket_manager = WebSocketManager()

