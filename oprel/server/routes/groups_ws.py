from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class ConnectionManager:
    def __init__(self):
        # Maps group_id to a list of active websocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, group_id: str):
        await websocket.accept()
        if group_id not in self.active_connections:
            self.active_connections[group_id] = []
        self.active_connections[group_id].append(websocket)
        logger.info(f"WebSocket connected to group {group_id}. Total connections: {len(self.active_connections[group_id])}")

    def disconnect(self, websocket: WebSocket, group_id: str):
        if group_id in self.active_connections:
            if websocket in self.active_connections[group_id]:
                self.active_connections[group_id].remove(websocket)
            if not self.active_connections[group_id]:
                del self.active_connections[group_id]
            logger.info(f"WebSocket disconnected from group {group_id}.")

    async def broadcast_to_group(self, group_id: str, message: dict):
        if group_id in self.active_connections:
            # We iterate over a copy in case connections drop during iteration
            for connection in list(self.active_connections[group_id]):
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send to WebSocket in group {group_id}: {e}")
                    self.disconnect(connection, group_id)

manager = ConnectionManager()

@router.websocket("/groups/{group_id}/ws")
async def websocket_endpoint(websocket: WebSocket, group_id: str):
    await manager.connect(websocket, group_id)
    try:
        while True:
            # Currently a one-way street (Server -> Client)
            # but we keep the loop alive to detect disconnects.
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, group_id)
