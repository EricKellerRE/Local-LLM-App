from __future__ import annotations

import httpx


class ServerClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=None)

    def _request(self, method: str, path: str, **kwargs):
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as exc:
            raise RuntimeError(f"The local model server is not running at {self.base_url}. Start server.ps1 first.") from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Local model server error ({exc.response.status_code}): {exc.response.text}") from exc

    def health(self):
        return self._request("GET", "/health")

    def create_chat(self, title: str = "New chat"):
        return self._request("POST", "/api/chats", json={"title": title})

    def list_chats(self):
        return self._request("GET", "/api/chats")

    def get_chat(self, chat_id: str):
        return self._request("GET", f"/api/chats/{chat_id}")

    def send(self, chat_id: str, content: str):
        return self._request("POST", f"/api/chats/{chat_id}/messages", json={"content": content})

    def scratchpad(self, chat_id: str):
        return self._request("GET", f"/api/chats/{chat_id}/scratchpad")
