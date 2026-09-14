from __future__ import annotations

import os
from pathlib import Path

from local_model_app.client import ServerClient
from local_model_app.config import _load_dotenv


def main() -> None:
    root = Path(__file__).resolve().parent
    _load_dotenv(root / ".env")
    client = ServerClient(os.getenv("LOCAL_MODEL_SERVER_URL", "http://127.0.0.1:8765"))
    try:
        chat = client.create_chat()
    except RuntimeError as exc:
        print(exc)
        return
    chat_id = chat["id"]
    print("Local Model Chat. Commands: /new, /chats, /resume ID, /status, /scratchpad, /quit")

    while True:
        try:
            message = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if not message:
            continue
        if message == "/quit":
            return
        try:
            if message == "/new":
                chat_id = client.create_chat()["id"]
                print(f"Started new chat {chat_id}")
            elif message == "/chats":
                for item in client.list_chats():
                    marker = "*" if item["id"] == chat_id else " "
                    print(f"{marker} {item['id']}  {item['title']}")
            elif message.startswith("/resume "):
                resumed = client.get_chat(message.split(maxsplit=1)[1])
                chat_id = resumed["id"]
                print(f"Resumed: {resumed['title']}")
                for item in resumed["messages"]:
                    print(f"{item['role']}> {item['content']}")
            elif message == "/status":
                print(client.health())
            elif message == "/scratchpad":
                entries = client.scratchpad(chat_id)["entries"]
                print("\n".join(f"[{item['kind']}] {item['content']}" for item in entries) or "Scratchpad is empty.")
            else:
                print(f"\nassistant> {client.send(chat_id, message)['content']}")
        except RuntimeError as exc:
            print(f"\n{exc}")


if __name__ == "__main__":
    main()
