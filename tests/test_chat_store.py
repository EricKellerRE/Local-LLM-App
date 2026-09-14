import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from local_model_app.chat_store import ChatStore


class ChatStoreTests(unittest.TestCase):
    def test_chat_round_trip_and_resume_order(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            store = ChatStore(Path(directory) / "chats.sqlite3")
            chat = store.create_chat()
            store.append_exchange(chat["id"], "First question", "First answer")
            store.append_exchange(chat["id"], "Second question", "Second answer")
            self.assertEqual(store.get_chat(chat["id"])["title"], "First question")
            self.assertEqual(
                [item["role"] for item in store.messages(chat["id"])],
                ["user", "assistant", "user", "assistant"],
            )
            self.assertEqual(store.list_chats()[0]["id"], chat["id"])
            store.close()

    def test_delete_chat_removes_messages(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            store = ChatStore(Path(directory) / "chats.sqlite3")
            chat = store.create_chat("Disposable")
            store.append_exchange(chat["id"], "hello", "goodbye")
            store.delete_chat(chat["id"])
            self.assertEqual(store.list_chats(), [])
            with self.assertRaises(KeyError):
                store.get_chat(chat["id"])
            store.close()

    def test_plugin_selection_is_per_chat_and_cascades_on_delete(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            store = ChatStore(Path(directory) / "chats.sqlite3")
            first = store.create_chat("First")
            second = store.create_chat("Second")
            store.set_plugin_selected(first["id"], "example.plugin", selected=True)
            self.assertEqual(store.selected_plugins(first["id"]), ["example.plugin"])
            self.assertEqual(store.selected_plugins(second["id"]), [])
            self.assertEqual(store.plugin_selection_count("example.plugin"), 1)
            store.delete_chat(first["id"])
            self.assertEqual(store.plugin_selection_count("example.plugin"), 0)
            store.close()

    def test_archive_is_lossless_and_reversible(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            store = ChatStore(Path(directory) / "chats.sqlite3")
            chat = store.create_chat("Keep me")
            store.append_exchange(chat["id"], "question", "answer")
            scratchpad_path = Path(directory) / "scratchpads" / f"{chat['id']}.jsonl"
            scratchpad_path.write_text('{"kind":"plan","content":"remember me"}\n', encoding="utf-8")
            activity_path = Path(directory) / "tool_activity" / f"{chat['id']}.jsonl"
            activity_path.write_text('{"event":"tool_call","tool":"example__read"}\n', encoding="utf-8")
            store.set_plugin_selected(chat["id"], "example.plugin", selected=True)
            archived = store.archive_chat(chat["id"])
            self.assertIsNotNone(archived["archived_at"])
            self.assertEqual(store.list_chats(), [])
            self.assertEqual(store.list_chats(archived=True)[0]["id"], chat["id"])
            self.assertEqual(
                [item["content"] for item in store.archived_messages(chat["id"])],
                ["question", "answer"],
            )
            archive_path = Path(directory) / "archives" / f"{chat['id']}.json.gz"
            self.assertEqual(archive_path.read_bytes()[:2], b"\x1f\x8b")
            self.assertFalse(scratchpad_path.exists())
            self.assertFalse(activity_path.exists())
            self.assertEqual(store.archived_plugins(chat["id"]), ["example.plugin"])
            restored = store.archive_chat(chat["id"], archived=False)
            self.assertIsNone(restored["archived_at"])
            self.assertEqual(store.list_chats()[0]["id"], chat["id"])
            self.assertEqual([item["content"] for item in store.messages(chat["id"])], ["question", "answer"])
            self.assertIn("remember me", scratchpad_path.read_text(encoding="utf-8"))
            self.assertIn("example__read", activity_path.read_text(encoding="utf-8"))
            self.assertEqual(store.selected_plugins(chat["id"]), ["example.plugin"])
            self.assertFalse(archive_path.exists())
            store.close()


if __name__ == "__main__":
    unittest.main()
