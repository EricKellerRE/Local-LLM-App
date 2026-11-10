import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from local_model_app.config import _load_dotenv
from local_model_app.app_settings import AppSettingsStore
from scripts.export_portable import model_shards


class PortableTests(unittest.TestCase):
    def test_relative_paths_follow_bundle_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root/'.env').write_text('LOCAL_MODEL_ID=models/test.gguf\nGRID_WORKSHOP_ROOT=../Grid-Workshop\nLOCAL_LLAMA_SERVER=runtimes/llama-server.exe\n')
            with patch.dict(os.environ, {}, clear=True):
                _load_dotenv(root/'.env')
                self.assertEqual(os.environ['LOCAL_MODEL_ID'], str(root/'models/test.gguf'))
                self.assertEqual(os.environ['GRID_WORKSHOP_ROOT'], str((root.parent/'Grid-Workshop').resolve()))
            (root/'config').mkdir()
            (root/'config/app-settings.json').write_text('{"data_directory":"data","models_directory":"models"}')
            paths = AppSettingsStore(root).paths()
            self.assertEqual(paths.data_directory, root/'data')

    def test_all_shards_required(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory).resolve()/'test-00001-of-00002.gguf'
            first.touch()
            with self.assertRaisesRegex(ValueError, 'Missing'):
                model_shards(first)
            second = Path(directory).resolve()/'test-00002-of-00002.gguf'
            second.touch()
            self.assertEqual(model_shards(first), [first, second])
