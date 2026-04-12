# ASCII-ONLY
from __future__ import annotations

import json
import os
import tempfile
import unittest


class AppSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_settings_path = os.environ.get("QM_USER_SETTINGS_PATH")
        self._old_api_key = os.environ.get("CRYPTOCOM_API_KEY")
        self._old_api_secret = os.environ.get("CRYPTOCOM_API_SECRET")
        os.environ["QM_USER_SETTINGS_PATH"] = os.path.join(self._tmpdir.name, "user_settings.json")
        os.environ.pop("CRYPTOCOM_API_KEY", None)
        os.environ.pop("CRYPTOCOM_API_SECRET", None)

        from config.manager import ConfigManager

        ConfigManager._store = {}
        ConfigManager._loaded = False

    def tearDown(self) -> None:
        from config.manager import ConfigManager

        ConfigManager._store = {}
        ConfigManager._loaded = False
        if self._old_settings_path is None:
            os.environ.pop("QM_USER_SETTINGS_PATH", None)
        else:
            os.environ["QM_USER_SETTINGS_PATH"] = self._old_settings_path
        if self._old_api_key is None:
            os.environ.pop("CRYPTOCOM_API_KEY", None)
        else:
            os.environ["CRYPTOCOM_API_KEY"] = self._old_api_key
        if self._old_api_secret is None:
            os.environ.pop("CRYPTOCOM_API_SECRET", None)
        else:
            os.environ["CRYPTOCOM_API_SECRET"] = self._old_api_secret
        self._tmpdir.cleanup()

    def test_config_manager_persists_settings_to_file(self) -> None:
        from config.manager import ConfigManager

        ConfigManager.set("prediction.base_quote", "USD")
        ConfigManager.set("prediction.max_assets", 24)

        path = ConfigManager.path()
        self.assertTrue(os.path.isfile(path))
        with open(path, "r", encoding="ascii") as handle:
            payload = json.load(handle)

        self.assertEqual(payload["prediction.base_quote"], "USD")
        self.assertEqual(payload["prediction.max_assets"], 24)

        ConfigManager._store = {}
        ConfigManager._loaded = False
        self.assertEqual(ConfigManager.get("prediction.base_quote"), "USD")
        self.assertEqual(ConfigManager.get("prediction.max_assets"), 24)

    def test_crypto_com_api_reads_keys_from_config_manager(self) -> None:
        from config.manager import ConfigManager
        from prediction_engine.crypto_com_api import CryptoComAPI

        ConfigManager.set("prediction.api_key", "abc123")
        ConfigManager.set("prediction.api_secret", "secret456")

        api = CryptoComAPI()

        self.assertEqual(api._api_key, "abc123")
        self.assertEqual(api._api_secret, "secret456")


if __name__ == "__main__":
    unittest.main()