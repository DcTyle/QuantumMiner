# ASCII-ONLY
import unittest
import importlib.util
import os


def _load_sha256d_compute_share():
    here = os.path.dirname(os.path.abspath(__file__))
    mod_path = os.path.normpath(os.path.join(here, "..", "miner", "algos", "sha256d_pow.py"))
    spec = importlib.util.spec_from_file_location("sha256d_pow_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod.sha256d_compute_share


def _zero_header_hex():
    # 80-byte header, all zeros
    return (b"\x00" * 80).hex()


class TestBTCSha256d(unittest.TestCase):
    def test_sha256d_changes_with_nonce(self):
        sha256d_compute_share = _load_sha256d_compute_share()
        job = {"header_hex": _zero_header_hex()}
        s1 = sha256d_compute_share(job, 1)
        s2 = sha256d_compute_share(job, 2)
        h1 = str(s1.get("hash_hex", ""))
        h2 = str(s2.get("hash_hex", ""))
        self.assertEqual(len(h1), 64)
        self.assertEqual(len(h2), 64)
        self.assertNotEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
