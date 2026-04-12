"""
ASCII-ONLY
Mining algorithms integration

Exports only algorithms that are present in this bundle.
Avoid importing optional modules that may not be available
to keep package import side-effect free for test discovery.
"""

# Etchash (present in this repo)
try:
	from .etchash import etchash_mix_digest, etchash_compute_share  # type: ignore
except Exception:
	pass

# KawPow (optional, may be absent in this build)
try:
	from .kawpow import kawpow_compute_share  # type: ignore
except Exception:
	pass

# Scrypt (optional)
try:
	from .scrypt_pow import scrypt_compute_share, scrypt_hash_1024_1_1_256  # type: ignore
except Exception:
	pass

# SHA256d (present, pure python)
try:
	from .sha256d_pow import sha256d_compute_share  # type: ignore
except Exception:
	pass
