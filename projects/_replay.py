from __future__ import annotations

import hashlib


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return max(1, int.from_bytes(digest.digest()[:8], "big") % (2**31 - 1))
