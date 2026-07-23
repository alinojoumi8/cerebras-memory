"""Explicitly download and validate optional local retrieval models."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_settings  # noqa: E402
from reranking import FlashRankReranker  # noqa: E402
from stdio import configure_utf8_stdio  # noqa: E402


def main() -> int:
    configure_utf8_stdio()
    settings = load_settings()
    if not settings.reranker.enabled:
        print(json.dumps({"reranker": {"enabled": False, "status": "disabled"}}))
        return 0
    status = FlashRankReranker(settings.reranker).warm()
    print(json.dumps({"reranker": status}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
