#!/usr/bin/env python3
"""Serving entrypoint for the frozen-ALBERT Chinese movie recommender.

Loads a trained checkpoint (``out_probe/best.pt``) via
:class:`chinese_recommender.ChineseRanker` and exposes the browser chat
server defined in :mod:`chat_app`. CPU-only; no API keys required.

Run::

    python serve_chinese.py --host 0.0.0.0 --port 8000 \\
        --checkpoint out_probe/best.pt --movies TG-ReDial-Data/out/movies.json

``/api/health`` returns ``{"status": "ok"}`` for container health checks.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
_CODE_DIR = PROJECT_ROOT / "TG-ReDial-main"
if _CODE_DIR.is_dir() and str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from chat_app import make_handler
from chinese_recommender import ChineseRanker

logger = logging.getLogger("tgredial.serve")

DEFAULT_CHECKPOINT = PROJECT_ROOT / "out_probe" / "best.pt"
DATA_OUT = PROJECT_ROOT / "TG-ReDial-Data" / "out"
DEFAULT_MOVIES = DATA_OUT / "movies.json"
DEFAULT_TITLE_EMB = DATA_OUT / "title_embeddings_albert_tiny.npz"
DEFAULT_ORDER = DATA_OUT / "movie_id_order.json"


def load_ranker(checkpoint: Path, movies: Path | None = None,
                title_emb: Path | None = None, order: Path | None = None) -> ChineseRanker:
    """Load the trained probe + frozen encoder from disk.

    When ``movies``/``title_emb``/``order`` are omitted, ``ChineseRanker``
    auto-discovers them under ``TG-ReDial-Data/out/``.
    """
    import json
    import numpy as np

    if movies is not None and title_emb is not None and order is not None:
        import json as _json
        import numpy as _np
        movies_obj = _json.load(open(movies, encoding="utf-8"))
        title_emb_obj = _np.load(title_emb)["embeddings"]
        order_obj = _json.load(open(order, encoding="utf-8"))["ids"]
        return ChineseRanker.load(checkpoint, movies=movies_obj,
                                  title_emb=title_emb_obj, order=order_obj)
    return ChineseRanker.load(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Chinese movie recommender.")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(os.environ.get("CHECKPOINT", DEFAULT_CHECKPOINT)))
    parser.add_argument("--movies", type=Path,
                        default=Path(os.environ.get("MOVIES", DEFAULT_MOVIES)))
    parser.add_argument("--title-emb", type=Path,
                        default=Path(os.environ.get("TITLE_EMB", DEFAULT_TITLE_EMB)))
    parser.add_argument("--order", type=Path,
                        default=Path(os.environ.get("ORDER", DEFAULT_ORDER)))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.checkpoint.exists():
        parser.error(f"Checkpoint not found: {args.checkpoint}. Train first.")
    if not args.movies.exists():
        parser.error(f"Movies catalog not found: {args.movies}.")

    logger.info("Loading ranker from %s", args.checkpoint)
    ranker = load_ranker(args.checkpoint, args.movies, args.title_emb, args.order)
    logger.info("Ranker loaded (encoder + %d titles).", len(ranker.order))

    from http.server import ThreadingHTTPServer
    handler = make_handler(ranker)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    logger.info("Serving on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
