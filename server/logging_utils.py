from __future__ import annotations

import logging
import sys


def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(lvl)
    root.handlers.clear()
    root.addHandler(handler)


__all__ = ["setup_logging"]
