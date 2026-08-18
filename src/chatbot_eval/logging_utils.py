from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_file: Path | None, level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level: {level}")
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )
    # The Google SDK emits an informational AFC initialization line for ordinary generation calls
    # in some versions. Keep third-party transport chatter out of normal application logs; DEBUG
    # remains available when explicitly requested for troubleshooting.
    if numeric_level > logging.DEBUG:
        logging.getLogger("google_genai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
