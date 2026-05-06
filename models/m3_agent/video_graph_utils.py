"""Utilities for persisting ``VideoGraph`` instances to disk."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

from .video_graph import VideoGraph


LOGGER = logging.getLogger(__name__)


def save_video_graph(video_graph: VideoGraph, save_path: str) -> None:
    """Serialize ``video_graph`` to ``save_path`` using pickle."""

    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("wb") as handle:
        pickle.dump(video_graph, handle)

    LOGGER.info("Video graph saved to %s", destination)


def load_video_graph(video_graph_path: str) -> Optional[VideoGraph]:
    """Deserialize a ``VideoGraph`` from ``video_graph_path`` if it exists."""

    source = Path(video_graph_path)
    if not source.exists():
        LOGGER.warning("Video graph not found at %s", source)
        return None

    with source.open("rb") as handle:
        LOGGER.info("Loading video graph from %s", source)
        return pickle.load(handle)
