"""Tracker names shared by the API, the pipeline planner and the DeepStream worker.

NATIVE trackers run inside DeepStream (``nvtracker``, on the GPU, also follow objects on frames where the
detector is skipped). PYTHON trackers run in the worker process on the detector's boxes (pure NumPy);
they need a detection on every frame, so the detection interval is forced to 0 for them.
"""
from __future__ import annotations

from typing import Literal

NATIVE_TRACKERS = ("NvDCF", "NvSORT", "IOU", "NvDeepSORT")
PYTHON_TRACKERS = ("ByteTrack", "OCSORT", "BoTSORT", "StrongSORT")
ALL_TRACKERS = NATIVE_TRACKERS + PYTHON_TRACKERS

TrackerType = Literal["NvDCF", "NvSORT", "IOU", "NvDeepSORT", "ByteTrack", "OCSORT", "BoTSORT", "StrongSORT"]

# trackers that use appearance (Re-ID embeddings) when a 'reid' model is assigned to the camera
APPEARANCE_TRACKERS = ("NvDeepSORT", "BoTSORT", "StrongSORT")


def is_python_tracker(name: str | None) -> bool:
    return name in PYTHON_TRACKERS
