"""VST / VMS adapter interface (recording, storage, playback).

The NVIDIA reference architecture places VST (Video Storage Toolkit) beside the
perception layer for store-and-playback. Its REST API differs between releases,
so this module defines the contract only. A concrete adapter must be written
against the VST version you actually deploy (check its API docs) and enabled by
setting VST_BASE_URL. Real-time inference never depends on this adapter.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class VmsAdapter(ABC):
    name = "abstract"

    @abstractmethod
    async def register_camera(self, camera_id: str, rtsp_url: str, username: str | None, password: str | None) -> str: ...

    @abstractmethod
    async def remove_camera(self, camera_id: str) -> None: ...

    @abstractmethod
    async def start_recording(self, camera_id: str) -> None: ...

    @abstractmethod
    async def stop_recording(self, camera_id: str) -> None: ...

    @abstractmethod
    async def playback_url(self, camera_id: str, start: datetime, end: datetime) -> str: ...

    @abstractmethod
    async def health(self) -> dict[str, Any]: ...


class VmsNotConfigured(RuntimeError):
    pass


class DisabledVmsAdapter(VmsAdapter):
    """Used when no VST/VMS is deployed. Every call reports that clearly."""

    name = "disabled"

    async def _no(self, *_a, **_k):
        raise VmsNotConfigured("No VST/VMS adapter configured (set VST_BASE_URL and implement a VmsAdapter)")

    register_camera = remove_camera = start_recording = stop_recording = playback_url = _no  # type: ignore[assignment]

    async def health(self) -> dict[str, Any]:
        return {"adapter": self.name, "available": False}


def get_vms_adapter(settings) -> VmsAdapter:
    if settings.vst_base_url:
        raise VmsNotConfigured(
            "VST_BASE_URL is set but no VST adapter implementation is installed for this VST version. "
            "Implement app.services.vst.VmsAdapter against your VST API and register it here."
        )
    return DisabledVmsAdapter()
