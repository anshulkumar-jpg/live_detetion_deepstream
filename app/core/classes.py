"""Which classes of a model can be chosen per camera.

A model version lists its class names in ``labels``. ``nvinfer_properties`` may already exclude classes with
``filter-out-class-ids`` (for example the vehicle model is a COCO model that only keeps car, motorcycle, bus, truck).
The remaining classes are the ones a camera may select.
"""
from __future__ import annotations

from typing import Any


def parse_class_ids(raw: Any) -> set[int]:
    if raw in (None, ""):
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {int(x) for x in raw}
    return {int(x) for x in str(raw).replace(",", ";").split(";") if x.strip() != ""}


def class_count(props: dict[str, Any], labels: list[str]) -> int:
    return int(props.get("num-detected-classes") or len(labels) or 0)


def selectable_classes(props: dict[str, Any], labels: list[str]) -> list[tuple[int, str]]:
    """[(class_id, name)] a camera can choose from."""
    excluded = parse_class_ids(props.get("filter-out-class-ids"))
    return [(i, labels[i] if i < len(labels) else str(i)) for i in range(class_count(props, labels)) if i not in excluded]


def filter_out_for(props: dict[str, Any], labels: list[str], selected: list[int]) -> str:
    """nvinfer ``filter-out-class-ids`` value that keeps only ``selected``."""
    keep = set(selected)
    return ";".join(str(i) for i in range(class_count(props, labels)) if i not in keep)
