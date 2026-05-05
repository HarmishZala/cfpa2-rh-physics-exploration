"""Scene builder — converts CV detections into a structured dict for the VLM."""

from __future__ import annotations

from .detector import Detection


def build_scene_state(
    detections: list[Detection],
    frame_width: int,
    frame_height: int,
    step_idx: int = 0,
    artifacts_found: int = 0,
    artifacts_total: int = 0,
    exploration_history: list[str] | None = None,
) -> dict:
    """Build a structured scene description for the VLM.

    This mirrors the role of ``core.map_serializer.build_map_state_json()``
    but for real-world camera input instead of a 2D occupancy grid.
    """
    det_list = []
    for i, d in enumerate(detections):
        det_list.append(
            {
                "idx": i,
                "label": d.label,
                "confidence": round(d.confidence, 3),
                "region": d.region,
                "center_x": d.center_x,
                "center_y": d.center_y,
                "area_fraction": round(d.area_fraction, 4),
                "bbox": list(d.bbox),
            }
        )

    return {
        "frame_width": frame_width,
        "frame_height": frame_height,
        "step_idx": step_idx,
        "detections": det_list,
        "detection_count": len(det_list),
        "artifacts_found": artifacts_found,
        "artifacts_total": artifacts_total,
        "recent_directions": exploration_history[-5:] if exploration_history else [],
    }


def summarise_detections(detections: list[Detection]) -> str:
    """One-line human-readable summary of current detections for the VLM prompt."""
    if not detections:
        return "No objects detected in the current view."

    parts = []
    for d in detections:
        parts.append(f"{d.label} ({d.confidence:.0%}, {d.region})")
    return "Detected: " + ", ".join(parts)
