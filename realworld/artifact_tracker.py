"""Artifact tracker — confirms detections across multiple frames to avoid false positives."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ConfirmedArtifact:
    """An artifact that has been confirmed across multiple frames."""

    name: str
    position: str  # "left", "center", "right" (last seen)
    first_seen_step: int
    confirmed_step: int
    timestamp: float


class ArtifactTracker:
    """Tracks object detections over time and confirms artifacts.

    An artifact is confirmed when the same class is detected in the same
    screen region for ``confirm_frames`` consecutive detection cycles.
    """

    def __init__(
        self,
        target_names: list[str],
        confirm_frames: int = 3,
        dedup_seconds: float = 10.0,
    ) -> None:
        self._targets = {n.lower() for n in target_names}
        self._confirm_frames = confirm_frames
        self._dedup_seconds = dedup_seconds

        # Pending: key = (label_lower, region) -> consecutive count + first_seen_step
        self._pending: dict[tuple[str, str], dict] = {}
        self._confirmed: list[ConfirmedArtifact] = []

    @property
    def total_count(self) -> int:
        return len(self._targets)

    @property
    def found_count(self) -> int:
        return len(self._confirmed)

    @property
    def found_artifacts(self) -> list[ConfirmedArtifact]:
        return list(self._confirmed)

    @property
    def all_found(self) -> bool:
        found_names = {a.name.lower() for a in self._confirmed}
        return self._targets.issubset(found_names)

    def update(
        self,
        detections: list,
        step_idx: int,
    ) -> list[ConfirmedArtifact]:
        """Process new detections and return any newly confirmed artifacts.

        Parameters
        ----------
        detections : list of Detection objects from detector.py
        step_idx   : current exploration step

        Returns
        -------
        List of newly confirmed artifacts (empty if none).
        """
        # Build set of (label, region) seen this frame
        seen: set[tuple[str, str]] = set()
        for det in detections:
            label = det.label.lower()
            if label in self._targets:
                seen.add((label, det.region))

        # Update pending counts
        new_confirmed: list[ConfirmedArtifact] = []
        keys_to_remove = []

        for key in list(self._pending):
            if key in seen:
                self._pending[key]["count"] += 1
                if self._pending[key]["count"] >= self._confirm_frames:
                    # Check dedup
                    if not self._is_duplicate(key[0], key[1]):
                        artifact = ConfirmedArtifact(
                            name=key[0],
                            position=key[1],
                            first_seen_step=self._pending[key]["first_step"],
                            confirmed_step=step_idx,
                            timestamp=time.time(),
                        )
                        self._confirmed.append(artifact)
                        new_confirmed.append(artifact)
                    keys_to_remove.append(key)
            else:
                # Reset if not seen
                keys_to_remove.append(key)

        for k in keys_to_remove:
            self._pending.pop(k, None)

        # Add newly seen entries
        for key in seen:
            if key not in self._pending:
                self._pending[key] = {"count": 1, "first_step": step_idx}

        return new_confirmed

    def update_from_vlm(
        self,
        artifact_found: dict | None,
        step_idx: int,
    ) -> ConfirmedArtifact | None:
        """Process an artifact_found dict from the VLM response.

        The VLM may identify artifacts that the CV model missed (or vice versa).
        VLM identifications are trusted immediately (no multi-frame confirmation).
        """
        if not artifact_found:
            return None

        name = str(artifact_found.get("name", "")).lower()
        position = str(artifact_found.get("position", "center"))

        if not name or name not in self._targets:
            return None

        if self._is_duplicate(name, position):
            return None

        artifact = ConfirmedArtifact(
            name=name,
            position=position,
            first_seen_step=step_idx,
            confirmed_step=step_idx,
            timestamp=time.time(),
        )
        self._confirmed.append(artifact)
        return artifact

    def _is_duplicate(self, name: str, position: str) -> bool:
        """Check if this artifact was already confirmed recently."""
        now = time.time()
        for a in self._confirmed:
            if a.name == name and (now - a.timestamp) < self._dedup_seconds:
                return True
        return False
