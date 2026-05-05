"""Real-world VLM client — sends camera frames + CV detections to a VLM for navigation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from core.vlm_interface import build_vlm_client, BaseVLMClient

from .scene_builder import summarise_detections


@dataclass
class RealWorldVLMResponse:
    """Parsed navigation response from the VLM."""

    direction: str  # "forward", "left", "right", "back", "stop"
    angle_degrees: float  # suggested turn angle (0 = straight)
    confidence: float  # 0-1
    artifact_found: dict | None  # {"name": "...", "position": "left|center|right"} or None
    reasoning: str
    raw_response: str


_VALID_DIRECTIONS = {"forward", "left", "right", "back", "stop"}


class RealWorldVLMClient:
    """Wraps an existing VLM backend with a real-world exploration prompt.

    Instead of selecting frontiers on a grid map, this client asks the VLM
    to choose a navigation direction based on a live camera image and
    structured CV detection results.
    """

    def __init__(
        self,
        backend: str,
        api_key: str,
        model: str | None = None,
    ) -> None:
        self._backend_name = backend
        self._inner: BaseVLMClient = build_vlm_client(backend, api_key)
        # For direct API calls we need the raw SDK clients
        self._backend = backend.lower().strip()
        self._api_key = api_key
        self._model = model

    def query(
        self,
        goal_prompt: str,
        scene_state: dict,
        image_b64: str,
        artifact_names: list[str] | None = None,
    ) -> RealWorldVLMResponse:
        """Query the VLM with a camera frame and scene context.

        Parameters
        ----------
        goal_prompt   : natural-language exploration goal
        scene_state   : dict from build_scene_state()
        image_b64     : base64-encoded JPEG of the current camera frame
        artifact_names: list of target artifact names to search for
        """
        prompt = _build_realworld_prompt(goal_prompt, scene_state, artifact_names)
        raw = self._call_backend(prompt, image_b64)
        return _parse_realworld_response(raw)

    # ------------------------------------------------------------------
    # Backend dispatch — reuses the same SDK patterns as core/vlm_interface.py
    # ------------------------------------------------------------------

    def _call_backend(self, prompt: str, image_b64: str) -> str:
        if self._backend == "groq":
            return self._call_groq(prompt, image_b64)
        elif self._backend in ("claude", "claude_haiku", "anthropic"):
            return self._call_claude(prompt, image_b64)
        elif self._backend == "gemini":
            return self._call_gemini(prompt, image_b64)
        else:
            raise ValueError(f"Unknown backend: {self._backend}")

    def _call_groq(self, prompt: str, image_b64: str) -> str:
        from groq import Groq  # type: ignore

        client = Groq(api_key=self._api_key)
        model = self._model or "meta-llama/llama-4-scout-17b-16e-instruct"
        response = client.chat.completions.create(
            model=model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.choices[0].message.content or ""

    def _call_claude(self, prompt: str, image_b64: str) -> str:
        import anthropic  # type: ignore

        client = anthropic.Anthropic(api_key=self._api_key)
        model = self._model or "claude-haiku-4-5-20251001"
        response = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.content[0].text if response.content else ""

    def _call_gemini(self, prompt: str, image_b64: str) -> str:
        import base64

        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model or "gemini-1.5-flash")
        image_part = {
            "mime_type": "image/jpeg",
            "data": base64.b64decode(image_b64),
        }
        response = model.generate_content([prompt, image_part])
        return response.text or ""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_realworld_prompt(
    goal_prompt: str,
    scene_state: dict,
    artifact_names: list[str] | None = None,
) -> str:
    detections = scene_state.get("detections", [])
    if detections:
        det_lines = "\n".join(
            f"  [{d['idx']}] {d['label']} (conf={d['confidence']:.0%}, region={d['region']}, "
            f"size={d['area_fraction']:.1%} of frame)"
            for d in detections
        )
    else:
        det_lines = "  (no objects detected)"

    artifact_str = ", ".join(artifact_names) if artifact_names else "any notable objects"

    recent = scene_state.get("recent_directions", [])
    history_str = " → ".join(recent) if recent else "(just started)"

    return (
        f"You are an AI agent guiding a robot through a real-world environment.\n"
        f"The image shows the robot's current LIVE camera view.\n\n"
        f"GOAL: {goal_prompt}\n"
        f"TARGET ARTIFACTS TO FIND: {artifact_str}\n\n"
        f"Objects detected by computer vision:\n"
        f"{det_lines}\n\n"
        f"Exploration step: {scene_state.get('step_idx', 0)}\n"
        f"Artifacts found: {scene_state.get('artifacts_found', 0)} / "
        f"{scene_state.get('artifacts_total', 0)}\n"
        f"Recent navigation: {history_str}\n\n"
        f"Based on what you see in the camera image and the detection results:\n"
        f"1. Decide which direction the robot should move to explore and find artifacts.\n"
        f"2. If you see a target artifact, identify it.\n\n"
        f"Respond ONLY with a JSON object (no markdown, no extra text):\n"
        f'{{"direction": "<forward|left|right|back|stop>", '
        f'"angle_degrees": <0-180>, '
        f'"confidence": <0.0-1.0>, '
        f'"artifact_found": {{"name": "<artifact name>", "position": "<left|center|right>"}} or null, '
        f'"reasoning": "<brief explanation>"}}'
    )


# ---------------------------------------------------------------------------
# Response parser (same multi-layer fallback pattern as core/vlm_interface.py)
# ---------------------------------------------------------------------------

def _parse_realworld_response(raw: str) -> RealWorldVLMResponse:
    """Parse navigation direction from VLM text output."""

    # Layer 1: find first JSON block containing "direction"
    match = re.search(r'\{[^{}]*"direction"[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            direction = str(data.get("direction", "forward")).lower().strip()
            if direction not in _VALID_DIRECTIONS:
                direction = "forward"

            angle = float(data.get("angle_degrees", 0))
            angle = max(0.0, min(angle, 180.0))

            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(confidence, 1.0))

            artifact = data.get("artifact_found")
            if artifact and not isinstance(artifact, dict):
                artifact = None

            reasoning = str(data.get("reasoning", ""))

            return RealWorldVLMResponse(
                direction=direction,
                angle_degrees=angle,
                confidence=confidence,
                artifact_found=artifact,
                reasoning=reasoning,
                raw_response=raw,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Layer 2: scan for direction keywords
    raw_lower = raw.lower()
    for d in _VALID_DIRECTIONS:
        if d in raw_lower:
            return RealWorldVLMResponse(
                direction=d,
                angle_degrees=45.0,
                confidence=0.3,
                artifact_found=None,
                reasoning=f"(direction '{d}' parsed from text)",
                raw_response=raw,
            )

    # Layer 3: safe fallback
    return RealWorldVLMResponse(
        direction="forward",
        angle_degrees=0.0,
        confidence=0.1,
        artifact_found=None,
        reasoning="(fallback — could not parse VLM response)",
        raw_response=raw,
    )
