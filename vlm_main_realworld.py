"""Real-world VLM-guided exploration entry point.

Uses a live camera feed + YOLO-World CV detection + VLM reasoning
to guide exploration and identify user-specified artifacts.

Usage:
    python vlm_main_realworld.py \\
        --goal "explore the room and find artifacts" \\
        --artifacts "red ball,blue cone,fire extinguisher" \\
        --vlm-backend groq

    # Use iPhone via Continuity Camera (usually device 1)
    python vlm_main_realworld.py \\
        --goal "find the exit sign" \\
        --artifacts "exit sign" \\
        --camera 1 --vlm-backend gemini
"""
from __future__ import annotations

import os
import sys
import time

# Load .env file (API keys etc.)
from dotenv import load_dotenv
load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

import argparse
import json
from pathlib import Path

import cv2

from realworld.camera import CameraProvider
from realworld.detector import ObjectDetector
from realworld.display import Display
from realworld.scene_builder import build_scene_state, summarise_detections
from realworld.realworld_vlm import RealWorldVLMClient, RealWorldVLMResponse
from realworld.artifact_tracker import ArtifactTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-world VLM-guided exploration")
    p.add_argument(
        "--goal", type=str, required=True,
        help='Natural language goal, e.g. "explore and find the red ball"',
    )
    p.add_argument(
        "--artifacts", type=str, required=True,
        help='Comma-separated artifact names, e.g. "red ball,blue cone,exit sign"',
    )
    p.add_argument("--camera", type=int, default=0, help="Camera device index (0=webcam, 1=iPhone)")
    p.add_argument("--width", type=int, default=1280, help="Camera resolution width")
    p.add_argument("--height", type=int, default=720, help="Camera resolution height")
    p.add_argument(
        "--vlm-backend", type=str, default="groq",
        choices=["groq", "gemini", "claude"],
    )
    p.add_argument("--vlm-model", type=str, default=None, help="Override VLM model name")
    p.add_argument(
        "--vlm-interval", type=float, default=3.0,
        help="Seconds between VLM queries (default 3.0)",
    )
    p.add_argument(
        "--model-size", type=str, default="s",
        choices=["s", "m", "l"],
        help="YOLO-World model size: s(mall), m(edium), l(arge)",
    )
    p.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--confirm-frames", type=int, default=3, help="Frames to confirm an artifact")
    p.add_argument("--no-display", action="store_true", help="Headless mode (print commands only)")
    p.add_argument("--output-dir", type=str, default="outputs/realworld", help="Output directory for logs")
    p.add_argument("--max-steps", type=int, default=500, help="Maximum exploration steps")
    return p.parse_args()


def _resolve_api_key(backend: str) -> str:
    """Look up API key from environment variables."""
    key_map = {
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }
    env_var = key_map.get(backend, "")
    key = os.environ.get(env_var, "")
    if not key:
        print(f"ERROR: {env_var} not set. Export it or add to .env file.")
        sys.exit(1)
    return key


def main() -> None:
    args = parse_args()
    artifact_names = [name.strip() for name in args.artifacts.split(",") if name.strip()]

    if not artifact_names:
        print("ERROR: --artifacts must specify at least one target.")
        sys.exit(1)

    # Resolve API key
    api_key = _resolve_api_key(args.vlm_backend)

    # Prepare output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Real-World VLM Exploration")
    print("=" * 60)
    print(f"  Goal:       {args.goal}")
    print(f"  Artifacts:  {artifact_names}")
    print(f"  Camera:     device {args.camera} ({args.width}x{args.height})")
    print(f"  CV Model:   YOLO-World-{args.model_size}")
    print(f"  VLM:        {args.vlm_backend}" + (f" ({args.vlm_model})" if args.vlm_model else ""))
    print(f"  VLM query interval: {args.vlm_interval}s")
    print("=" * 60)
    print()

    # Initialise components
    print("[Init] Opening camera...")
    camera = CameraProvider(device=args.camera, width=args.width, height=args.height)

    print(f"[Init] Loading YOLO-World-{args.model_size}...")
    detector = ObjectDetector(model_size=args.model_size, confidence=args.confidence)
    detector.set_classes(artifact_names)

    print(f"[Init] Connecting to {args.vlm_backend} VLM...")
    vlm = RealWorldVLMClient(
        backend=args.vlm_backend,
        api_key=api_key,
        model=args.vlm_model,
    )

    tracker = ArtifactTracker(
        target_names=artifact_names,
        confirm_frames=args.confirm_frames,
    )

    display = Display() if not args.no_display else None

    # Exploration log
    log: list[dict] = []
    exploration_history: list[str] = []
    last_vlm_response: RealWorldVLMResponse | None = None
    last_vlm_time: float = 0.0
    step = 0
    fps_timer = time.time()
    fps = 0.0

    print()
    print("[Running] Press 'q' to quit.")
    print()

    try:
        while step < args.max_steps:
            frame_start = time.time()
            step += 1

            # 1. Capture frame
            frame = camera.capture()
            h, w = frame.shape[:2]

            # 2. Run CV detection
            detections = detector.detect(frame)

            # 3. Update artifact tracker from CV
            newly_confirmed = tracker.update(detections, step)
            for art in newly_confirmed:
                print(f"  *** ARTIFACT CONFIRMED (CV): {art.name} at {art.position} (step {step}) ***")

            # 4. Query VLM periodically
            now = time.time()
            if now - last_vlm_time >= args.vlm_interval:
                scene_state = build_scene_state(
                    detections=detections,
                    frame_width=w,
                    frame_height=h,
                    step_idx=step,
                    artifacts_found=tracker.found_count,
                    artifacts_total=tracker.total_count,
                    exploration_history=exploration_history,
                )

                # Annotate frame for VLM (draw boxes before encoding)
                annotated = frame.copy()
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated, f"{det.label} {det.confidence:.0%}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )

                import base64
                _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

                try:
                    last_vlm_response = vlm.query(
                        goal_prompt=args.goal,
                        scene_state=scene_state,
                        image_b64=image_b64,
                        artifact_names=artifact_names,
                    )
                    exploration_history.append(last_vlm_response.direction)

                    # Check VLM artifact identification
                    vlm_artifact = tracker.update_from_vlm(
                        last_vlm_response.artifact_found, step,
                    )
                    if vlm_artifact:
                        print(f"  *** ARTIFACT CONFIRMED (VLM): {vlm_artifact.name} at {vlm_artifact.position} ***")

                    # Print navigation command
                    r = last_vlm_response
                    nav_cmd = {
                        "step": step,
                        "direction": r.direction,
                        "angle_degrees": r.angle_degrees,
                        "confidence": r.confidence,
                        "reasoning": r.reasoning,
                    }
                    print(f"  [Nav] {json.dumps(nav_cmd)}")

                    # Log entry
                    log.append({
                        "step": step,
                        "direction": r.direction,
                        "angle": r.angle_degrees,
                        "confidence": r.confidence,
                        "detections": summarise_detections(detections),
                        "artifacts_found": tracker.found_count,
                        "reasoning": r.reasoning,
                    })

                except Exception as e:
                    print(f"  [VLM Error] {e}")

                last_vlm_time = now

            # 5. Update display
            if display is not None:
                # Calculate FPS
                elapsed = time.time() - fps_timer
                if elapsed > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 0.001))
                fps_timer = time.time()

                display.update(
                    frame=frame,
                    detections=detections,
                    direction=last_vlm_response.direction if last_vlm_response else "",
                    reasoning=last_vlm_response.reasoning if last_vlm_response else "",
                    artifacts_found=tracker.found_count,
                    artifacts_total=tracker.total_count,
                    fps=fps,
                )
                key = display.poll_key(1)
                if key == ord("q"):
                    print("\n[Quit] User pressed 'q'.")
                    break

            # 6. Check completion
            if tracker.all_found:
                print(f"\n  *** ALL ARTIFACTS FOUND at step {step}! ***")
                break

    except KeyboardInterrupt:
        print("\n[Quit] Interrupted.")

    finally:
        camera.release()
        if display is not None:
            display.close()

    # Save exploration log
    if log:
        import pandas as pd
        log_csv = out_dir / "exploration_log.csv"
        pd.DataFrame(log).to_csv(log_csv, index=False)
        print(f"\n[Saved] Exploration log: {log_csv}")

    # Print final summary
    print()
    print("=" * 60)
    print("Exploration Summary")
    print("=" * 60)
    print(f"  Goal:              {args.goal}")
    print(f"  Steps:             {step}")
    print(f"  Artifacts found:   {tracker.found_count}/{tracker.total_count}")
    for art in tracker.found_artifacts:
        print(f"    - {art.name} (position: {art.position}, step: {art.confirmed_step})")
    print(f"  All found:         {tracker.all_found}")
    print("=" * 60)


if __name__ == "__main__":
    main()
