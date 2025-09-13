from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from typing import List, Tuple

from PIL import Image


def parse_frame_info(filename: str) -> Tuple[int, int]:
    """
    Parse filename like 'frame_000123_1715612345678.png' -> (index, timestamp_ms)
    Returns (index, timestamp_ms). If timestamp not present, returns (index, None).
    """
    m = re.match(r"frame_(\d+)_(\d+)\.png$", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    m2 = re.match(r"frame_(\d+)\.png$", filename)
    if m2:
        return int(m2.group(1)), None  # type: ignore
    return -1, None  # type: ignore


def find_latest_session(images_root: str) -> str | None:
    if not os.path.isdir(images_root):
        return None
    sessions = [d for d in os.listdir(images_root) if d.startswith("session_")]
    if not sessions:
        return None
    sessions.sort()
    return os.path.join(images_root, sessions[-1])


def load_frames(session_dir: str) -> Tuple[List[Image.Image], List[int]]:
    files = [f for f in os.listdir(session_dir) if f.endswith(".png")]
    items: List[Tuple[int, int | None, str]] = []
    for f in files:
        idx, ts = parse_frame_info(f)
        if idx >= 0:
            items.append((idx, ts, f))
    items.sort(key=lambda x: x[0])
    frames: List[Image.Image] = []
    timestamps: List[int] = []
    for _, ts, f in items:
        img = Image.open(os.path.join(session_dir, f)).convert("P", palette=Image.ADAPTIVE, colors=128)
        frames.append(img)
        timestamps.append(ts if ts is not None else 0)
    return frames, timestamps


def compute_durations_ms(timestamps: List[int], speed: float = 1.0, min_ms: int = 5, max_ms: int = 250) -> List[int]:
    if not timestamps:
        return []
    durations: List[int] = []
    prev = timestamps[0]
    # Default first-frame duration (scaled)
    first_ms = int(round(33 / max(1e-6, speed)))
    first_ms = max(min_ms, min(first_ms, max_ms))
    durations.append(first_ms)
    for ts in timestamps[1:]:
        base = ts - prev if ts and prev else 33
        # Scale by speed (higher speed -> shorter duration)
        scaled = int(round(max(1, base) / max(1e-6, speed)))
        # Clamp to reasonable bounds
        scaled = max(min_ms, min(scaled, max_ms))
        durations.append(scaled)
        prev = ts
    return durations


def main():
    base_dir = os.environ.get("MAIN_DIRECTORY", os.getcwd())
    gifs_dir = os.path.join(base_dir, "example_gifs")
    images_root = os.path.join(gifs_dir, "imagesforgif")

    # Args: [session_dir] [--speed X]
    session_dir: str = ""
    speed: float = 3.0
    args = sys.argv[1:]
    # crude parsing to avoid adding argparse dependency
    if args:
        # if first arg doesn't start with '-', treat as session_dir
        if not args[0].startswith("-"):
            session_dir = args[0]
            args = args[1:]
        # parse flags
        i = 0
        while i < len(args):
            if args[i] in ("--speed", "-s") and i + 1 < len(args):
                try:
                    speed = float(args[i + 1])
                    if speed <= 0:
                        raise ValueError
                except ValueError:
                    print("Invalid speed; must be a number > 0")
                    sys.exit(2)
                i += 2
            else:
                print(f"Unknown argument: {args[i]}")
                print("Usage: python make_gif.py [session_dir] [--speed X]")
                sys.exit(2)
    else:
        session_dir = find_latest_session(images_root) or ""

    if not session_dir or not os.path.isdir(session_dir):
        print("No session directory found with frames.")
        sys.exit(1)

    frames, timestamps = load_frames(session_dir)
    if not frames:
        print("No frames found in session directory.")
        sys.exit(1)

    durations = compute_durations_ms(timestamps, speed=speed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(gifs_dir, f"recording_{timestamp}.gif")

    first, rest = frames[0], frames[1:]
    first.save(
        out_path,
        save_all=True,
        append_images=rest,
        duration=durations if len(durations) == len(frames) else 33,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"Saved GIF to {out_path} (speed x{speed})")


if __name__ == "__main__":
    main()
