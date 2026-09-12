#!/usr/bin/env python3
"""Measure the returned mp4s so RETURN-METADATA.csv carries facts, not guesses.

Reads whatever run_pack.py recorded in runs.jsonl (seed, target, video id, wall
clock) and joins it with what ffprobe actually finds in the file, because the
requested duration and the encoded duration are not the same number: H3 aligns
to 17n+5 frames at 24 fps, so a 6 s request does not have to come back as 6.000 s.

    python3 probe_outputs.py --dir /data/vibecut-h3/out/ref2va

Writes probe.csv next to the videos and prints the table.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

FIELDS = [
    "file", "video_id", "seed", "requested_seconds", "requested_short_edge",
    "width", "height", "fps", "duration_s", "frames", "video_codec",
    "audio_codec", "audio_hz", "audio_channels", "size_mb", "wall_minutes",
]


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def describe(path: Path, meta: dict) -> dict:
    probed = ffprobe(path)
    streams = probed.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    rate = video.get("avg_frame_rate") or "0/1"
    try:
        fps = float(Fraction(rate)) if not rate.endswith("/0") else 0.0
    except (ZeroDivisionError, ValueError):
        fps = 0.0
    target = meta.get("target") or {}
    return {
        "file": path.name,
        "video_id": meta.get("video_id", ""),
        "seed": meta.get("seed", ""),
        "requested_seconds": target.get("duration_seconds", ""),
        "requested_short_edge": target.get("short_edge", ""),
        "width": video.get("width", ""),
        "height": video.get("height", ""),
        "fps": f"{fps:g}",
        "duration_s": f"{float(probed.get('format', {}).get('duration', 0)):.3f}",
        "frames": video.get("nb_frames", ""),
        "video_codec": video.get("codec_name", ""),
        "audio_codec": audio.get("codec_name", "") or "NONE",
        "audio_hz": audio.get("sample_rate", ""),
        "audio_channels": audio.get("channels", ""),
        "size_mb": f"{path.stat().st_size / 1e6:.1f}",
        "wall_minutes": f"{meta.get('wall_seconds', 0) / 60:.1f}" if meta.get("wall_seconds") else "",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, type=Path, help="Directory holding the mp4s and runs.jsonl")
    p.add_argument("--out", type=Path, default=None, help="CSV destination (default <dir>/probe.csv)")
    args = p.parse_args(argv)

    videos = sorted(args.dir.glob("*.mp4"))
    if not videos:
        print(f"no mp4 found in {args.dir}", file=sys.stderr)
        return 2

    # Last record wins, so a re-rolled shot reports the run that produced the
    # file currently on disk.
    ledger = args.dir / "runs.jsonl"
    by_shot: dict[str, dict] = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                by_shot[str(record.get("shot"))] = record

    rows = []
    for video in videos:
        try:
            rows.append(describe(video, by_shot.get(video.stem, {})))
        except FileNotFoundError:
            print("ffprobe not found; install ffmpeg or run this on the Mac after pulling the mp4s",
                  file=sys.stderr)
            return 2
        except subprocess.CalledProcessError as exc:
            print(f"{video.name}: ffprobe failed: {exc.stderr.strip()[:200]}", file=sys.stderr)

    dest = args.out or (args.dir / "probe.csv")
    with dest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    width = max(len(r["file"]) for r in rows)
    print(f"{'file'.ljust(width)}  {'WxH':>10}  {'fps':>5}  {'dur':>7}  {'frames':>6}  {'audio':>6}  {'MB':>6}")
    for r in rows:
        print(f"{r['file'].ljust(width)}  {str(r['width']) + 'x' + str(r['height']):>10}  "
              f"{r['fps']:>5}  {r['duration_s']:>7}  {str(r['frames']):>6}  {r['audio_codec']:>6}  {r['size_mb']:>6}")
    print(f"\n{len(rows)} file(s) -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
