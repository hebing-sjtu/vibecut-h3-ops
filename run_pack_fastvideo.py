#!/usr/bin/env python3
"""Run a MiniMax-H3 Ref2VA request pack through FastVideo instead of SGLang.

Same packs, same resume behaviour and same `runs.jsonl` as `run_pack.py`, but it
drives `fastvideo.VideoGenerator` in-process rather than POSTing to a server.
FastVideo's HTTP surface only carries a single `input_reference`, so ordered
multi-reference Ref2VA is reachable only through the Python API.

The generator is built once and reused for every shot, so the 60-GiB transformer
is loaded one time rather than once per shot.

    python3 run_pack_fastvideo.py --pack "$PACK" --out "$OUT/ref2va" --num-gpus 8

Unlike the SGLang route the request's `target` is not sent verbatim: FastVideo
takes frames, not seconds, so `duration_seconds` is converted with H3's own
`align_num_frames` (17n+5 at 24 fps) and both numbers are recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pack import condition_paths, log, preflight  # noqa: E402

# H3's DiT has 56 attention heads, and sequence parallelism shards them, so a
# GPU count that does not divide 56 fails inside the model rather than here.
VALID_SP = (1, 2, 4, 7, 8, 14, 28)


def target_geometry(target: dict) -> tuple[int, int]:
    """Canvas for a request's `target`, using H3's own aspect resolution."""
    from fastvideo.pipelines.basic.minimax_h3.packing import resolve_canvas_size

    ratio = str(target.get("aspect_ratio") or "auto")
    if ratio == "auto":
        # Ref2VA resolves `auto` to the model's 16:9 fallback rather than
        # inheriting a reference asset's geometry.
        ratio = "16:9"
    try:
        width, height = (float(part) for part in ratio.split(":"))
    except ValueError as exc:
        raise ValueError(f"cannot read aspect_ratio {ratio!r}") from exc
    return resolve_canvas_size(width, height)


def target_frames(target: dict) -> tuple[int, float]:
    """Frame count for a request's duration, and the seconds it really buys."""
    from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames

    seconds = float(target.get("duration_seconds") or 0)
    if not 4.0 <= seconds <= 15.0:
        raise ValueError(f"duration_seconds must be between 4 and 15, got {seconds}")
    frames = align_num_frames(int(round(seconds * MINIMAX_H3_FPS)))
    return frames, frames / MINIMAX_H3_FPS


def build_references(request: dict, path: Path) -> list:
    """The request's conditions as ordered FastVideo references.

    Order is preserved because it is semantic: it has to match the one-based
    material tags in the prompt.
    """
    from fastvideo.pipelines.basic.minimax_h3 import MiniMaxH3Reference

    references = []
    for index, cond in enumerate(request.get("conditions") or []):
        media_type = str(cond.get("type") or "image")
        role = str(cond.get("role") or "")
        if role and role != "reference":
            log(f"  {path.stem} [{index}]: role={role!r} on a ref2va condition; sending as a reference")
        uri = str(cond.get("uri", ""))
        source = uri[len("file://"):] if uri.startswith("file://") else uri
        references.append(MiniMaxH3Reference(source=source, media_type=media_type))
    return references


def build_generator(num_gpus: int, model_path: str):
    from fastvideo import VideoGenerator
    from fastvideo.api import (ComponentConfig, EngineConfig, GeneratorConfig, OffloadConfig, ParallelismConfig,
                               PipelineSelection)

    log(f"loading {model_path} on {num_gpus} GPU(s); this is minutes, not seconds")
    return VideoGenerator.from_config(
        GeneratorConfig(
            model_path=model_path,
            engine=EngineConfig(
                num_gpus=num_gpus,
                use_fsdp_inference=num_gpus > 1,
                parallelism=ParallelismConfig(tp_size=1, sp_size=num_gpus),
                offload=OffloadConfig(
                    dit=False,
                    dit_layerwise=False,
                    text_encoder=True,
                    vae=True,
                    pin_cpu_memory=False,
                ),
            ),
            pipeline=PipelineSelection(
                workload_type="i2v",
                components=ComponentConfig(override_pipeline_cls_name="MiniMaxH3Ref2VAModularPipeline"),
            ),
        ))


def run_shot(generator, path: Path, request: dict, mp4: Path, args: argparse.Namespace) -> dict:
    from fastvideo.api import GenerationRequest, InputConfig, OutputConfig, SamplingConfig

    target = request.get("target") or {}
    height, width = target_geometry(target)
    frames, seconds = target_frames(target)
    short_edge = target.get("short_edge")
    if short_edge and int(short_edge) != min(height, width):
        log(f"  {path.stem}: request asks short_edge={short_edge}, canvas resolves to {width}x{height}")
    seed = args.seed if args.seed is not None else int(request.get("seed") or 0)
    references = build_references(request, path)

    started = time.time()
    log(f"{path.stem}: {len(references)} ref(s), {width}x{height}, "
        f"{frames} frames ({seconds:.3f}s for a {target.get('duration_seconds')}s request), seed={seed}")
    result = generator.generate(
        GenerationRequest(
            prompt=request["prompt"],
            negative_prompt="",
            inputs=InputConfig(references=references),
            sampling=SamplingConfig(
                height=height,
                width=width,
                num_frames=frames,
                fps=24,
                num_inference_steps=args.steps,
                guidance_scale=1.0,
                batch_cfg=False,
                seed=seed,
            ),
            output=OutputConfig(output_path=str(mp4), save_video=True, return_frames=False),
        ))
    elapsed = time.time() - started

    produced = Path(getattr(result, "video_path", None) or mp4)
    if produced != mp4 and produced.is_file():
        produced.replace(mp4)
    if not mp4.is_file():
        raise RuntimeError(f"generation reported success but {mp4} is missing")
    log(f"{path.stem}: done in {elapsed / 60:.1f} min -> {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)")

    # Same shape probe_outputs.py reads, plus what only this route knows.
    return {
        "shot": path.stem,
        "video_id": f"fastvideo:{path.stem}",
        "endpoint": f"fastvideo/{args.model_path}",
        "request_file": str(path),
        "mp4": str(mp4),
        "bytes": mp4.stat().st_size,
        "wall_seconds": round(elapsed, 1),
        "seed": seed,
        "target": target,
        "num_frames": frames,
        "aligned_seconds": round(seconds, 3),
        "num_inference_steps": args.steps,
        "canvas": f"{width}x{height}",
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None, help="Default <pack>/out/ref2va; pass a sibling of the pack")
    p.add_argument("--model-path", default="MiniMaxAI/MiniMax-H3")
    p.add_argument("--num-gpus", type=int, default=8, help=f"Also the sp degree; must be one of {VALID_SP}")
    p.add_argument("--steps", type=int, default=50, help="The schedule the checkpoint was distilled for")
    p.add_argument("--only", default="", help="Comma-separated shot prefixes, e.g. 01,02")
    p.add_argument("--seed", type=int, default=None, help="Override the pack's seed; only for re-rolling a shot")
    p.add_argument("--force", action="store_true", help="Regenerate shots whose mp4 already exists")
    p.add_argument("--check", action="store_true", help="Preflight only; does not import fastvideo or touch a GPU")
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args(argv)

    if args.num_gpus not in VALID_SP:
        log(f"--num-gpus {args.num_gpus} does not divide H3's 56 attention heads; use one of {VALID_SP}")
        return 2

    req_dir = args.pack / "requests" / "ref2va"
    paths = sorted(req_dir.glob("*.json"))
    if not paths:
        log(f"no request JSON under {req_dir}")
        return 2
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        paths = [q for q in paths if q.stem.split("-")[0] in wanted or q.stem in wanted]
    if not paths:
        log("no requests selected")
        return 2

    shots = []
    for path in paths:
        try:
            shots.append((path, json.loads(path.read_text())))
        except json.JSONDecodeError as exc:
            log(f"{path.name}: unreadable JSON: {exc}")
            return 2

    problems = preflight(shots, "ref2va")
    for problem in problems:
        log(f"PREFLIGHT {problem}")
    if args.check:
        for path, request in shots:
            target = request.get("target") or {}
            log(f"  {path.stem}: refs={len(request.get('conditions') or [])} seed={request.get('seed')} "
                f"{target.get('short_edge')}px {target.get('aspect_ratio')} {target.get('duration_seconds')}s "
                f"({len(condition_paths(request))} local file(s))")
        log(f"{len(shots)} request(s) checked, {len(problems)} problem(s)")
        return 1 if problems else 0
    if problems and not args.skip_preflight:
        log("refusing to load the model for a pack that does not validate; "
            "references the process cannot read make Ref2VA silently unconstrained")
        return 2

    out_dir = args.out or (args.pack / "out" / "ref2va")
    out_dir.mkdir(parents=True, exist_ok=True)

    pending, skipped = [], []
    for path, request in shots:
        mp4 = out_dir / f"{path.stem}.mp4"
        (skipped if mp4.exists() and not args.force else pending).append((path, request, mp4))
    if skipped:
        log(f"already present, skipping (use --force to redo): {', '.join(p.stem for p, _, _ in skipped)}")
    if not pending:
        log("nothing to do")
        return 0

    log(f"{len(pending)} shot(s) -> {out_dir}")
    generator = build_generator(args.num_gpus, args.model_path)
    results, failures = [], []
    try:
        for path, request, mp4 in pending:
            try:
                results.append(run_shot(generator, path, request, mp4, args))
            except Exception as exc:  # one bad shot must not cost the batch
                log(f"{path.stem}: FAILED: {type(exc).__name__}: {exc}")
                failures.append((path.stem, f"{type(exc).__name__}: {exc}"))
    finally:
        generator.shutdown()

    ledger = out_dir / "runs.jsonl"
    with ledger.open("a") as fh:
        for record in results:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    log(f"{len(results)} succeeded, {len(failures)} failed; ledger appended to {ledger}")
    for shot, why in failures:
        log(f"  failed {shot}: {why}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
