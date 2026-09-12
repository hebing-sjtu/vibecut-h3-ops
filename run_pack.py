#!/usr/bin/env python3
"""Submit the vibecut-h3 request pack to one or more SGLang MiniMax-H3 endpoints.

Stdlib only, so it runs on the GPU host without installing anything. One shot is
in flight per endpoint: the shots are independent requests against a loaded
model, and the pack asks for them one at a time so a single bad shot can be
redone without redoing the batch.

    python3 run_pack.py --pack /data/vibecut-h3 --out /data/vibecut-h3/out

Re-running skips shots whose .mp4 is already on disk, so an interrupted batch
resumes instead of restarting.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# GET /v1/videos/{id} reports a status string. The release documents `completed`;
# the others are accepted so a wrapper that spells it differently is not read as
# a hang. Anything unrecognized keeps polling rather than being called a failure.
DONE = {"completed", "succeeded", "success", "complete"}
FAILED = {"failed", "error", "errored", "cancelled", "canceled"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http(url: str, *, data: bytes | None = None, method: str = "GET", timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def http_json(url: str, *, data: bytes | None = None, method: str = "GET", timeout: int = 120) -> dict:
    return json.loads(http(url, data=data, method=method, timeout=timeout).decode("utf-8"))


def condition_paths(request: dict) -> list[str]:
    """Local filesystem paths referenced by a request's conditions.

    http(s) URIs are left out: those are the server's business, not something
    this host can check.
    """
    out = []
    for cond in request.get("conditions") or []:
        uri = str(cond.get("uri", ""))
        if uri.startswith("file://"):
            out.append(uri[len("file://"):])
        elif uri.startswith("/"):
            out.append(uri)
    return out


def preflight(shots: list[tuple[Path, dict]], expect_task: str) -> list[str]:
    """Everything checkable before a single GPU-second is spent."""
    problems = []
    for path, request in shots:
        task = request.get("task")
        if expect_task and task != expect_task:
            problems.append(f"{path.name}: task={task!r}, expected {expect_task!r}")
        if not str(request.get("prompt", "")).strip():
            problems.append(f"{path.name}: empty prompt")
        if task == "t2va" and request.get("conditions"):
            problems.append(f"{path.name}: t2va must carry an empty conditions array")
        if task == "ref2va" and not request.get("conditions"):
            problems.append(f"{path.name}: ref2va carries no conditions")
        for ref in condition_paths(request):
            if not Path(ref).exists():
                problems.append(f"{path.name}: condition uri not readable here: {ref}")
    return problems


def submit(base: str, request: dict, *, timeout: int) -> str:
    payload = json.dumps(request).encode("utf-8")
    resp = http_json(f"{base}/v1/videos", data=payload, method="POST", timeout=timeout)
    video_id = resp.get("id")
    if not video_id:
        raise RuntimeError(f"submit returned no id: {json.dumps(resp)[:400]}")
    return str(video_id)


def wait(base: str, video_id: str, *, poll: int, deadline: float) -> dict:
    """Poll until the job leaves the queue. Transient GET failures are tolerated:
    a 502 from a proxy mid-generation should not discard a job already running."""
    consecutive_errors = 0
    last = ""
    while True:
        if time.time() > deadline:
            raise TimeoutError(f"{video_id} still {last or 'unknown'} at timeout")
        try:
            state = http_json(f"{base}/v1/videos/{video_id}", timeout=60)
            consecutive_errors = 0
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            consecutive_errors += 1
            if consecutive_errors >= 10:
                raise RuntimeError(f"{video_id}: status unreadable 10x: {exc}") from exc
            time.sleep(poll)
            continue
        status = str(state.get("status", "")).lower()
        if status != last:
            log(f"  {video_id} -> {status or '(no status field)'}")
            last = status
        if status in DONE:
            return state
        if status in FAILED:
            raise RuntimeError(f"{video_id} {status}: {json.dumps(state.get('error') or state)[:400]}")
        time.sleep(poll)


def fetch(base: str, video_id: str, dest: Path, *, timeout: int) -> None:
    """Download to a temp name and rename, so an interrupted transfer is never
    mistaken for a finished shot by the resume check."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    body = http(f"{base}/v1/videos/{video_id}/content", timeout=timeout)
    if len(body) < 1024:
        raise RuntimeError(f"{video_id}: content is {len(body)} bytes, not a video")
    tmp.write_bytes(body)
    tmp.rename(dest)


def run_shot(base: str, path: Path, request: dict, out_dir: Path, args: argparse.Namespace) -> dict:
    started = time.time()
    name = path.stem
    mp4 = out_dir / f"{name}.mp4"
    log(f"{name}: submitting to {base}")
    video_id = submit(base, request, timeout=args.http_timeout)
    log(f"{name}: id={video_id}")
    state = wait(base, video_id, poll=args.poll, deadline=started + args.shot_timeout)
    fetch(base, video_id, mp4, timeout=args.http_timeout)
    elapsed = time.time() - started
    log(f"{name}: done in {elapsed / 60:.1f} min -> {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)")
    (out_dir / f"{name}.response.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))
    return {
        "shot": name,
        "video_id": video_id,
        "endpoint": base,
        "request_file": str(path),
        "mp4": str(mp4),
        "bytes": mp4.stat().st_size,
        "wall_seconds": round(elapsed, 1),
        "seed": request.get("seed"),
        "target": request.get("target"),
        "reported_duration": state.get("duration"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack", required=True, type=Path, help="Pack root as the server sees it")
    p.add_argument("--task", default="ref2va", choices=["ref2va", "t2va"],
                   help="Which request set to send (default ref2va, the pack's primary path)")
    p.add_argument("--base-url", action="append", default=[],
                   help="Endpoint, repeatable. One shot runs per endpoint at a time. "
                        "Default http://localhost:30011 for ref2va, :30010 for t2va.")
    p.add_argument("--out", type=Path, default=None, help="Where the mp4s land (default <pack>/out/<task>)")
    p.add_argument("--only", default="", help="Comma-separated shot prefixes, e.g. 01,02")
    p.add_argument("--poll", type=int, default=10, help="Status poll interval, seconds")
    p.add_argument("--shot-timeout", type=int, default=5400, help="Per-shot wall clock budget, seconds")
    p.add_argument("--http-timeout", type=int, default=600, help="Per-HTTP-call timeout, seconds")
    p.add_argument("--force", action="store_true", help="Regenerate shots whose mp4 already exists")
    p.add_argument("--seed", type=int, default=None,
                   help="Override the pack's seed. Use only to re-roll a shot that failed review.")
    p.add_argument("--check", action="store_true", help="Preflight only: validate and exit")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Send even if condition files are unreadable here (server sees a different mount)")
    args = p.parse_args(argv)

    req_dir = args.pack / "requests" / args.task
    if not req_dir.is_dir():
        log(f"no request directory at {req_dir}")
        return 2
    paths = sorted(req_dir.glob("*.json"))
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

    problems = preflight(shots, args.task)
    for problem in problems:
        log(f"PREFLIGHT {problem}")
    if problems and not (args.skip_preflight or args.check):
        log("refusing to spend GPU time on a pack that does not validate; "
            "fix the mount (see build_prompt_pack.py --mount-root) or pass --skip-preflight")
        return 2
    if args.check:
        log(f"{len(shots)} request(s) checked, {len(problems)} problem(s)")
        for path, request in shots:
            refs = len(request.get("conditions") or [])
            target = request.get("target") or {}
            log(f"  {path.stem}: task={request.get('task')} refs={refs} "
                f"seed={request.get('seed')} {target.get('short_edge')}px "
                f"{target.get('aspect_ratio')} {target.get('duration_seconds')}s")
        return 1 if problems else 0

    bases = args.base_url or ["http://localhost:30011" if args.task == "ref2va" else "http://localhost:30010"]
    bases = [b.rstrip("/") for b in bases]
    out_dir = args.out or (args.pack / "out" / args.task)
    out_dir.mkdir(parents=True, exist_ok=True)

    pending: queue.Queue = queue.Queue()
    skipped = []
    for path, request in shots:
        if args.seed is not None:
            request["seed"] = args.seed
        mp4 = out_dir / f"{path.stem}.mp4"
        if mp4.exists() and not args.force:
            skipped.append(path.stem)
            continue
        pending.put((path, request))
    if skipped:
        log(f"already present, skipping (use --force to redo): {', '.join(skipped)}")
    if pending.empty():
        log("nothing to do")
        return 0

    log(f"{pending.qsize()} shot(s) over {len(bases)} endpoint(s); output -> {out_dir}")
    results: list[dict] = []
    failures: list[tuple[str, str]] = []
    lock = threading.Lock()

    def worker(base: str) -> None:
        while True:
            try:
                path, request = pending.get_nowait()
            except queue.Empty:
                return
            try:
                record = run_shot(base, path, request, out_dir, args)
                with lock:
                    results.append(record)
            except Exception as exc:  # one bad shot must not cost the batch
                log(f"{path.stem}: FAILED: {exc}")
                with lock:
                    failures.append((path.stem, str(exc)))
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(b,), daemon=True) for b in bases]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ledger = out_dir / "runs.jsonl"
    with ledger.open("a") as fh:
        for record in sorted(results, key=lambda r: r["shot"]):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    log(f"{len(results)} succeeded, {len(failures)} failed; ledger appended to {ledger}")
    for shot, why in failures:
        log(f"  failed {shot}: {why}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
