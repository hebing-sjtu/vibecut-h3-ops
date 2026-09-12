#!/usr/bin/env python3
"""Point a request pack's `conditions[].uri` at the directory it actually sits in.

Only needed when the pack's own regenerator (`source/build_prompt_pack.py
--mount-root ...`) is unavailable — that script rewrites prompts and requests
together and stays the primary route. This one touches nothing but the `uri`
strings.

It does not take the old root as input. For each uri it walks the path from the
left and keeps the longest trailing segment that exists under the new root, so
the only rewrites it will make are ones that land on a file actually on disk:

    /data/vibecut-h3/refs/portrait.png   ->  <root>/refs/portrait.png

Dry run by default.

    python3 retarget_uris.py --pack /data/binghe/material/vibecut-h3-female
    python3 retarget_uris.py --pack /data/binghe/material/vibecut-h3-female --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEME = "file://"


def is_local(uri: str) -> bool:
    """A uri this script is responsible for. http(s) references are the server's
    business and are left untouched."""
    raw = uri[len(SCHEME):] if uri.startswith(SCHEME) else uri
    return raw.startswith("/")


def resolve(uri: str, root: Path) -> tuple[str, str] | None:
    """New uri and the relative path it resolved to, or None if nothing matched."""
    scheme = SCHEME if uri.startswith(SCHEME) else ""
    raw = uri[len(SCHEME):] if scheme else uri
    parts = [p for p in raw.split("/") if p]
    for i in range(len(parts)):
        rel = "/".join(parts[i:])
        if (root / rel).is_file():
            return f"{scheme}{root / rel}", rel
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pack", required=True, type=Path,
                   help="Pack root as the SGLang server will see it (the level holding shots.json)")
    p.add_argument("--apply", action="store_true", help="Write the changes; default is a dry run")
    args = p.parse_args(argv)

    root = args.pack
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    requests = sorted((root / "requests").rglob("*.json"))
    if not requests:
        print(f"no request JSON under {root / 'requests'}", file=sys.stderr)
        return 2

    changed = unresolved = already = remote = 0
    for path in requests:
        try:
            request = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"{path.relative_to(root)}: unreadable JSON: {exc}", file=sys.stderr)
            return 2
        conditions = request.get("conditions") or []
        dirty = False
        for index, cond in enumerate(conditions):
            uri = str(cond.get("uri", ""))
            if not uri:
                continue
            if not is_local(uri):
                remote += 1
                continue
            outcome = resolve(uri, root)
            if outcome is None:
                print(f"UNRESOLVED {path.relative_to(root)} [{index}]: {uri}")
                unresolved += 1
                continue
            new_uri, rel = outcome
            if new_uri == uri:
                already += 1
                continue
            print(f"{path.relative_to(root)} [{index}]: {rel}")
            cond["uri"] = new_uri
            dirty = True
            changed += 1
        if dirty and args.apply:
            # Trailing newline and indent 1 match how the pack ships its JSON
            # closely enough to keep a diff readable; the server ignores both.
            path.write_text(json.dumps(request, indent=1, ensure_ascii=False) + "\n")

    verb = "rewrote" if args.apply else "would rewrite"
    summary = (f"\n{len(requests)} request(s): {verb} {changed} uri, {already} already correct, "
               f"{unresolved} unresolved")
    if remote:
        summary += f", {remote} remote url left alone"
    print(summary)
    if unresolved:
        print("Unresolved uris point at files missing from this pack. Re-extract the pack or "
              "use source/build_prompt_pack.py; do not send these to the server.", file=sys.stderr)
        return 1
    if changed and not args.apply:
        print("Dry run. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
