#!/usr/bin/env python3
"""
bundle_repo.py — produce a single-file text bundle of your repo.

Usage:
  python src/ngimager/tools/bundle_repo.py . -o repo_bundle.txt

What it does:
  - Writes a directory tree.
  - For each text file, writes a header with path/size/sha256 and the content.
  - Skips binaries and large files (configurable).
  - Skips typical junk dirs (.git, __pycache__, build artifacts).
"""

from __future__ import annotations
import argparse
import fnmatch
import hashlib
import json
import os
import sys
from pathlib import Path

DEFAULT_INCLUDE_EXT = {
    ".py", ".toml", ".md", ".rst", ".txt",
    ".yml", ".yaml", ".ini", ".cfg",
    ".json", ".csv", ".tsv",
    ".hpp", ".h", ".c", ".cpp",
    ".sh", ".bat", ".ps1",
}
DEFAULT_EXCLUDE_EXT = {
    ".inp", ".out"
}
DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    "build", "dist", ".venv", "venv", ".idea", ".vscode",
    ".ipynb_checkpoints",
}
DEFAULT_INCLUDE_PATH = {
    "examples/imaging_datasets/PHITS_simple_ng_source/usrdef.out"
}
MAX_FILE_BYTES = 400_000   # per file cap (~400 KB)
MAX_TOTAL_BYTES = 5_000_000  # overall cap (~5 MB)

BANNER = "="*78

def is_textlike(p: Path) -> bool:
    """Heuristic to decide if a file is text-like."""
    if p.suffix.lower() in DEFAULT_INCLUDE_EXT:
        return True
    # Heuristic: small files without NUL bytes
    try:
        with open(p, "rb") as f:
            chunk = f.read(4096)
        return b"\x00" not in chunk
    except Exception:
        return False

def sha256_of_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def walk_repo(root: Path):
    """Yield all files under root, skipping DEFAULT_EXCLUDE_DIRS."""
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs in-place
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDE_DIRS]
        for fn in sorted(filenames):
            yield Path(dirpath) / fn

def build_tree(root: Path) -> str:
    """Return a simple text tree of the repo."""
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDE_DIRS]
        rel = Path(dirpath).relative_to(root)
        indent = "  " * (0 if str(rel) == "." else len(rel.parts))
        if str(rel) != ".":
            lines.append(f"{indent}{rel}/")
        for fn in sorted(filenames):
            lines.append(f"{indent}  {fn}")
    return "\n".join(lines)

def _normalize_rel_path(s: str) -> str:
    """Normalize a root-relative path string to a canonical form for matching."""
    # Make it POSIX-like and strip leading ./ if present
    s = s.replace("\\", "/")
    if s.startswith("./"):
        s = s[2:]
    return s

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bundle a repo into a single text file for ChatGPT-style tools."
    )
    ap.add_argument("repo_root", help="Path to repo root")
    ap.add_argument(
        "-o", "--out", default="repo_bundle.txt",
        help="Output file (default: repo_bundle.txt)",
    )
    ap.add_argument(
        "--max-file-bytes", type=int, default=MAX_FILE_BYTES,
        help=f"Per-file size cap in bytes (default: {MAX_FILE_BYTES})",
    )
    ap.add_argument(
        "--max-total-bytes", type=int, default=MAX_TOTAL_BYTES,
        help=f"Total bundle size cap in bytes (default: {MAX_TOTAL_BYTES})",
    )
    ap.add_argument(
        "--ext", nargs="*", default=sorted(DEFAULT_INCLUDE_EXT),
        help="File extensions to treat as text and include by default "
             "(space-separated, e.g. .py .toml .md)",
    )
    ap.add_argument(
        "--exclude-ext", nargs="*", default=sorted(DEFAULT_EXCLUDE_EXT),
        help="File extensions to exclude entirely (e.g. .dat .root .big). "
             "Dot prefix is optional.",
    )
    ap.add_argument(
        "--exclude-pattern", nargs="*", default=[],
        help="Glob patterns of root-relative paths to exclude "
             "(e.g. 'data/**/*.txt' 'sim_outputs/*').",
    )
    ap.add_argument(
        "--include-path", nargs="*", default=sorted(DEFAULT_INCLUDE_PATH),
        help="Root-relative file paths to force-include, even if excluded by "
             "extension or pattern (e.g. 'data/toy_example.txt').",
    )

    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    if not root.exists():
        print(f"Repo root not found: {root}", file=sys.stderr)
        sys.exit(1)

    # Normalize include/ext controls
    include_ext = {e if e.startswith(".") else f".{e}" for e in args.ext}
    exclude_ext = {e if e.startswith(".") else f".{e}" for e in (args.exclude_ext or [])}
    exclude_patterns = list(args.exclude_pattern or [])

    # Normalize include-paths to a canonical POSIX-ish form
    include_paths = {
        _normalize_rel_path(p) for p in (args.include_path or [])
    }

    total_written = 0

    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        # Header metadata
        meta = {
            "root": str(root),
            "include_ext": sorted(include_ext),
            "exclude_ext": sorted(exclude_ext),
            "exclude_dirs": sorted(DEFAULT_EXCLUDE_DIRS),
            "exclude_patterns": exclude_patterns,
            "include_paths": sorted(include_paths),
            "max_file_bytes": args.max_file_bytes,
            "max_total_bytes": args.max_total_bytes,
        }
        out.write(BANNER + "\n")
        out.write("REPO_BUNDLE_V1\n")
        out.write(json.dumps(meta) + "\n")
        out.write(BANNER + "\n\n")

        # Directory tree
        out.write("DIRECTORY_TREE\n")
        out.write(BANNER + "\n")
        out.write(build_tree(root))
        out.write("\n\n" + BANNER + "\n\n")

        # Files
        out.write("FILES_BEGIN\n")
        out.write(BANNER + "\n")

        for p in walk_repo(root):
            rel = p.relative_to(root)
            if not p.is_file():
                continue

            rel_key = _normalize_rel_path(rel.as_posix())
            suffix = p.suffix.lower()

            # Force-include exact relative paths, overriding exclude ext/patterns
            force_include = rel_key in include_paths

            if not force_include:
                # Exclude by extension
                if suffix in exclude_ext:
                    continue

                # Exclude by glob-style pattern on relative path
                if any(fnmatch.fnmatch(rel_key, pat) for pat in exclude_patterns):
                    continue

            # Skip large/binary/unknown (unless it's a known text extension)
            if not (suffix in include_ext or is_textlike(p)):
                # Even for force_include we still require "text-like"
                continue

            try:
                b = p.read_bytes()
            except Exception:
                continue

            if len(b) > args.max_file_bytes:
                # Truncate but still include header + prefix
                body = b[: args.max_file_bytes]
                truncated = True
            else:
                body = b
                truncated = False

            sha = sha256_of_bytes(b)
            hdr = {
                "path": str(rel),
                "size": len(b),
                "sha256": sha,
                "truncated": truncated,
            }
            block_header = f"<<<FILE {json.dumps(hdr)}>>>"
            block_footer = f"<<<END_FILE {hdr['path']}>>>"

            block_bytes = len(block_header) + len(block_footer) + len(body) + 16
            if total_written + block_bytes > args.max_total_bytes:
                out.write("<<<BUNDLE_LIMIT_REACHED>>>\n")
                break

            out.write(block_header + "\n")
            try:
                text = body.decode("utf-8")
            except UnicodeDecodeError:
                # best-effort latin-1
                text = body.decode("latin-1", errors="replace")
            out.write(text + "\n")
            out.write(block_footer + "\n\n")
            total_written += block_bytes

        out.write(BANNER + "\n")
        out.write("FILES_END\n")

    print(f"Wrote {args.out} (~{total_written/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
