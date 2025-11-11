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
import argparse, hashlib, os, sys, textwrap, json
from pathlib import Path

DEFAULT_INCLUDE_EXT = {
    ".py", ".toml", ".md", ".rst", ".txt",
    ".yml", ".yaml", ".ini", ".cfg",
    ".json", ".csv", ".tsv",
    ".hpp", ".h", ".c", ".cpp",
    ".sh", ".bat", ".ps1",
}
DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    "build", "dist", ".venv", "venv", ".idea", ".vscode",
    ".ipynb_checkpoints",
}
MAX_FILE_BYTES = 400_000   # per file cap (~400 KB)
MAX_TOTAL_BYTES = 5_000_000  # overall cap (~5 MB)

BANNER = "="*78

def is_textlike(p: Path) -> bool:
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
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs in-place
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDE_DIRS]
        for fn in sorted(filenames):
            yield Path(dirpath) / fn

def build_tree(root: Path) -> str:
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_root", help="Path to repo root")
    ap.add_argument("-o", "--out", default="repo_bundle.txt", help="Output file")
    ap.add_argument("--max-file-bytes", type=int, default=MAX_FILE_BYTES)
    ap.add_argument("--max-total-bytes", type=int, default=MAX_TOTAL_BYTES)
    ap.add_argument("--ext", nargs="*", default=sorted(DEFAULT_INCLUDE_EXT),
                    help="Additional file extensions to include")
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    if not root.exists():
        print(f"Repo root not found: {root}", file=sys.stderr)
        sys.exit(1)

    include_ext = set(args.ext)
    total_written = 0

    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        # Header metadata
        meta = {
            "root": str(root),
            "include_ext": sorted(include_ext),
            "exclude_dirs": sorted(DEFAULT_EXCLUDE_DIRS),
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
            # Skip large/binary/unknown
            if not (p.suffix.lower() in include_ext or is_textlike(p)):
                continue
            try:
                b = p.read_bytes()
            except Exception:
                continue
            if len(b) > args.max_file_bytes:
                # Truncate but still include header + prefix
                prefix = b[:args.max_file_bytes]
                truncated = True
                body = prefix
            else:
                body = b
                truncated = False

            sha = sha256_of_bytes(b)
            hdr = {
                "path": str(rel).replace("\\", "/"),
                "size_bytes": len(b),
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
