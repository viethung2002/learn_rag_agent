#!/usr/bin/env python3
"""Bundle project source (no vendored libs) and upload to NotebookLM via notebooklm-cli."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from nlm.core.client import NotebookLMClient

ROOT = Path(__file__).resolve().parents[1]

# Paths / names to skip (libraries, caches, secrets, binaries)
SKIP_DIR_NAMES = {
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "htmlcov",
    "dist",
    "build",
    "wheels",
    "eggs",
    ".eggs",
    "lib",
    "lib64",
    "opensearch_data",
    "ollama_data",
    "data",
    "airflow/logs",
    "airflow/plugins",
}

SKIP_FILE_NAMES = {
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
}

SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".exe",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".webp",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".whl",
    ".egg",
    ".db",
    ".sqlite3",
}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".sql",
    ".ipynb",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".env.example",
    ".env.test",
    ".dockerignore",
    ".gitignore",
    "Dockerfile",
    "Makefile",
    "compose.yml",
    "pyproject.toml",
    "README.md",
    "LICENSE",
}

MAX_CHUNK_CHARS = 350_000


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    parts = rel.split("/")
    for i in range(len(parts)):
        segment = "/".join(parts[: i + 1])
        if segment in SKIP_DIR_NAMES or parts[i] in SKIP_DIR_NAMES:
            return True
    if path.name in SKIP_FILE_NAMES:
        return True
    if path.name.startswith(".env") and path.name not in {".env.example", ".env.test"}:
        return True
    suffix = path.suffix.lower()
    if suffix in SKIP_SUFFIXES:
        return True
    if path.is_file() and suffix not in TEXT_SUFFIXES and path.name not in TEXT_SUFFIXES:
        return True
    return False


def collect_files() -> list[Path]:
    files: list[Path] = []
    try:
        import subprocess as sp

        listed = sp.check_output(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            text=False,
        ).split(b"\0")
        candidates = [ROOT / p.decode("utf-8") for p in listed if p]
    except (FileNotFoundError, sp.CalledProcessError):
        candidates = [p for p in ROOT.rglob("*") if p.is_file()]

    for path in sorted(candidates):
        if not path.is_file():
            continue
        if should_skip(path):
            continue
        files.append(path)
    return files


def file_block(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""
    return f"\n\n{'=' * 72}\nFILE: {rel}\n{'=' * 72}\n\n{content}"


def build_chunks(files: list[Path]) -> list[tuple[str, str]]:
    header = (
        f"# arxiv-paper-curator source bundle\n"
        f"Root: {ROOT}\n"
        f"Files: {len(files)}\n"
    )
    chunks: list[tuple[str, str]] = []
    current_title = "arxiv-paper-curator (part 1)"
    current = header
    part = 1

    for path in files:
        block = file_block(path)
        if not block:
            continue
        if len(current) + len(block) > MAX_CHUNK_CHARS and len(current) > len(header):
            chunks.append((current_title, current))
            part += 1
            current_title = f"arxiv-paper-curator (part {part})"
            current = f"# Continued — part {part}\n\n"
        current += block

    if current.strip():
        chunks.append((current_title, current))
    return chunks


def upload_chunks(notebook_id: str, chunks: list[tuple[str, str]], dry_run: bool) -> None:
    if dry_run:
        return
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    with NotebookLMClient() as client:
        for title, text in chunks:
            print(f"Uploading {title!r} ({len(text):,} chars)...", flush=True)
            client.add_source_text(notebook_id, text, title=title)
            print("  done")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook-id",
        default="a7210c1b-5e45-4471-931a-398f9b44607f",
        help="NotebookLM notebook ID",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = collect_files()
    print(f"Collected {len(files)} files (libraries/caches excluded).")
    chunks = build_chunks(files)
    print(f"Split into {len(chunks)} text source(s).")
    upload_chunks(args.notebook_id, chunks, args.dry_run)


if __name__ == "__main__":
    main()
