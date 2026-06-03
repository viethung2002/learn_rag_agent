#!/usr/bin/env python3
"""Sync NotebookLM from latest GitHub (push local branch first, then one URL source)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from nlm.core.client import NotebookLMClient

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK_ID = "a7210c1b-5e45-4471-931a-398f9b44607f"
DEFAULT_REMOTE = "origin"
DEFAULT_BRANCH = "hai-dev"
DEFAULT_REPO_URL = "https://github.com/viethung2002/learn_rag_agent"


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def github_source_url(branch: str) -> str:
    return f"{DEFAULT_REPO_URL}/tree/{branch}"


def push_to_github(remote: str, branch: str, skip_push: bool) -> None:
    status = git("status", "--porcelain")
    if status:
        print("Uncommitted changes detected — commit them before push.")
        print(status)
        raise SystemExit(1)

    ahead = git("rev-list", "--count", f"{remote}/{branch}..HEAD")
    if ahead == "0":
        print(f"Branch {branch} is already up to date with {remote}/{branch}.")
        if skip_push:
            return
    if skip_push:
        print(f"Warning: {ahead} commit(s) not pushed; GitHub may be stale.")
        return

    print(f"Pushing {branch} to {remote}...")
    subprocess.check_call(["git", "-C", str(ROOT), "push", remote, branch])
    print("Push done.")


def list_sources(client: NotebookLMClient, notebook_id: str) -> list[dict]:
    sources = client.list_sources(notebook_id)
    return sources if isinstance(sources, list) else []


def sync_notebook(
    notebook_id: str,
    branch: str,
    replace_all: bool,
    dry_run: bool,
) -> None:
    url = github_source_url(branch)
    if dry_run:
        print(f"Would set sole source: {url}")
        return

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    with NotebookLMClient() as client:
        existing = list_sources(client, notebook_id)
        for src in existing:
            sid = src.get("id") or src.get("source_id")
            title = src.get("title", sid)
            if not sid:
                continue
            if replace_all:
                print(f"Deleting: {title}")
                client.delete_source(sid)
            else:
                src_url = (src.get("url") or "").rstrip("/")
                if src_url.rstrip("/") == url.rstrip("/"):
                    print("GitHub source already present; nothing to do.")
                    return

        print(f"Adding GitHub source: {url}")
        client.add_source_url(notebook_id, url)
        print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push branch to GitHub and sync NotebookLM with one GitHub URL source.",
    )
    parser.add_argument("--notebook-id", default=DEFAULT_NOTEBOOK_ID)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--branch", default=None, help=f"Default: current branch or {DEFAULT_BRANCH}")
    parser.add_argument("--skip-push", action="store_true", help="Only sync NotebookLM, do not git push")
    parser.add_argument("--no-replace", action="store_true", help="Do not delete existing sources")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    branch = args.branch or git("branch", "--show-current") or DEFAULT_BRANCH
    push_to_github(args.remote, branch, args.skip_push)
    sync_notebook(
        args.notebook_id,
        branch,
        replace_all=not args.no_replace,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
