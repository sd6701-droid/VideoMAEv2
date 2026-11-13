#!/usr/bin/env bash
# Pull the latest code for a branch from origin (e.g. on the cluster).
# Local uncommitted edits are stashed, then re-applied after the pull.
# Usage: bash scripts/git_pull.sh [branch]   (default: current branch)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"

git fetch origin
STASHED=0
if ! git diff --quiet || ! git diff --cached --quiet; then
    git stash push -m "git_pull.sh auto-stash $(date '+%F %T')"
    STASHED=1
fi

if [ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]; then
    git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" --track "origin/$BRANCH"
fi
git pull --rebase origin "$BRANCH"

if [ "$STASHED" -eq 1 ]; then
    git stash pop || echo "WARNING: re-applying your local edits hit conflicts; resolve them, then run 'git stash drop'."
fi
git log --oneline -1
