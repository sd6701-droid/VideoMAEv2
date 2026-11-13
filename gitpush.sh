#!/usr/bin/env bash
# Commit all local changes and push the current branch to origin.
# Usage: bash scripts/git_push.sh "commit message"
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MSG="${1:-update $(date '+%Y-%m-%d %H:%M')}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

git add -A
if git diff --cached --quiet; then
    echo "Nothing new to commit."
else
    git status --short
    git commit -m "$MSG"
fi

# set the upstream on the first push of a new branch
if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    git push origin "$BRANCH"
else
    git push -u origin "$BRANCH"
fi
echo "Pushed $BRANCH to origin."
