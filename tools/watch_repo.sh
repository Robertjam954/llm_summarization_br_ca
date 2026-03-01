#!/bin/bash
# ============================================
#  GitHub Repo Change Watcher
#  Polls the remote for new commits and sends
#  a macOS notification when changes are found.
# ============================================
#
# Usage:
#   ./tools/watch_repo.sh              # polls every 5 minutes (default)
#   ./tools/watch_repo.sh 60           # polls every 60 seconds
#   POLL_INTERVAL=120 ./tools/watch_repo.sh
#
# To run in background:
#   nohup ./tools/watch_repo.sh &
#
# To stop:
#   kill $(cat /tmp/watch_repo.pid)

REPO_DIR="/Users/robertjames/Documents/llm_summarization_br_ca"
POLL_INTERVAL="${1:-${POLL_INTERVAL:-300}}"  # default 5 minutes
BRANCH="main"
PID_FILE="/tmp/watch_repo.pid"

echo $$ > "$PID_FILE"
echo "[watch_repo] Watching $REPO_DIR ($BRANCH) every ${POLL_INTERVAL}s"
echo "[watch_repo] PID: $$ (saved to $PID_FILE)"

cd "$REPO_DIR" || { echo "Repo dir not found"; exit 1; }

# Get initial remote HEAD
git fetch origin "$BRANCH" --quiet 2>/dev/null
LAST_SHA=$(git rev-parse "origin/$BRANCH" 2>/dev/null)
echo "[watch_repo] Current remote HEAD: ${LAST_SHA:0:8}"

while true; do
    sleep "$POLL_INTERVAL"

    # Fetch latest from remote
    git fetch origin "$BRANCH" --quiet 2>/dev/null
    CURRENT_SHA=$(git rev-parse "origin/$BRANCH" 2>/dev/null)

    if [ "$CURRENT_SHA" != "$LAST_SHA" ]; then
        # Get commit details
        NEW_COMMITS=$(git log "$LAST_SHA..$CURRENT_SHA" --oneline 2>/dev/null)
        COMMIT_COUNT=$(echo "$NEW_COMMITS" | wc -l | tr -d ' ')
        LATEST_MSG=$(git log -1 --format="%s" "origin/$BRANCH" 2>/dev/null)
        AUTHOR=$(git log -1 --format="%an" "origin/$BRANCH" 2>/dev/null)

        # macOS notification
        osascript -e "display notification \"$COMMIT_COUNT new commit(s) by $AUTHOR\n$LATEST_MSG\" with title \"llm_summarization_br_ca\" subtitle \"Remote repo updated\" sound name \"Glass\""

        echo "[watch_repo] $(date '+%H:%M:%S') — $COMMIT_COUNT new commit(s): $LATEST_MSG"
        LAST_SHA="$CURRENT_SHA"
    fi
done
