#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# vast.sh — the single entry point for driving the remote GPU box.
#
# Read docs/RUNBOOK.md first. The short version: this repo runs on a Vast.ai
# A100, not on the laptop. Code is edited locally, pushed with `sync`, and run
# with `run`. Artifacts (data/, reports/, checkpoints/) are gitignored and live
# only on the box until you `pull` them.
#
#   ./scripts/vast.sh status              # GPU, running jobs, disk, git state
#   ./scripts/vast.sh sync                # push src/ configs/ scripts/ to the box
#   ./scripts/vast.sh run sft_ar.py       # nohup + log; survives disconnect
#   ./scripts/vast.sh run rl_grpo.py --steps 300
#   ./scripts/vast.sh fg roundtrip_eval.py --swap --limit 500   # foreground
#   ./scripts/vast.sh watch rl_grpo        # tail that job's log
#   ./scripts/vast.sh logs                 # list logs, newest first
#   ./scripts/vast.sh pull                 # copy reports/*.json back to local
#   ./scripts/vast.sh stop rl_grpo         # kill a running job
#   ./scripts/vast.sh shell                # interactive ssh
#
# Two facts this wrapper exists to hide, both of which have already cost time:
#   1. The box prints a harmless port-forward warning on EVERY ssh connection.
#      It is filtered here rather than silenced, so real errors still surface.
#   2. There is NO bare `python` on the box — only the project venv. Every
#      invocation goes through $PY explicitly.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REMOTE=vast
REPO=/root/nla-fraud
PY=$REPO/.venv/bin/python
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NOISE='bind \[127.0.0.1\]|channel_setup_fwd_listener_tcpip|Could not request local forwarding|Welcome to vast|Have fun|AI agents: READ'

ssh_q() { ssh -o ConnectTimeout=15 "$REMOTE" "$@" 2>&1 | grep -Ev "$NOISE" || true; }

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit "${1:-0}"; }

cmd="${1:-status}"; shift || true

case "$cmd" in
  sync)
    # tar over ssh: no rsync dependency on either end, and honours excludes.
    echo ">> pushing src/ configs/ scripts/ -> $REMOTE:$REPO"
    # Capture rather than pipe-to-grep so a genuine tar/ssh failure still
    # aborts; the banner noise is filtered from the captured output instead.
    if ! out=$(tar czf - -C "$LOCAL_REPO" --exclude='__pycache__' --exclude='*.pyc' \
                   src configs scripts \
                 | ssh "$REMOTE" "tar xzf - -C $REPO" 2>&1); then
      echo "$out" | grep -Ev "$NOISE" >&2 || true
      echo ">> SYNC FAILED" >&2; exit 1
    fi
    echo "$out" | grep -Ev "$NOISE" || true
    echo ">> done"
    ;;

  pull)
    echo ">> pulling reports/*.json + *.jsonl -> local"
    mkdir -p "$LOCAL_REPO/reports"
    scp -q "$REMOTE:$REPO/reports/*.json" "$LOCAL_REPO/reports/" 2>/dev/null || true
    scp -q "$REMOTE:$REPO/reports/*.jsonl" "$LOCAL_REPO/reports/" 2>/dev/null || true
    ls -la "$LOCAL_REPO/reports/" | tail -n +2
    ;;

  status)
    ssh_q "
      echo '=== GPU ==='
      nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
      echo '=== running jobs ==='
      ps aux | grep '[s]rc/.*\.py' | awk '{printf \"  pid %s  cpu %s%%  %s\n\", \$2, \$3, \$12\" \"\$13\" \"\$14}' || echo '  (none)'
      echo '=== disk ==='
      df -h / | tail -1
      echo '=== git ==='
      cd $REPO && git log --oneline -3 && echo '--- uncommitted ---' && (git status --short || true)
    "
    ;;

  run)
    [ $# -gt 0 ] || usage 1
    script="$1"; shift
    name="$(basename "$script" .py)"
    log="$REPO/logs/$name.log"
    echo ">> launching $script $* (nohup)"
    ssh_q "cd $REPO && mkdir -p logs reports && \
           nohup $PY src/$script $* > $log 2>&1 & echo \"   pid \$! -> $log\""
    echo ">> follow with: $0 watch $name"
    ;;

  fg)
    [ $# -gt 0 ] || usage 1
    script="$1"; shift
    ssh_q "cd $REPO && $PY src/$script $*"
    ;;

  watch)
    name="${1:-}"
    if [ -z "$name" ]; then
      name="$(ssh_q "ls -t $REPO/logs/*.log 2>/dev/null | head -1 | xargs -r basename" | tr -d '\r')"
      name="${name%.log}"
      echo ">> newest log: $name"
    fi
    ssh_q "tail -f $REPO/logs/$name.log"
    ;;

  logs)
    ssh_q "ls -lat $REPO/logs/*.log 2>/dev/null || echo '(no logs yet)'"
    ;;

  stop)
    pat="${1:-}"
    [ -n "$pat" ] || { echo "usage: $0 stop <pattern>  (e.g. rl_grpo)" >&2; exit 1; }
    ssh_q "pkill -f '$pat' && echo 'stopped $pat' || echo 'nothing matching $pat'"
    ;;

  shell)
    ssh "$REMOTE"
    ;;

  -h|--help|help) usage 0 ;;
  *) echo "unknown command: $cmd" >&2; usage 1 ;;
esac
