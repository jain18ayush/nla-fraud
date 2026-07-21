#!/usr/bin/env bash
# Phase 5 runner — syncs rl_grpo.py to the vast box, smoke-tests it, then
# launches the real run under nohup so it survives SSH disconnect.
#
#   ./scripts/run_phase5.sh smoke     # 3 steps, tiny batch — just checks the path
#   ./scripts/run_phase5.sh run       # full run from configs/experiment.yaml
#   ./scripts/run_phase5.sh run 300   # full run, capped at 300 steps
#   ./scripts/run_phase5.sh watch     # tail the log
#   ./scripts/run_phase5.sh samples   # stream the generated explanations
#   ./scripts/run_phase5.sh pull      # copy reports/ back to this machine
#
# The vast SSH alias prints a harmless port-forward warning on every connect;
# it is filtered out below rather than silenced, so real errors still surface.

set -euo pipefail

REMOTE=vast
REPO=/root/nla-fraud
# The box has no bare `python` on PATH — only the project venv. Use it
# explicitly; `uv run` would work too but re-resolves deps on every call.
PY=$REPO/.venv/bin/python
LOG=$REPO/reports/phase5.log
LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ssh_q() { ssh -o ConnectTimeout=15 "$REMOTE" "$@" 2>&1 \
  | grep -v 'bind \[127.0.0.1\]:8080\|channel_setup_fwd_listener_tcpip\|Could not request local forwarding\|Welcome to vast\|Have fun\|AI agents: READ'; }

sync_code() {
  echo ">> syncing rl_grpo.py -> $REMOTE:$REPO/src/"
  scp -q "$LOCAL_REPO/src/rl_grpo.py" "$REMOTE:$REPO/src/rl_grpo.py"
}

case "${1:-run}" in
  smoke)
    sync_code
    echo ">> smoke test (3 steps, B=2 G=2)"
    ssh_q "cd $REPO && $PY src/rl_grpo.py --smoke --micro-batch 2"
    ;;

  run)
    sync_code
    STEPS_ARG=""
    [ -n "${2:-}" ] && STEPS_ARG="--steps $2"
    echo ">> launching full run $STEPS_ARG (nohup; log: $LOG)"
    ssh_q "cd $REPO && mkdir -p reports && \
           nohup $PY src/rl_grpo.py $STEPS_ARG > $LOG 2>&1 & \
           echo \"started pid \$!\""
    echo ">> follow with: $0 watch     (or: $0 samples)"
    ;;

  watch)
    ssh_q "tail -f $LOG"
    ;;

  samples)
    # The explanations are the point of this run — stream them as they land.
    ssh_q "tail -f $REPO/reports/phase5_samples.jsonl" \
      | python3 -c '
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("{"):
        print(line); continue
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue
    print(f"\n── step {d[\"step\"]}  reward={d[\"reward\"]:+.4f}  mse={d[\"mse_nrm\"]:.4f}")
    print(d["explanation"])
'
    ;;

  pull)
    echo ">> pulling reports/ from $REMOTE"
    scp -q "$REMOTE:$REPO/reports/phase5_rl.json" "$LOCAL_REPO/reports/" || true
    scp -q "$REMOTE:$REPO/reports/phase5_samples.jsonl" "$LOCAL_REPO/reports/" || true
    echo ">> done"
    ;;

  stop)
    ssh_q "pkill -f rl_grpo.py && echo stopped || echo 'nothing running'"
    ;;

  *)
    echo "usage: $0 {smoke|run [steps]|watch|samples|pull|stop}" >&2
    exit 1
    ;;
esac
