#!/usr/bin/env bash
# T011 joint driver (T011 TASK.md section 10).
#
#   four-GPU read-only preflight
#     -> ONE joint concurrent 250 s run (fixed_1a1f and qf4_feedback at the same
#        time, sharing one t0/barrier and one shared Local front end; each arm
#        runs its one-start warmup calibration before the barrier release)
#     -> both arms fully drained
#     -> ONE streaming verify
#     -> ONE offline summary
#     -> RUN_COMPLETE.json
#
# The joint start parent (run_t011.py --joint-run) EXCLUSIVELY holds the single
# shared two-GPU lock for the whole joint run; this driver only CHECKS that the
# lock is free before starting and never takes it.  The child arms never
# re-acquire it.  A non-zero stage exit stops the sequence; every stage uses an
# explicit rc check, never a `&&` chain that could silently skip a stage.
set -uo pipefail
N11=/home/wangzengyu/edge_paper/experiments/T011_demand_feedback_20261007
PY=/home/wangzengyu/edge_paper/environments/torch/bin/python
LOCK=/home/wangzengyu/edge_paper/experiments/v20_two_gpu_validation_20261004/.live_two_gpu.lock
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGDIR=$N11/logs
LOG=$LOGDIR/T011_joint_$STAMP.log
RESULT=$LOGDIR/T011_joint_$STAMP.json
STATE=$LOGDIR/RUN_STATE.json
COMPLETE=$LOGDIR/RUN_COMPLETE.json
STAGE_RC=$LOGDIR/T011_stage_rc.jsonl
mkdir -p "$LOGDIR" "$N11/analysis" "$N11/evidence"
exec >>"$LOG" 2>&1
utc () { date -u +%Y-%m-%dT%H:%M:%SZ; }
write_state () {
  status=$1
  body=$2
  printf '{"status":"%s",%s,"stamp":"%s","utc":"%s","driver_pid":%d,"log":"%s"}\n' \
    "$status" "$body" "$STAMP" "$(utc)" "$$" "$LOG" >"$STATE.tmp"
  mv "$STATE.tmp" "$STATE"
}
stage_rc () {
  printf '{"stage":"%s","rc":%d,"utc":"%s"}\n' "$1" "$2" "$(utc)" >>"$STAGE_RC"
}
printf '{"driver_pid":%d,"stamp":"%s","log":"%s","cwd":"%s","hostname":"%s","utc":"%s","argv":"%s"}\n' \
  "$$" "$STAMP" "$LOG" "$N11" "$(hostname)" "$(utc)" "$*" >"$LOGDIR/driver_PID.json"
echo "T011_DRIVER_START $STAMP cwd=$N11 pid=$$"
echo "T011_DRIVER_HOST host=$(hostname) nproc=$(nproc) utc=$(utc)"
nvidia-smi --query-gpu=uuid,name,utilization.gpu,memory.used --format=csv,noheader || true
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader || true
free -m
df -h / "$N11" || true
cd "$N11" || exit 9
echo "T011_DRIVER_CD $(pwd)"

JOINT_PID=""
stop_own_children () {
  if [ -n "$JOINT_PID" ]; then
    echo "T011_DRIVER_STOP_OWN_GROUP pgid=$JOINT_PID"
    kill -TERM -- "-$JOINT_PID" 2>/dev/null || true
    sleep 3
    kill -KILL -- "-$JOINT_PID" 2>/dev/null || true
  fi
}
finish () {
  status=$1
  stage=$2
  rc=$3
  extra=$4
  printf '{"status":"%s","stage":"%s","rc":%d,"stamp":"%s","utc":"%s","driver_pid":%d,"log":"%s"%s}\n' \
    "$status" "$stage" "$rc" "$STAMP" "$(utc)" "$$" "$LOG" "$extra" >"$RESULT"
  cp "$RESULT" "$COMPLETE"
  write_state "$status" "\"stage\":\"$stage\",\"rc\":$rc"
  echo "T011_DRIVER_$status stage=$stage rc=$rc utc=$(utc)"
  exit "$rc"
}

if pgrep -f 'run_t0[0-9][0-9]\.py|run_v2[0-9]\.py|run_t00[0-9]\.py' >/dev/null; then
  echo "T011_DRIVER_ABORT other_runner_running"
  finish aborted_other_runner preflight 3 ''
fi
if ! flock -n "$LOCK" -c true; then
  echo "T011_DRIVER_ABORT lock_busy $LOCK"
  finish aborted_lock_busy preflight 4 ''
fi
echo "T011_DRIVER_LOCK_FREE $LOCK (the joint parent will hold it)"

rm -f "$COMPLETE"

if [ ! -f "$N11/evidence/cpu_checks.json" ]; then
  echo "T011_DRIVER_ABORT missing evidence/cpu_checks.json (run --check first)"
  finish aborted_missing_cpu_checks preflight 5 ''
fi

echo "T011_STAGE_PREFLIGHT $(utc)"
"$PY" -u "$N11/run_t011.py" --preflight
rc=$?; stage_rc preflight "$rc"
if [ "$rc" -ne 0 ]; then finish failed preflight "$rc" ''; fi

echo "T011_STAGE_JOINT_RUN $(utc)"
setsid "$PY" -u "$N11/run_t011.py" --joint-run &
JOINT_PID=$!
wait "$JOINT_PID"
rc=$?
stage_rc joint_run "$rc"
if [ "$rc" -ne 0 ]; then
  stop_own_children
  finish failed joint_run "$rc" ''
fi
JOINT_PID=""

echo "T011_STAGE_VERIFY $(utc)"
"$PY" -u "$N11/run_t011.py" --verify
rc=$?; stage_rc verify "$rc"
if [ "$rc" -ne 0 ]; then finish failed verify "$rc" ''; fi

JOINT_DIR=$(sed -n 's/.*"folder": "\([^"]*\)".*/\1/p' "$LOGDIR/LATEST_JOINT.json" | head -n 1)
if [ -n "$JOINT_DIR" ] && [ -f "$JOINT_DIR/verification.json" ]; then
  cp "$JOINT_DIR/verification.json" "$N11/evidence/verification.json"
  echo "T011_DRIVER_VERIFICATION_COPIED $N11/evidence/verification.json"
fi

echo "T011_STAGE_SUMMARIZE $(utc)"
"$PY" -u "$N11/run_t011.py" --summarize
rc=$?; stage_rc summarize "$rc"
if [ "$rc" -ne 0 ]; then finish failed summarize "$rc" ''; fi

if [ ! -f "$N11/evidence/metrics.json" ]; then
  finish failed summarize 6 ''
fi
finish complete done 0 ",\"joint_folder\":\"$JOINT_DIR\""
