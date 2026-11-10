#!/usr/bin/env bash
# T011 launcher (T011 TASK.md section 10).
#
#   ./start_t011.sh                      -> detached joint pair driver
#                                           (preflight -> joint 250 s -> verify -> summary)
#   ./start_t011.sh --prepare-only       -> foreground run_t011.py step
#   ./start_t011.sh --check --out evidence/cpu_checks.json
#   ./start_t011.sh --freeze
#   ./start_t011.sh --preflight
#   ./start_t011.sh --verify
#   ./start_t011.sh --summarize
#
# IMPORTANT: this launcher never exports CUDA_VISIBLE_DEVICES.  The joint start
# parent must not initialise a CUDA context; each child GPU arm sets its own
# mask before it creates its own actors.  CPU libraries stay single threaded.
set -uo pipefail
N11=/home/wangzengyu/edge_paper/experiments/T011_demand_feedback_20261007
PY=/home/wangzengyu/edge_paper/environments/torch/bin/python
LOCK=/home/wangzengyu/edge_paper/experiments/v20_two_gpu_validation_20261004/.live_two_gpu.lock
LOGDIR=$N11/logs
if [ ! -e "$LOCK" ]; then echo "T011_LAUNCHER missing lock $LOCK"; exit 7; fi
mkdir -p "$LOGDIR"
cd "$N11" || exit 9
echo "T011_LAUNCHER cwd=$(pwd) entry=$N11/run_t011.py argv=$*"
echo "T011_LAUNCHER user=$(id -un) host=$(hostname) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
test -f "$N11/run_t011.py" || { echo "T011_LAUNCHER missing run_t011.py"; exit 8; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES

if [ "$#" -gt 0 ]; then
  exec "$PY" -u "$N11/run_t011.py" "$@"
fi

# No arguments: start the joint pair driver detached (nohup/setsid, stdin
# /dev/null, stdout/stderr to N11/logs, driver PID echoed).  The driver itself
# only CHECKS the shared lock; the joint start parent takes it exclusively.
if ! flock -n "$LOCK" -c true; then
  echo "T011_LAUNCHER shared lock busy: $LOCK"
  exit 4
fi
if pgrep -f 'run_t0[0-9][0-9]\.py|run_v2[0-9]\.py|run_t00[0-9]\.py' >/dev/null; then
  echo "T011_LAUNCHER another runner is active"
  exit 3
fi
rm -f "$LOGDIR/RUN_COMPLETE.json"
nohup setsid bash "$N11/launch_t011_joint.sh" >"$LOGDIR/driver.out" 2>&1 </dev/null &
DRIVER_PID=$!
sleep 2
echo "T011_LAUNCHER driver_pid=$DRIVER_PID out=$LOGDIR/driver.out"
echo "T011_LAUNCHER follow: tail -f $LOGDIR/driver.out"
