#!/usr/bin/env bash
# T014 detached driver (TASK s5).
#
#   start_t012.sh [--run-id T014] [-- python args ...]
#     no extra args : start the ONE detached joint experiment as a nohup/setsid
#                     driver for the given run id (default T014)
#     extra args    : run `run.py --run-id <id> <args>` in the foreground with
#                     the same python/env
#
# TASK s5: the script locates the project and run.py from its OWN path, writes
# every default directory under results/<run-id>, and never uses the historical
# archive path or ENTRY=run_t012.py.
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${T014_PROJECT:-$SELF_DIR}"
RUN_ID="${T014_RUN_ID:-T014}"
PY="${T014_PY:-/home/wangzengyu/edge_paper/environments/torch/bin/python}"
LOCK="${T014_SHARED_LOCK:-/home/wangzengyu/edge_paper/experiments/v20_two_gpu_validation_20261004/.live_two_gpu.lock}"

# explicit --run-id on the command line wins over the environment default
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --run-id=*) RUN_ID="${1#*=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

ART="$PROJECT/results/$RUN_ID"
LOGDIR="$ART/logs"
ENTRY="$PROJECT/run.py"
LAUNCH="$PROJECT/launch_t012_joint.sh"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
unset CUDA_VISIBLE_DEVICES

if [ ! -d "$PROJECT" ]; then
  echo "project directory missing: $PROJECT" >&2
  exit 9
fi
if [ ! -f "$LOCK" ]; then
  echo "public two-GPU lock file missing: $LOCK" >&2
  exit 7
fi
mkdir -p "$LOGDIR" || exit 9
cd "$PROJECT" || exit 9
if [ ! -f "$ENTRY" ]; then
  echo "entry point missing: $ENTRY" >&2
  exit 8
fi
if [ ! -f "$LAUNCH" ]; then
  echo "joint launcher missing: $LAUNCH" >&2
  exit 8
fi

if [ "${#ARGS[@]}" -gt 0 ]; then
  # explicit tooling invocation (e.g. --prepare-only / --check / --freeze): the
  # caller keeps full control of TASK_REFERENCE_JSON, so nothing is forced here
  exec "$PY" -u "$ENTRY" --run-id "$RUN_ID" "${ARGS[@]}"
fi

# TASK s5.2 / s5.6: ONE explicit effective reference address shared by the CPU
# checks, the factory, the runtime and the analysis.  It is the merged effective
# copy that `run.py --run-id <id> --prepare-only` generated under
# results/<run-id>/evidence from the issued reference in results/<run-id>/inputs.
REFERENCE="${T014_REFERENCE_JSON:-$ART/evidence/REFERENCE.json}"
if [ ! -f "$REFERENCE" ]; then
  echo "effective reference missing: $REFERENCE" >&2
  echo "run --prepare-only with TASK_REFERENCE_JSON=<issued> first" >&2
  exit 5
fi
export TASK_REFERENCE_JSON="$REFERENCE"
export TASK_REFERENCE_SHA256="$(sha256sum "$REFERENCE" | awk '{print $1}')"

# the joint parent takes the exclusive public lock for the whole run: start only
# when it is free, and never while another joint driver is alive
if ! flock -n "$LOCK" -c true; then
  echo "public two-GPU lock is held by another run: $LOCK" >&2
  exit 4
fi
if pgrep -f "$LAUNCH" >/dev/null 2>&1; then
  echo "another joint driver is already running:" >&2
  pgrep -a -f "$LAUNCH" >&2
  exit 3
fi

# a stale completion marker must never be read as this run's result
rm -f "$LOGDIR/RUN_COMPLETE.json" "$LOGDIR/RUN_STATE.json"

T014_RUN_ID="$RUN_ID" T014_PROJECT="$PROJECT" T014_PY="$PY" T014_SHARED_LOCK="$LOCK" \
T014_REFERENCE_JSON="$REFERENCE" TASK_REFERENCE_JSON="$REFERENCE" \
TASK_REFERENCE_SHA256="$TASK_REFERENCE_SHA256" \
  nohup setsid bash "$LAUNCH" >"$LOGDIR/driver.out" 2>&1 </dev/null &
DRIVER_PID=$!
echo "$DRIVER_PID" > "$LOGDIR/driver.pid"
sleep 2
if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
  echo "detached driver exited within 2 s; see $LOGDIR/driver.out" >&2
  exit 6
fi
echo "driver_pid=$DRIVER_PID"
echo "driver_log=$LOGDIR/driver.out"
echo "state=$LOGDIR/RUN_STATE.json"
echo "reference=$REFERENCE"
echo "reference_sha256=$TASK_REFERENCE_SHA256"
exit 0
