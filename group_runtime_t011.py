"""T011 GPU group runtime: one of two concurrent disjoint two-GPU arms (qf4).

The joint parent starts two of these (one per policy) plus the single shared
Local front end (``shared_local_t011``).  Each group owns its own two CUDA
actors, its own GPU queues/quota/commands, its own qf4 controller, its own GPU
activation and its own post-Head.  Groups never share queue objects, GPU
results or cross-group Suffix release. Historical references retain their arm
accounting; configuration.resource_accounting can select the same scheduler for
both policies so a fixed 1F1A control isolates the allocation policy.

qf4 corrections over group_runtime_t010 (TASK.md sections 3-4):

  R1  real-time ordering: every loop iteration first processes actor DONE and
      expiry/release, then dispatches, then drains public source frames inside
      a bounded 0.5 ms wall slice (checked after every frame); a backlog is
      consumed in batches, never 4096-at-once to the end.  Every deadline,
      cycle, view and dispatch call takes a FRESH monotonic clock; the aligned
      100 ms tick only labels the window_id and never replaces ``now``.
  R2  frozen roles: Direct GPU_READY admission tries exactly the first public
      quality candidate and never another role on congestion; EARLY tries the
      first quality_order role (Suffix/A) and never resource-downgrades.
      Quality thresholds and need rules are unchanged from N10.
  R3  unique lifecycle: per-component OFFER/ENQUEUE/DISPATCH/wait-remove and
      per-root TERMINAL are idempotent; a Prefix DISPATCH keeps the Suffix
      reserve (pending) until the Head decides; Suffix keeps affinity; A /
      reject / Head failure RELEASE the original card and clear the reserve; a
      root expired while Prefix/Head/Backend is in flight gets deferred
      cleanup only (real GPU wall charged, cache released, no second terminal,
      no resurrected Head or back segment).
  R4  one-start warmup calibration (TASK.md section 4): after the actors load
      their models and BEFORE the shared t0, each card runs Full 8 discard +
      32 keep, Accurate 8 + 32 and Prefix->Suffix 8 + 32 pairs (same negative
      task_id / bank_index 0 per pair, unique across pairs and cards) through
      the NORMAL actor protocol; the gate floor is this card's measured P95,
      mean updates independently; the frozen N10 intrinsic table stays as
      ``source_prior_legacy``.  No physical probe process, no CUDA actor extra.
  R5  the Runtime's real cost table (mean/gate/delta with versions) feeds the
      view, the J wiring and the real select_job gate; the qf4 candidate gate
      probe builds the next-cycle profile copy (spent=0, commitments kept) and
      calls the frozen quota_forecast select_job - no second dispatch gate.

T012 closed-loop repair (docs/T012_INTERFACE.md s2/s4/s6.3):

  C1  ONE stable component_id '<root_id>:<stage>:<attempt>' is attached to
      OFFER/ENQUEUE/REJECT/DISPATCH/COMPONENT_FINISH/REMOVE and is the ONLY
      observer key; the legacy 'offer-<task>-<stage>' string and the integer
      job_id observer keys are gone.
  C2  REJECT is written together with ADMISSION_OUTCOME outcome='rejected'
      and calls observer.component_reject (the only source of R).
  C3  the four delivery endpoints (compute_end_ns / actor_finish_ns /
      parent_received_ns / terminal_available_ns) are recorded; on-time is
      late ONLY when terminal_available_ns > deadline_abs_ns; a late parent
      still writes TERMINAL and keeps the wall charge.
  C4  current admission uses finite capacity and a no-queue inference/return
      budget from the ABSOLUTE deadline. Current-allocation ETA cannot reject;
      the physical budget is checked again before GPU service is booked.
  C5  every handler / source slice / loop gap / controller sync is timed with
      handler_timed() and reports P50/P95 and a TRUE all-time max.
  C6  disk logs go through one bounded queue with a single writer; overflow
      writes LOG_BACKPRESSURE and invalidates the control evidence of the
      window instead of dropping control data silently.
"""

from resource_axis import CARD_COUNT, CARD_IDS, card_shares

import gc
import gzip
import heapq
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
from dispatch_profile import profiler
from collections import Counter, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent

NS_PER_S = 1_000_000_000
PERIOD_NS = 50_000_000
TICK_NS = 100_000_000
WINDOW_NS = 500_000_000
from resource_axis import AXIS_MIN_FAST, AXIS_MAX_FAST, bounded_grid, demand_reference
SOURCE_SLICE_NS = 500_000          # 0.5 ms public source wall slice (s3.1)
FRAME_BATCH_MAX = 32               # T012 s3.1: <=32 frames per source slice
RESULT_BATCH_MAX = 256
DECISION_SCHEMA = 'QF4R_decision_v3'
GPU_KINDS = ('FAST_FULL', 'FAST_PREFIX', 'FAST_SUFFIX', 'ACCURATE')
WARMUP_DISCARD = 8
WARMUP_KEEP = 32
LOG_QUEUE_MAX = 8192               # bounded disk-log queue (T012 C6)
LOG_FLUSH_BATCH = 32               # amortize flush/yield without large GIL bursts
LOG_WRITER_IDLE_S = 0.002          # background writer wake-up when idle
LOG_WRITER_JOIN_S = 10.0           # total shutdown deadline for the writer
LOG_VALUE_KEYS = ('tasks', 'quota', 'controller', 'components', 'input_receipts')
# T014 s3.1.1: the background writer must not starve the latency-sensitive
# loop.  Measured on the real runtime round: a 512-record batch of json.dumps +
# flush kept the writer on the GIL long enough that the main thread waited a
# whole default (5 ms) switch interval, producing real 3-11 ms GPU_READY
# handler stalls.  The writer therefore takes a small batch and then EXPLICITLY
# yields, so the hot path is never parked behind it.  Both values are recorded
# in the summary.
LOG_WRITER_YIELD_S = 0.0001
REPLAY_TICK_NS = 500_000_000       # one replay request per NEW 500 ms window
REPLAY_AGE_GATE_NS = 750_000_000   # T012 R2-D: older results are stale_result


def local_opportunity_grid(current, gates, profile):
    """Current share plus the nearest quota boundary with a new start per role.

    This cheap prefilter bounds replay to at most three candidates. The actual
    deadline replay still decides whether a new start produces a useful root.
    It does not claim an optimum over the full allocation axis.
    """
    current = round(float(current), 10)
    present = {r.get('kind') for r in (profile.get('roots') or [])}
    fast_kinds = [k for k in ('FAST_FULL', 'FAST_PREFIX', 'FAST_SUFFIX')
                  if k in present or ('FAST_PREFIX' in present and k == 'FAST_SUFFIX')]
    if not fast_kinds:
        fast_kinds = ['FAST_FULL', 'FAST_PREFIX', 'FAST_SUFFIX']
    def shares(g):
        return card_shares(g)
    initial = shares(current)
    out = {current}
    for role, direction, kinds in (('Fast', 1, fast_kinds),
                                   ('Accurate', -1, ['ACCURATE'])):
        for step in range(1, CARD_COUNT * 10 + 1):
            candidate = round(current + direction * 0.1 * step, 10)
            if candidate < AXIS_MIN_FAST or candidate > AXIS_MAX_FAST:
                break
            nxt = shares(candidate)
            opportunity = False
            for card in CARD_IDS:
                row = gates.get(card, gates.get(str(card), {}))
                for kind in kinds:
                    gate = float(row.get(kind) or 0)
                    if gate <= 0 or not math.isfinite(gate):
                        continue
                    old = initial[card] if role == 'Fast' else 1.0 - initial[card]
                    new = nxt[card] if role == 'Fast' else 1.0 - nxt[card]
                    if int(new * PERIOD_NS / gate + 1e-9) > int(old * PERIOD_NS / gate + 1e-9):
                        opportunity = True
            if opportunity:
                out.add(candidate)
                break
    return sorted(out)
PERF_SAMPLE_MAX = 1024             # bounded percentile window (max is exact)


def service_debt_snapshot(s):
    """JSON-safe card accounting with the persistent ledger identity checked."""
    d = s.get('_service_debt')
    if d is None:
        return {}
    snapshot = {k: v for k, v in d.items() if k != 'inflight'}
    snapshot['inflight_count'] = len(d['inflight'])
    snapshot['invariant_error_ns'] = {
        r: d['balance'][r] - d['origin'][r] - d['earned'][r] + d['used'][r]
           + d['idle_repaid'][r] - d['idle_repaid']['Accurate' if r == 'Fast' else 'Fast']
        for r in ('Fast', 'Accurate')}
    return snapshot


def write(path, payload):
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True))
    return path


def _jsonable(value, path='$'):
    """Recursive JSON basic-type conversion for the public summary (T014 s3.5.7).

    deque/set/Counter/Path are converted explicitly; ANY other type raises a
    TypeError that names its path instead of being hidden behind
    ``default=str``.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (Counter, dict)):
        return {str(k): _jsonable(v, '%s.%s' % (path, k))
                for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_jsonable(v, path + '{}') for v in sorted(value, key=lambda x: str(x))]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, '%s[%d]' % (path, i)) for i, v in enumerate(value)]
    if isinstance(value, deque):
        return [_jsonable(v, path + '[]') for v in value]
    raise TypeError('summary value at %s is not a JSON basic type: %r' % (path, type(value)))


def pin_cpu(cpu):
    """Pin this process to one core and report the affinity actually applied."""
    info = {'requested': int(cpu)}
    if not hasattr(os, 'sched_setaffinity'):
        info.update({'applied': False, 'reason': 'sched_setaffinity unavailable'})
        return info
    try:
        os.sched_setaffinity(0, {int(cpu)})
    except OSError as exc:
        info.update({'applied': False, 'reason': repr(exc)})
        return info
    info.update({'applied': True, 'actual': sorted(os.sched_getaffinity(0))})
    return info


def _pctl(samples, q):
    """Nearest-rank percentile over a bounded sample list (0 -> None)."""
    vals = sorted(float(v) for v in samples)
    if not vals:
        return None
    idx = int(math.ceil(q * len(vals))) - 1
    if idx < 0:
        idx = 0
    if idx >= len(vals):
        idx = len(vals) - 1
    return vals[idx]


def _ms_stats(samples, true_max_ns):
    """{p50_ms, p95_ms, max_ms, n} from a bounded sample deque plus the exact
    all-time max (never truncated by the sample cap)."""
    vals = [float(v) for v in samples]
    return {'p50_ms': None if not vals else _pctl(vals, 0.50) / 1e6,
            'p95_ms': None if not vals else _pctl(vals, 0.95) / 1e6,
            'max_ms': int(true_max_ns) / 1e6,
            'n': len(vals)}


class _HandlerTimer(object):
    """handler_timed(name) context manager (T012 s6.3): reports the REAL wall
    of the enclosed block to the runtime perf ring and to the observer."""

    __slots__ = ('rt', 'name', 't0')

    def __init__(self, rt, name):
        self.rt = rt
        self.name = str(name)
        self.t0 = 0

    def __enter__(self):
        self.t0 = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb):
        ns = time.perf_counter_ns() - self.t0
        try:
            self.rt._note_handler(self.name, ns)
        except Exception:
            pass
        return False


class ForecastUnknown(object):
    """T014 s3.1.5/s3.1.9: structured UNKNOWN prediction result.

    Never a number and never ``inf``.  ``inf`` means the prediction COMPLETED
    and found no feasible start inside the horizon - a different fact that keeps
    its own rejection path.  The elapsed wall time, the processed transition
    count and the termination reason are carried with the result.
    """

    __slots__ = ('reason', 'elapsed_ns', 'transitions', 'budget_ns')

    def __init__(self, reason=None, elapsed_ns=None, transitions=None, budget_ns=None):
        self.reason = reason
        self.elapsed_ns = elapsed_ns
        self.transitions = transitions
        self.budget_ns = budget_ns

    def as_dict(self):
        return {'unknown': True, 'reason': self.reason, 'elapsed_ns': self.elapsed_ns,
                'transitions': self.transitions, 'budget_ns': self.budget_ns}

    def __bool__(self):
        return False

    def __repr__(self):
        return 'ForecastUnknown(%r, elapsed_ns=%r, transitions=%r, budget_ns=%r)' % (
            self.reason, self.elapsed_ns, self.transitions, self.budget_ns)


def _log_io_process(folder, keys, batches, replies, cpu):
    """JSON encoding and disk I/O in an interpreter outside the scheduler GIL."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        os.sched_setaffinity(0, {int(cpu)})
        writer = _LogWriter(folder, keys)
        replies.put({'ready': True, 'pid': os.getpid(),
                     'cpu_affinity': list(os.sched_getaffinity(0))})
        while True:
            items = batches.get()
            if items is None:
                writer._flush_handles()
                for handle in writer._handles.values():
                    handle.close()
                if writer.error:
                    raise RuntimeError(writer.error)
                replies.put({'closed': True})
                return
            if any(key not in keys for key, _item in items):
                raise RuntimeError('Unexpected disk log stream')
            writer._write(items)
            if writer.error:
                raise RuntimeError(writer.error)
            replies.put({'written': len(items), 'bytes': writer.bytes,
                'write_ns': writer.write_ns, 'writes': writer.writes, 'flushes': writer.flushes})
    except BaseException as exc:
        replies.put({'error': repr(exc)})
        raise


class _LogWriter(object):
    """The ONE background disk-log writer (T014 s3.1.1).

    The run loop only appends small immutable ``(key, text)`` records to two
    bounded in-memory queues; a dedicated daemon thread owns every file handle
    and performs the actual serialization-free batch write + flush.  The main
    loop NEVER touches a file handle.

    Backpressure is explicit: when the value queue reaches ``maxlen`` the
    writer reports ``paused`` so the runtime pauses new control work (a
    SIGSTOP-like pause) instead of dropping a value row.  Only the explicitly
    low-value HOT queue may drop, and every drop is counted.
    """

    def __init__(self, folder, keys, maxlen=LOG_QUEUE_MAX, batch=LOG_FLUSH_BATCH, process_cpu=None):
        self.folder = Path(folder)
        self.keys = list(keys)
        self.maxlen = int(maxlen)
        self.batch = int(batch)
        self.value_q = deque()
        self.hot_q = deque()
        self.park_q = deque()
        self._seq = 0
        self._written_seq = 0
        self._cv = threading.Condition()
        self._stop = False
        self._handles = {}
        self._thread = None
        self.error = None
        self.control_active = False
        self.enqueued = 0
        self.enqueue_bytes = 0
        self.enqueue_ns = 0
        self.enqueue_max_ns = 0
        self.hot_enqueued = 0
        self.hot_dropped = 0
        self.value_parked = 0
        self.value_dropped = 0
        self.enqueue_errors = 0
        self.writes = 0
        self.flushes = 0
        self.bytes = 0
        self.write_ns = 0
        self.paused_events = 0
        self.pause_windows = set()
        self.started_utc_ns = time.time_ns()
        self.joined = False
        self.process_cpu = process_cpu
        self._io_process = None
        self.io_ready = None

    # -- producer side (never blocks on disk) ---------------------------
    def enqueue(self, key, item, hot=False):
        t0 = time.perf_counter_ns()
        try:
            with self._cv:
                if hot:
                    if len(self.hot_q) >= self.maxlen:
                        self.hot_dropped += 1
                        return False
                    self.hot_q.append((key, item))
                    self.hot_enqueued += 1
                    self._seq += 1
                    self.enqueued += 1
                    self.enqueue_bytes += (len(item)
                                           if isinstance(item, str) else 0)
                elif len(self.value_q) >= self.maxlen:
                    # Explicit backpressure: a VALUE row is never dropped on the
                    # floor.  It is parked (bounded) and the runtime pauses new
                    # control work until the writer catches up.  Only a park
                    # list that is itself full can evict, and that eviction is
                    # counted and reported - never silent.
                    self.value_parked += 1
                    self.paused_events += 1
                    if len(self.park_q) >= self.maxlen:
                        self.value_dropped += 1
                        return False
                    self.park_q.append((key, item))
                    self._seq += 1
                    self.enqueued += 1
                else:
                    self.value_q.append((key, item))
                    self._seq += 1
                    self.enqueued += 1
                    self.enqueue_bytes += (len(item)
                                           if isinstance(item, str) else 0)
                self._cv.notify_all()
        except Exception:  # noqa: BLE001 - the count must never break the loop
            self.enqueue_errors += 1
            return False
        finally:
            ns = time.perf_counter_ns() - t0
            self.enqueue_ns += ns
            if ns > self.enqueue_max_ns:
                self.enqueue_max_ns = ns
        return True

    @property
    def paused(self):
        return bool(self.park_q) or len(self.value_q) >= self.maxlen

    @property
    def queued(self):
        return len(self.value_q) + len(self.park_q) + len(self.hot_q)

    def kick(self):
        with self._cv:
            self._cv.notify_all()

    def _handle(self, key):
        fh = self._handles.get(key)
        if fh is None:
            fh = (gzip.open(self.folder / (key + '.jsonl.gz'), 'wt', encoding='utf8', compresslevel=1)
                  if key == 'input_receipts' else
                  open(self.folder / (key + '.jsonl'), 'w', encoding='utf8'))
            self._handles[key] = fh
        return fh

    def handle(self, key):
        """Only legal from the writer thread or after it has exited."""
        return self._handle(key)

    # -- writer thread ---------------------------------------------------
    def start(self):
        if self._thread is None:
            if self.process_cpu is not None:
                import multiprocessing as mp
                ctx = mp.get_context('spawn')
                self._io_batches, self._io_replies = ctx.Queue(maxsize=2), ctx.Queue(maxsize=2)
                self._io_process = ctx.Process(target=_log_io_process,
                    args=(str(self.folder), self.keys, self._io_batches, self._io_replies, self.process_cpu))
                self._io_process.start()
                try:
                    self.io_ready = self._io_replies.get(timeout=30)
                    if not self.io_ready.get('ready'):
                        raise RuntimeError('Log process failed to start: %r' % self.io_ready)
                except BaseException:
                    self._io_process.terminate()
                    self._io_process.join(timeout=2)
                    raise
            self._thread = threading.Thread(target=self._run, name='t014-log-writer',
                                            daemon=True)
            self._thread.start()
        return self._thread

    def _run(self):
        try:
            while True:
                with self._cv:
                    while (not self.value_q and not self.park_q and not self.hot_q
                           and not self._stop):
                        self._cv.wait(LOG_WRITER_IDLE_S)
                    if (self._stop and not self.value_q and not self.park_q
                            and not self.hot_q):
                        break
                    items = self._take()
                if items:
                    self._write(items)
                    # Explicit yield: the hot path must never be parked behind
                    # this writer for a whole GIL switch interval.
                    if LOG_WRITER_YIELD_S > 0:
                        time.sleep(LOG_WRITER_YIELD_S)
                else:
                    self._flush_handles()
        except Exception as exc:  # noqa: BLE001 - reported to the main process
            self.error = repr(exc)
            with self._cv:
                self._cv.notify_all()

    def _take(self):
        items = []
        for source in (self.value_q, self.park_q, self.hot_q):
            while source and len(items) < self.batch:
                items.append(source.popleft())
            if len(items) >= self.batch:
                break
        return items

    def _write(self, items):
        if self._io_process is not None:
            self._io_batches.put(items, timeout=LOG_WRITER_JOIN_S)
            reply = self._io_replies.get(timeout=LOG_WRITER_JOIN_S)
            if reply.get('written') != len(items) or reply.get('error'):
                raise RuntimeError('Log batch not acknowledged: %r' % reply)
            self.bytes, self.write_ns = reply['bytes'], reply['write_ns']
            self.writes, self.flushes = reply['writes'], reply['flushes']
            with self._cv:
                self._written_seq += len(items)
                self._cv.notify_all()
            return
        t0 = time.perf_counter_ns()
        by_key = {}
        for key, item in items:
            # A log record can be large.  Do not start its JSON encoding while
            # the parent is building/scoring the next allocation decision.
            # This changes writer scheduling only; no evidence is discarded.
            while self.control_active and not self._stop:
                time.sleep(0.0001)
            if not isinstance(item, str):
                try:
                    item = json.dumps(item) + '\n'
                except (TypeError, ValueError) as exc:
                    self.error = 'serialize: %r' % (exc,)
                    item = json.dumps({'error': 'unserializable',
                                       'repr': repr(item)[:400]}) + '\n'
            by_key.setdefault(key, []).append(item)
        n = 0
        for key, lines in by_key.items():
            payload = ''.join(lines)
            fh = self._handle(key)
            fh.write(payload)
            n += len(payload)
        self._flush_handles()
        self.write_ns += time.perf_counter_ns() - t0
        self.writes += 1
        self.bytes += n
        with self._cv:
            # A producer can append while this batch is being written. Only
            # acknowledge this batch, never the producer's newer sequence.
            self._written_seq += len(items)
            self._cv.notify_all()

    def _flush_handles(self):
        if self._io_process is not None:
            return
        for fh in self._handles.values():
            try:
                fh.flush()
            except Exception:
                self.error = self.error or 'flush failed'
        self.flushes += 1

    # -- consumer side ---------------------------------------------------
    def flush_sync(self, timeout=None):
        """Wait until every record enqueued so far is written+flushed."""
        with self._cv:
            target = self._seq
            deadline = None if timeout is None else time.monotonic() + float(timeout)
            self._cv.notify_all()
            while self._written_seq < target:
                if self.error is not None:
                    break
                if deadline is not None:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return False
                    self._cv.wait(left)
                else:
                    self._cv.wait(1.0)
            return self._written_seq >= target

    def close(self, timeout=LOG_WRITER_JOIN_S):
        deadline = time.monotonic() + float(timeout)
        self.flush_sync(max(0.0, deadline - time.monotonic()))
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(max(0.0, deadline - time.monotonic()))
            if self._thread.is_alive():
                self.error = self.error or 'writer close timed out'
                if self._io_process is not None and self._io_process.is_alive():
                    self._io_process.terminate()
                    self._io_process.join(timeout=2)
                return False  # never race a live writer with a second drain
        # last chance: drain whatever is left synchronously (shutdown only)
        while self.queued:
            items = self._take()
            if not items:
                break
            self._write(items)
        self._flush_handles()
        for fh in self._handles.values():
            try:
                fh.close()
            except Exception:
                pass
        if self._io_process is not None:
            try:
                self._io_batches.put(None, timeout=2)
                reply = self._io_replies.get(timeout=max(0.01, deadline-time.monotonic()))
                self._io_process.join(timeout=2)
                if not reply.get('closed') or self._io_process.exitcode != 0:
                    self.error = self.error or 'log process did not close cleanly'
            except Exception as exc:
                self.error = self.error or repr(exc)
            finally:
                if self._io_process.is_alive():
                    self._io_process.terminate()
                    self._io_process.join(timeout=2)
        self.joined = True
        return self.queued == 0 and self.error is None

    def stats(self):
        return {
            'mode': 'background_process' if self._io_process is not None else 'background_thread',
            'io_process_ready': self.io_ready,
            'alive': bool(self._thread is not None and self._thread.is_alive()),
            'joined': self.joined,
            'queued': self.queued,
            'value_queued': len(self.value_q),
            'parked': len(self.park_q),
            'max_recorded': self._seq,
            'written': self._written_seq,
            'enqueued': self.enqueued,
            'hot_enqueued': self.hot_enqueued,
            'hot_dropped': self.hot_dropped,
            'value_parked': self.value_parked,
            'value_dropped': self.value_dropped,
            'enqueue_errors': self.enqueue_errors,
            'enqueue_bytes': self.enqueue_bytes,
            'enqueue_ms_total': self.enqueue_ns / 1e6,
            'enqueue_max_ms': self.enqueue_max_ns / 1e6,
            'write_ms_total': self.write_ns / 1e6,
            'writes': self.writes,
            'flushes': self.flushes,
            'bytes': self.bytes,
            'paused_events': self.paused_events,
            'pause_windows': len(self.pause_windows),
            'error': self.error,
        }


class _LogSink(object):
    """Front end of the background writer; keeps the legacy stream API.

    ``write`` only enqueues.  ``flush`` is the SYNCHRONOUS barrier used by the
    reader helpers (``streams['events'].flush()`` then read the file) and by
    shutdown; the per-loop call uses ``limit`` and stays non-blocking.
    """

    def __init__(self, rt, key, path):
        self.rt = rt
        self.key = str(key)
        self.path = path

    def write(self, item):
        # TASK s3.1.1: the caller hands over an IMMUTABLE RECORD, never a
        # serialized string -- serialization belongs to the writer thread.
        self.rt._enqueue_log(self.key, item)

    def flush(self):
        self.rt._flush_logs(limit=None)

    def fileno(self):
        return self.rt._log_writer.handle(self.key).fileno()

    def close(self):
        pass  # the writer thread owns every handle


def resolve_reference_path(explicit=None):
    """The ONE explicit effective-reference path for this run (T014 s3.1.8)."""
    src = str(explicit or os.environ.get('TASK_REFERENCE_JSON')
              or os.environ.get('T014_REFERENCE_JSON') or '').strip()
    if src:
        return Path(src).resolve()
    return (HERE / 'evidence' / 'REFERENCE.json').resolve()


def file_sha256(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _resolve_effective_reference(path=None, sha256=None, document=None):
    """Return (path, sha256, document) for the ONE injected effective reference.

    Explicitly injected only: either a path (whose bytes are hashed and, when a
    sha was also injected, verified) or an already-read document.  Never a
    silent read-back of whatever happens to sit in HERE/evidence.
    """
    if document is not None:
        if not isinstance(document, dict):
            raise RuntimeError('injected reference document is not an object')
        return (str(path) if path else None), (str(sha256) if sha256 else None), document
    p = resolve_reference_path(path)
    if not p.exists():
        raise RuntimeError('effective reference not found: %s' % p)
    digest = file_sha256(p)
    if sha256 is not None and str(sha256) != digest:
        raise RuntimeError('effective reference sha256 mismatch for %s: injected %s got %s'
                           % (p, sha256, digest))
    with open(str(p), encoding='utf8') as fh:
        doc = json.load(fh)
    return str(p), digest, doc


ALGORITHM_CONFIG_KEYS = (
    ('probe_ms', 'probe_first_post_ms'),
    ('probe_extended_ms', 'probe_extended_post_ms'),
    ('probe_post_levels_ms', 'probe_post_levels_ms'),
    ('probe_max_extensions', 'probe_max_extensions'),
    ('probe_pre_ms', 'probe_pre_ms'),
    ('probe_pre_cohorts', 'probe_pre_cohorts'),
    ('settle_ms', 'settle_ms'),
    ('probe_post_alignment_ms', 'probe_post_alignment_ms'),
    ('window_ms', 'window_ms'),
    ('tick_ms', 'tick_ms'),
    ('period_ms', 'sub_window_ms'),
    ('cache_max_input_age_ms', 'cache_max_input_age_ms'),
    ('replay_repeats', 'replay_repeats'),
    ('transition_timeout_ms', 'transition_timeout_ms'),
    ('stall_lag_p95_ms', 'stall_receipt_p95'),
    ('min_price_opportunity_margin_ms', 'quota_safe_margin_floor_ms'),
)


def reference_configuration(reference):
    """T014 s3.1.8: the ONE effective controller configuration.

    The injected reference is authoritative.  Its ``algorithm`` block owns the
    tightened values, so those OVERRIDE the T011-era ``configuration`` block
    instead of being silently shadowed by it.  This is the single mapping the
    production runtime and the checks both call -- there is no second copy.
    """
    ref = reference if isinstance(reference, dict) else {}
    # ONE source of truth for the frozen identity: the controller module.
    from queue_feedback_controller import (ALGORITHM_REVISION, CONTROLLER_VERSION,
                                           IMPLEMENTATION_REVISION, TASK_ID)
    configuration = dict(ref.get('configuration') or {})
    configuration['controller_version'] = CONTROLLER_VERSION
    configuration['algorithm_revision'] = ALGORITHM_REVISION
    configuration['implementation_revision'] = IMPLEMENTATION_REVISION
    configuration['task_id'] = TASK_ID
    alg = dict(ref.get('algorithm') or {})
    for cfg_key, alg_key in ALGORITHM_CONFIG_KEYS:
        if alg.get(alg_key) is not None:
            configuration[cfg_key] = alg[alg_key]
    return configuration


class SharedGroupRuntime(object):
    """One two-GPU arm of the joint T011 run (qf4)."""

    def __init__(self, name, spec, policy, gpu_uuids, folder, t0_ns, deadline_ns,
                 duration_s, bank, frame_q, status_q=None, actor_mode='spawn',
                 fake_actor_factory=None, warmup_id_base=-1_000_000,
                 reference_path=None, reference_sha256=None, reference_document=None):
        self.name = str(name)
        self.spec = dict(spec)
        self.policy = str(policy)
        self.gpu_uuids = [str(u) for u in gpu_uuids]
        if len(self.gpu_uuids) != CARD_COUNT or len(set(self.gpu_uuids)) != CARD_COUNT:
            raise ValueError('GPU UUID pool does not match process card geometry')
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.t0_ns = int(t0_ns) if t0_ns is not None else None
        self.deadline_ns = int(deadline_ns)
        self.duration_s = float(duration_s)
        self.bank = bank
        self.frame_q = frame_q
        self.status_q = status_q
        self.actor_mode = str(actor_mode)
        self.fake_actor_factory = fake_actor_factory
        self.warmup_id_base = int(warmup_id_base)

        import multiprocessing as mp
        self.mp = mp
        self.ctx = mp.get_context('spawn')

        self.gpu_ipc_budget = [400_000] * CARD_COUNT
        self.decision_budget = 800_000

        # T014 s3.1.8: EXACTLY ONE explicit effective reference injected by the
        # caller (defaults to project/evidence/REFERENCE.json, sha recorded).
        # The runtime never silently reads back HERE/evidence/REFERENCE.json and
        # never accepts a document whose recorded sha does not match.
        self.reference_path, self.reference_sha256, ref = _resolve_effective_reference(
            reference_path, reference_sha256, reference_document)
        self.reference = ref
        configuration = ref.get('configuration') or {}
        # N10 frozen common intrinsic table stays the source prior, clearly
        # labelled legacy; the per-card gate floor comes from THIS run's warmup
        self.common_intrinsic_ms = {str(k): float(v)
                                    for k, v in (ref.get('common_source_intrinsic_ms') or {}).items()}
        if not self.common_intrinsic_ms:
            raise RuntimeError('REFERENCE.json has no common_source_intrinsic_ms')
        self.source_prior_legacy = dict(self.common_intrinsic_ms)
        experiment_input = ref.get('experiment_input') or {}
        self.start_ms = float(experiment_input.get('controller_start_s', 10)) * 1000.0
        self.initial_fast = float(configuration.get('initial_fast', 1.0))

        from queue_feedback_controller import (QueueFeedbackController,
                                               make_controller_inputs_qf4,
                                               control_step, CONTROL_LOG_SCHEMA,
                                               CONTROLLER_VERSION,
                                               IMPLEMENTATION_REVISION,
                                               DECISION_SCHEMA as QC_DECISION_SCHEMA)
        from shared_local_t011 import whitelist_public, OUTCOME_ARRIVAL, \
            OUTCOME_LOCAL_TERMINAL, OUTCOME_GPU_READY, OUTCOME_PREFIX_READY, \
            OUTCOME_ABSTAIN, OUTCOME_QUALITY, gpu_intrinsic_frozen
        from queue_feedback_stats import CostBook
        self.whitelist_public = whitelist_public
        self.gpu_intrinsic_frozen = gpu_intrinsic_frozen
        self.OUTCOME_ARRIVAL = OUTCOME_ARRIVAL
        self.OUTCOME_LOCAL_TERMINAL = OUTCOME_LOCAL_TERMINAL
        self.OUTCOME_GPU_READY = OUTCOME_GPU_READY
        self.OUTCOME_PREFIX_READY = OUTCOME_PREFIX_READY
        self.OUTCOME_ABSTAIN = OUTCOME_ABSTAIN
        self.OUTCOME_QUALITY = OUTCOME_QUALITY
        self.control_step = control_step
        self.CONTROL_LOG_SCHEMA = CONTROL_LOG_SCHEMA
        self.DECISION_SCHEMA = QC_DECISION_SCHEMA

        self.cost_book = CostBook(cap=200)
        # conservative pre-warmup prior (never used once warmup seeds real
        # means); tagged clearly as the frozen legacy source prior
        # T012 s3.4: cost_ms predicts the SERVICE CHARGE (mean) while the start
        # budget uses the gate; both come from the ONE authoritative CostBook.
        self.cost_ns = {g: {k: int(round(self.common_intrinsic_ms[k] * 1e6)) + self.gpu_ipc_budget[g]
                            for k in GPU_KINDS} for g in CARD_IDS}
        self.gate_ns = {g: dict(self.cost_ns[g]) for g in CARD_IDS}
        self.mean_ns = {g: dict(self.cost_ns[g]) for g in CARD_IDS}
        self.cost_floor_ns = {g: dict(self.cost_ns[g]) for g in CARD_IDS}
        self.cost_refresh_ns = 0

        self.controller = None     # built after warmup (needs measured prices)
        self.controller_version = CONTROLLER_VERSION
        self.implementation_revision = IMPLEMENTATION_REVISION

        self.tasks = {}
        self.live = set()
        self.ended = set()
        self.jobs = []
        self.head_pending = set()          # task_ids with Prefix DONE, Head undecided
        self.gpu_quality_head = bool(self.spec.get('gpu_quality_head', False))
        self.inline_fast_continuation = bool(self.spec.get('inline_fast_continuation', False))
        if self.inline_fast_continuation and not self.gpu_quality_head:
            raise ValueError('Inline Fast requires resident GPU quality prediction')
        self.async_prefix_pipeline = (bool(self.spec.get('async_prefix_pipeline', False))
                                      and not self.gpu_quality_head)
        self.fair_input_processing = bool(self.spec.get('fair_input_processing', False))
        self.pending_input_outcomes = deque()
        self.ingress_lag_samples = deque(maxlen=PERF_SAMPLE_MAX)
        self.scheduler_input_wait_samples = deque(maxlen=PERF_SAMPLE_MAX)
        self.input_admission_wait_samples = deque(maxlen=PERF_SAMPLE_MAX)
        self.ingress_lag_max_ns = 0
        self.scheduler_input_wait_max_ns = 0
        self.input_admission_wait_max_ns = 0
        if self.gpu_quality_head:
            # Head service is inside measured FAST_PREFIX GPU service.
            self.decision_budget = 0
        self.pipeline_prefix_seen = set()
        self.pipeline_early_decisions = {}
        self.pipeline_completed_heads = set()
        self.pipeline_released = set()
        self.pipeline_head_samples = deque(maxlen=200)
        self.continuation_queues = {}
        self.network_return_ns = int(self.spec.get('network_return_ns',
                                                  getattr(self, 'network_return_ns', 0)))
        # Opt-in thin-budget rescue: unset or 0 keeps the frozen dispatch order
        # byte-identical, so every replay fixture and audit keeps its behaviour.
        self.rescue_margin = self.spec.get('rescue_margin', None)
        if self.rescue_margin is None:
            self.rescue_margin = float(os.environ.get('EDGE_RESCUE_MARGIN', '0') or 0)
        self.rescue_margin = float(self.rescue_margin)
        if not 0 <= self.rescue_margin <= 10:
            raise ValueError('rescue margin must lie in [0,10]')
        self.timed_events = []
        self.timed_sequence = 0
        self.raw_input_digest = hashlib.sha256()
        self.raw_input_frames = 0
        self.raw_input_arrivals = 0
        self.raw_input_plan = None
        if self.spec.get('frozen_input_audit'):
            plan = json.loads(Path(self.spec['frozen_plan_path']).read_text())
            self.raw_input_plan = {r['task_id']: (r['bank_index'], r['origin_node'],
                r['scheduled_offset_ns'], r['scheduled_offset_ns'] + self.deadline_ns) for r in plan}
        self.prefix_card = {}              # task_id -> card of the Prefix (affinity)
        self.deferred_cleanup = set()      # ended roots with in-flight obligations
        self.ready = {'Fast': [], 'Accurate': []}
        self.gpus = [{'busy': False, 'job': None, 'pinned': [], 'pending': set(),
                      'inline_fast_continuation': self.inline_fast_continuation,
                      'target': 0.0, 'effective': 0.0, 'transition': False,
                      'spent': Counter(), 'cycle_charges': {}, 'cycle': -1,
                      'budget_finish_ns': 0, 'idle_ns': 0.0, 'busy_ns': 0.0,
                      'busy_since': None} for _ in CARD_IDS]
        from live_runtime_v20_two_gpu import targets
        self.targets = targets
        for g, share in enumerate(targets(self.initial_fast)):
            self.gpus[g]['target'] = float(share)
            self.gpus[g]['effective'] = float(share)
        accounting = configuration.get('resource_accounting')
        if isinstance(accounting, dict):
            accounting = accounting.get(self.policy)
        # Historical algorithm documents name cycle accounting legacy_50ms.
        if accounting == 'legacy_50ms':
            accounting = 'quota_cycle'
        self.resource_accounting = str(accounting or
            ('quota_cycle' if self.policy == 'fixed_1a1f' else 'service_debt'))
        # A zero share forbids this role. Only an explicit quota change can
        # enable it on a card; idle lending must not mask allocation feedback.
        self.allow_idle_endpoint_loans = False
        for s in self.gpus:
            s['allow_idle_endpoint_loans'] = self.allow_idle_endpoint_loans
        self.delivery_objective = configuration.get('delivery_objective', 'latency_timeout')
        if self.delivery_objective not in ('quality_weighted_earliness', 'correct_ontime', 'latency_timeout'):
            raise ValueError('unsupported delivery_objective: %r' % self.delivery_objective)
        if self.resource_accounting not in ('quota_cycle', 'service_debt'):
            raise ValueError('unsupported resource_accounting: %r' % self.resource_accounting)
        if self.resource_accounting == 'service_debt':
            from quota_forecast_v8 import enable_service_debt
            for s in self.gpus:
                enable_service_debt(s)

        self.source_end = False
        self.seen_arrival = set()
        self.duplicate_frames = 0
        self.receipt_lag_max_ns = 0
        self.last_contig_frame_id = 0
        self.frame_gap_seen = False
        self.frame_publish_ns = {}
        self.watermark_ns = 0
        self.common_local_completed = 0
        # T012 C1: component_id lifecycle table (root:stage:attempt -> record)
        self.component_seq = {}
        self.components = {}
        self.rejected_components = 0
        self.delivery_lag_ns = deque(maxlen=512)
        self.delivery_budget_ns = None
        self.delivery_budget_source = 'bootstrap_ipc'
        self.bootstrap_delivery_budget_ns = int(
            max(self.gpu_ipc_budget) + self.decision_budget)
        # T012 C6: bounded single-writer disk log
        self.log_q = deque()
        # T014 s3.1.1: ONE background log writer owns every file handle; the
        # main loop only enqueues bounded, immutable records.
        self._log_writer = None
        # T014 s3.1.1: the GIL switch interval is observed, never mutated - the
        # writer yields explicitly instead (see LOG_WRITER_YIELD_S).
        try:
            self.gil_switch_interval_s = sys.getswitchinterval()
        except Exception:  # noqa: BLE001 - tuning must never break a run
            self.gil_switch_interval_s = None
        self.log_total = 0
        self.log_dropped = 0
        self.log_backpressure = False
        self.log_backpressure_windows = set()
        self.log_backpressure_events = 0
        self._bp_recent = False
        self._bp_reporting = False
        self._bp_last_window = -1
        self._stream_keys = {}
        self.log_flushes = 0
        # T012 C11: candidate replay worker (non-blocking, one request/window)
        self.replay_worker = None
        self.replay_profile = None
        self.replay_window = None
        self.replay_result = {'valid': False, 'invalid_reason': 'prediction_unknown'}
        self.replay_requests = 0
        self.replay_errors = 0
        self.replay_stale_results = 0
        self.replay_worker_status = 'init'
        # T012 C5: per-path perf
        self.handler_ns = {}
        self.max_handler_ns = {}
        self.source_slice_ns = deque(maxlen=PERF_SAMPLE_MAX)
        self.controller_sync_ns = deque(maxlen=PERF_SAMPLE_MAX)
        self.worker_ns = deque(maxlen=PERF_SAMPLE_MAX)
        self.max_source_slice_ns = 0
        self.max_controller_sync_ns = 0
        self.max_worker_ns = 0
        self.apply_count = 0
        self.apply_rejected = 0
        self.config_request_count = 0
        self.seen_command_ids = set()
        self.logged_command_ids = set()
        self.last_tick_ns = -1
        self.last_tick_aligned = -1
        self.controller_rows = 0
        self.command_seq = 0
        self.streams = {}
        self.counters = Counter()
        self.perf = {'tick_wall_ns': deque(maxlen=PERF_SAMPLE_MAX),
                     'source_slice_ns': self.source_slice_ns,
                     'main_loop_gap_ns': deque(maxlen=PERF_SAMPLE_MAX),
                     'done_lag_ns': deque(maxlen=PERF_SAMPLE_MAX),
                     'receipt_lag_ns': deque(maxlen=PERF_SAMPLE_MAX),
                     'controller_sync_ns': self.controller_sync_ns,
                     'worker_ns': self.worker_ns,
                     'max_source_slice_ns': 0,
                     'max_tick_wall_ns': 0, 'max_main_loop_gap_ns': 0,
                     'max_controller_sync_ns': 0, 'max_worker_ns': 0,
                     'forecast_ns': deque(maxlen=PERF_SAMPLE_MAX),
                     'max_forecast_ns': 0,
                     'handler_ns': self.handler_ns, 'max_handler_ns': self.max_handler_ns}
        # T014 s3.1.9: the forecast wall budget is a real, measured bound.
        self.forecast_budget_ns = int(round(float(
            (ref.get('algorithm') or {}).get('forecast_wall_budget_ms', 0.25)) * 1e6))
        self.last_forecast_unknown = False
        self.last_forecast_wall_ns = None
        self.last_loop_ns = None
        # T014 s3.1.4: every loop segment is timed with a FRESH clock and no
        # smoothing (max is exact, P95 is a bounded sample window).
        self.segments = {k: deque(maxlen=PERF_SAMPLE_MAX) for k in
                         ('done_read', 'expire_dispatch', 'source_slice', 'observer',
                          'controller', 'log_enqueue')}
        self.max_segment_ns = {k: 0 for k in self.segments}
        self.log_pause_ticks = 0
        self.rejected_profiles = {}
        self.warmup_report = None
        self.job_seq = 0

    # -- lifecycle -------------------------------------------------------
    def now(self):
        """FRESH monotonic experiment clock - never a cached loop value."""
        return time.perf_counter_ns() - self.t0_ns

    def start_actors(self):
        if self.actor_mode == 'fake':
            self._start_fake_actors()
            return self.actor_inventory
        from live_runtime_v20_two_gpu import gpu_actor, decision_actor
        self.results = self.ctx.Queue()
        self.commands = {}
        self.ps = []
        if self.async_prefix_pipeline:
            # Existing 64 waiting roots plus at most one active job per GPU.
            capacity = int(self.spec.get('edge_waiting_roots', 64)) + CARD_COUNT
            self.feature_queue = self.ctx.Queue(maxsize=capacity)
            self.continuation_queues = {role: self.ctx.Queue(maxsize=capacity)
                                        for role in ('Fast', 'Accurate', 'Abstain')}
        specs = [('gpu:%d' % g, gpu_actor, (g, self.bank)) for g in CARD_IDS]
        if not self.gpu_quality_head:
            specs.append(('decision:0', decision_actor, ()))
        for name, target, arg in specs:
            q = self.feature_queue if self.async_prefix_pipeline and name == 'decision:0' else self.ctx.Queue()
            self.commands[name] = q
            extra = ()
            if self.gpu_quality_head and name.startswith('gpu:'):
                extra = (None, None, True)
            if self.async_prefix_pipeline:
                extra = ((self.feature_queue,) if name.startswith('gpu:') else
                         (self.continuation_queues, self.spec.get('edge_model_only'),
                          bool(self.spec.get('quality_need_enabled', False))))
            p = self.ctx.Process(target=target, args=(*arg, q, self.results, *extra))
            p.start()
            self.ps.append(p)
        ready = []
        for _ in self.ps:
            r = self.results.get(timeout=300)
            if r.get('type') != 'READY':
                raise RuntimeError('group %s actor failed to become ready: %r' % (self.name, r))
            ready.append(r)
        self.actor_inventory = ready
        write(self.folder / 'actor_inventory.json', ready)
        return ready

    def _start_fake_actors(self):
        """In-process actor doubles implementing the REAL actor protocol
        (SAME commands/RESULTS schema); used by the CPU checks only."""
        self.results = self.fake_actor_factory['results']()
        self.commands = {}
        self.fake_actors = {}
        self.ps = []
        for name in tuple('gpu:%d' % g for g in CARD_IDS) + ('decision:0',):
            q = self.fake_actor_factory['commands']()
            self.commands[name] = q
            actor = self.fake_actor_factory['make'](name, q, self.results)
            self.fake_actors[name] = actor
        self.actor_inventory = [{'type': 'READY', 'resource': name}
                                for name in self.commands]
        write(self.folder / 'actor_inventory.json', self.actor_inventory)

    def pump_fake_actors(self):
        """Drive the in-process doubles until no queued command remains."""
        for actor in getattr(self, 'fake_actors', {}).values():
            actor.pump()

    def open_streams(self):
        if self._log_writer is not None:
            return
        # T014 s3.1.1: ONE background writer thread owns every file handle.
        keys = ['events', 'components', 'tasks', 'quota', 'controller']
        if self.spec.get('frozen_input_audit'):
            keys.append('input_receipts')
        self._log_writer = _LogWriter(self.folder, keys,
                                     process_cpu=self.spec.get('log_process_cpu'))
        self._log_writer.start()
        for key in keys:
            self.streams[key] = _LogSink(self, key, self.folder / (key + '.jsonl'))

    def close_streams(self):
        w = self._log_writer
        self.streams = {}
        if w is None:
            self.log_writer_joined = True
            return
        ok = False
        try:
            # T014 s3.1.1: the reported stats must be the FINAL ones, so the
            # join/flush happens first and the counters are read afterwards
            # (a pre-close read would report queued rows as unwritten).
            ok = bool(w.close(LOG_WRITER_JOIN_S))
            self.log_writer_stats = w.stats()
        except Exception:  # noqa: BLE001 - shutdown must never raise
            self.counters['log_close_errors'] += 1
            try:
                self.log_writer_stats = w.stats()
            except Exception:  # noqa: BLE001
                self.log_writer_stats = {}
        self.log_writer_joined = ok
        self._log_writer = None

    # -- bounded background-writer log (T014 s3.1.1) -----------------------
    def _log_paused(self):
        w = self._log_writer
        return bool(w is not None and w.paused)

    def _enqueue_log(self, key, item, hot=None):
        """Append one immutable record to the background writer's queue.

        The caller never touches a file handle AND never serializes: a mapping
        is shallow-copied here (one small allocation) and the writer thread
        does the ``json.dumps`` + file write.  ``hot=None`` derives the class
        from the stream key: only explicitly low-value streams may drop.
        """
        w = self._log_writer
        if w is None:
            self.log_dropped += 1
            return False
        if not isinstance(item, str):
            try:
                item = dict(item)
            except (TypeError, ValueError):
                item = str(item)
        if hot is None:
            hot = str(key) not in LOG_VALUE_KEYS
        ok = w.enqueue(key, item, hot=bool(hot))
        if ok:
            self.log_total += 1
            return True
        if hot:
            self.log_dropped += 1
        self._note_backpressure()
        return ok

    def _note_backpressure(self):
        """Queue overflow -> LOG_BACKPRESSURE; control work is PAUSED (s3.1.1),
        a value row is never dropped silently.  At most ONE backpressure row per
        500 ms window is emitted."""
        self.log_backpressure = True
        self._bp_recent = True
        win = int(self.now() // WINDOW_NS)
        self.log_backpressure_windows.add(win)
        obs = getattr(self.controller, 'observer', None)
        if obs is not None:
            obs.advance_to(self.now())
            obs.mark_flags(log_backpressure=True)
        if self._bp_reporting or win == self._bp_last_window:
            return
        self._bp_last_window = win
        self._bp_reporting = True
        try:
            now = self.now()
            self.log_backpressure_events += 1
            w = self._log_writer
            row = {'event_type': 'LOG_BACKPRESSURE', 'timestamp': now,
                   'dropped': self.log_dropped, 'total': self.log_total,
                   'queued': (w.queued if w is not None else 0),
                   'value_dropped': (w.value_dropped if w is not None else 0)}
            # The evidence row must never be lost: the writer parks it (it is a
            # value row), the main loop only reports it once per window.
            if w is not None:
                w.enqueue('events', row, hot=False)
                self.log_total += 1
            else:
                self.log_backpressure_events_reported_missing = \
                    getattr(self, 'log_backpressure_events_reported_missing', 0) + 1
        except Exception:  # noqa: BLE001 - reporting must never break the loop
            pass
        finally:
            self._bp_reporting = False

    def _flush_logs(self, limit=LOG_FLUSH_BATCH):
        """Per-loop call is NON-BLOCKING: it only wakes the writer thread.

        ``limit=None`` is the synchronous barrier used by the reader helpers and
        by shutdown, so ``streams[key].flush()`` then reading the file still
        observes every emitted line.
        """
        w = self._log_writer
        if w is None:
            return 0
        if limit is None:
            drained = 0 if w.flush_sync(LOG_WRITER_JOIN_S) else 0
            self.log_flushes += 1
            self.log_writer_flushes = getattr(self, 'log_writer_flushes', 0) + 1
            return drained
        w.kick()
        self.log_flushes += 1
        return 0

    # -- timing (T012 C5) --------------------------------------------------
    def _note_segment(self, name, ns):
        d = self.segments.get(name)
        if d is None:
            d = deque(maxlen=PERF_SAMPLE_MAX)
            self.segments[name] = d
            self.max_segment_ns[name] = 0
        ns = int(ns)
        d.append(ns)
        if ns > self.max_segment_ns[name]:
            self.max_segment_ns[name] = ns

    def _note_log_pause(self):
        """Backpressure pause of new control work (T014 s3.1.1)."""
        self.log_pause_ticks += 1
        win = int(self.now() // WINDOW_NS)
        self.log_backpressure = True
        self.log_backpressure_windows.add(win)
        obs = getattr(self.controller, 'observer', None)
        if obs is not None:
            obs.advance_to(self.now())
            obs.mark_flags(log_backpressure=True)

    def _control_log_backpressure(self, now):
        """Block current/past-two observation windows, not the entire run.

        The lifetime flag remains in the summary. A value-row loss or writer
        error is unrecoverable evidence loss and still blocks all later work.
        """
        writer = self._log_writer
        if writer is not None and (writer.paused or writer.error
                                   or writer.value_dropped):
            return True
        win = int(now) // WINDOW_NS
        return any(i in self.log_backpressure_windows
                   for i in (win, win - 1, win - 2))

    def handler_timed(self, name):
        return _HandlerTimer(self, name)

    def _note_handler(self, name, ns):
        d = self.handler_ns.get(name)
        if d is None:
            d = deque(maxlen=PERF_SAMPLE_MAX)
            self.handler_ns[name] = d
            self.max_handler_ns[name] = 0
        d.append(int(ns))
        if int(ns) > self.max_handler_ns[name]:
            self.max_handler_ns[name] = int(ns)
        obs = getattr(self.controller, 'observer', None) if self.controller else None
        if obs is not None and hasattr(obs, 'note_handler'):
            try:
                obs.note_handler(name, ns)
            except Exception:
                self.counters['perf_note_errors'] += 1

    def _note_source_slice(self, ns):
        self.source_slice_ns.append(int(ns))
        if int(ns) > self.max_source_slice_ns:
            self.max_source_slice_ns = int(ns)
        obs = getattr(self.controller, 'observer', None) if self.controller else None
        if obs is not None and hasattr(obs, 'note_source_slice'):
            try:
                obs.note_source_slice(ns)
            except Exception:
                self.counters['perf_note_errors'] += 1

    def _note_loop_gap(self, ns):
        d = self.perf['main_loop_gap_ns']
        d.append(int(ns))
        if int(ns) > self.perf['max_main_loop_gap_ns']:
            self.perf['max_main_loop_gap_ns'] = int(ns)
        obs = getattr(self.controller, 'observer', None) if self.controller else None
        if obs is not None and hasattr(obs, 'note_loop_gap'):
            try:
                obs.note_loop_gap(ns)
            except Exception:
                self.counters['perf_note_errors'] += 1

    def _note_controller_sync(self, ns):
        self.controller_sync_ns.append(int(ns))
        if int(ns) > self.max_controller_sync_ns:
            self.max_controller_sync_ns = int(ns)
        obs = getattr(self.controller, 'observer', None) if self.controller else None
        if obs is not None and hasattr(obs, 'note_controller_sync'):
            try:
                obs.note_controller_sync(ns)
            except Exception:
                self.counters['perf_note_errors'] += 1

    def _note_worker(self, ns):
        self.worker_ns.append(int(ns))
        if int(ns) > self.max_worker_ns:
            self.max_worker_ns = int(ns)
        obs = getattr(self.controller, 'observer', None) if self.controller else None
        if obs is not None and hasattr(obs, 'note_worker'):
            try:
                obs.note_worker(ns)
            except Exception:
                self.counters['perf_note_errors'] += 1

    # -- component identity (T012 C1) --------------------------------------
    def _component_id(self, root, stage, attempt):
        return '%s:%s:%d' % (root, stage, int(attempt))

    def component_id_for(self, task, stage):
        """The component_id of one stage of one root ('None' stage => None)."""
        comp = (task or {}).get('component_' + stage)
        return comp

    # -- cost book (T012 s3.4 / single authoritative table) ----------------
    def _refresh_costs(self, now_ns):
        """Copy mean/gate/floor from the ONE CostBook into the runtime tables;
        the executor cost_ns can never diverge from a book update."""
        for g in CARD_IDS:
            for kind in GPU_KINDS:
                mean = self.cost_book.mean_ns(g, kind)
                gate = self.cost_book.gate_ns(g, kind)
                floor = self.cost_book.floor_ns(g, kind)
                if mean is not None:
                    self.mean_ns[g][kind] = int(round(mean))
                    self.cost_ns[g][kind] = int(round(mean))
                if gate is not None:
                    self.gate_ns[g][kind] = int(round(gate))
                if floor is not None:
                    self.cost_floor_ns[g][kind] = int(round(floor))
        self.cost_refresh_ns = int(now_ns)

    def apply_stats(self):
        return {'apply_count': int(self.apply_count),
                'apply_rejected': int(self.apply_rejected),
                'config_request_count': int(self.config_request_count)}

    # -- delivery budget (T012 C4) -----------------------------------------
    def _update_delivery_budget(self):
        """P95 of this arm's recent actor-finish -> parent-available delay; the
        frozen/pre-warm IPC is the explicit bootstrap when no sample exists."""
        now = self.now() if self.t0_ns is not None else time.perf_counter_ns()
        previous = getattr(self, '_delivery_update_at', None)
        if previous is not None and 0 <= now - previous < PERIOD_NS:
            return self.delivery_budget_ns
        self._delivery_update_at = now
        if self.delivery_lag_ns:
            p95 = _pctl(self.delivery_lag_ns, 0.95)
            if p95 is not None and p95 > 0:
                self.delivery_budget_ns = int(p95)
                self.delivery_budget_source = 'healthy_done_lag_p95'
                return self.delivery_budget_ns
        self.delivery_budget_ns = int(self.bootstrap_delivery_budget_ns)
        self.delivery_budget_source = 'bootstrap_ipc'
        return self.delivery_budget_ns

    def delivery_budget(self):
        if self.delivery_budget_ns is None:
            self._update_delivery_budget()
        return int(self.delivery_budget_ns)

    # -- public event stream ---------------------------------------------
    def emit(self, event_type, **kw):
        now = self.now()
        row = {'event_type': event_type, 'timestamp': now}
        row.update(kw)
        self.streams['events'].write(row)
        if self.controller is not None:
            self.controller.observe(event_type, self.whitelist_public(kw), now)
        return now

    def component(self, r):
        self.streams['components'].write(r)

    def finish(self, i, role, prediction=None, reason=None, terminal_finish_ns=None,
               late_discarded=False, component_id=None, compute_end_ns=None,
               actor_finish_ns=None, parent_received_ns=None,
               terminal_available_ns=None, _network_delivered=False):
        """Exactly ONE terminal per root (s3.2); duplicates are refused.

        T012 C3: the on-time verdict uses ONLY ``terminal_available_ns`` (the
        instant the terminal becomes available at the parent) against the
        ABSOLUTE deadline; a late parent still writes the terminal and keeps
        its charge, it only gets ``ontime_by_terminal=False``.
        """
        if i in self.ended:
            self.counters['duplicate_terminal_refused'] += 1
            return False
        if (self.spec.get('real_network_return') and not _network_delivered
                and self.network_return_ns and prediction is not None
                and role in ('Fast', 'Accurate')):
            task = self.tasks[i]
            if task.get('network_return_pending'):
                return False
            task['network_return_pending'] = True
            due = self.now() + self.network_return_ns
            values = dict(i=i, role=role, prediction=prediction, reason=reason,
                terminal_finish_ns=terminal_finish_ns, late_discarded=late_discarded,
                component_id=component_id, compute_end_ns=compute_end_ns,
                actor_finish_ns=actor_finish_ns, parent_received_ns=parent_received_ns)
            self._later(due, 'NETWORK_RETURN', values)
            self.emit('NETWORK_RETURN', task_id=i, sent_ns=self.now(), available_ns=due,
                      duration_ns=self.network_return_ns)
            return True
        now = self.now()
        task = self.tasks.get(i) or {}
        fin = int(terminal_finish_ns) if terminal_finish_ns is not None else now
        acting_finish = int(actor_finish_ns) if actor_finish_ns is not None else fin
        compute_end = int(compute_end_ns) if compute_end_ns is not None else acting_finish
        parent_recv = int(parent_received_ns) if parent_received_ns is not None else now
        if terminal_available_ns is None:
            terminal_available = parent_recv
        else:
            terminal_available = int(terminal_available_ns)
        deadline_abs = int(task.get('deadline_abs_ns', 0))
        # T014 s3.1.1/s3.2.5: the on-time endpoint is INCLUSIVE
        # (terminal_available <= D is on time); the positive-service admission
        # still refuses a root whose D - fresh_now <= 0.
        late = bool(terminal_available > deadline_abs)
        ontime = not late
        self.streams['tasks'].write(
            {'task_id': i, 'finish_ns': fin, 'role': role, 'prediction': prediction,
             'late': late, 'reason': reason, 'terminal_finish_ns': fin,
             'observed_ns': now, 'group': self.name,
             'deadline_abs_ns': deadline_abs,
             'terminal_available_ns': terminal_available,
             'actor_finish_ns': acting_finish,
             'compute_end_ns': compute_end,
             'parent_received_ns': parent_recv,
             'component_id': component_id,
             'p_selected': task.get('p_selected_final'),
             'ontime_by_terminal': ontime,
             'late_discarded': bool(late_discarded)})
        self.emit('TERMINAL', task_id=i, role=role, reason=reason,
                  prediction=prediction, terminal_finish_ns=fin, late=late,
                  late_discarded=bool(late_discarded),
                  compute_end_ns=compute_end, actor_finish_ns=acting_finish,
                  parent_received_ns=parent_recv,
                  terminal_available_ns=terminal_available,
                  deadline_abs_ns=deadline_abs, component_id=component_id,
                  ontime_by_terminal=ontime,
                  p_selected=task.get('p_selected_final'))
        self.live.discard(i)
        self.ended.add(i)
        self.deferred_cleanup.discard(i)
        # T014 s3.1.2: EVERY successfully registered GPU root (GPU_READY with a
        # quality-selected directF/directA, or PREFIX_READY) gets exactly ONE
        # observer.root_terminal -- including Abstain, wait-timeout, admission
        # rejection, Head failure and a late DONE.  ``gpu_root_registered`` is an
        # explicit fact; a non-empty root_id is NOT enough (the common Local and
        # common-source Abstain must never enter the GPU root overall).
        if self.controller is not None and task.get('gpu_root_registered'):
            cid = task.get('root_id')
            if cid is not None:
                p = task.get('p_selected_final')
                self.controller.observer.root_terminal(
                    cid, role, ontime, now, p_selected=p,
                    late_discarded=late_discarded,
                    terminal_available_ns=terminal_available)
        return True

    def new_job_id(self):
        self.job_seq += 1
        return self.job_seq - 1

    def push(self, i, kind, gpu=None, stage=None, component_id=None):
        task = self.tasks[i]
        prob = task.get('probabilities') or {}
        role = 'Local' if kind.startswith('LOCAL') else 'Accurate' if kind == 'ACCURATE' else 'Fast'
        psel = prob.get(role)
        if isinstance(psel, bool):
            psel = None
        if role in ('Fast', 'Accurate'):
            task['p_selected_final'] = psel
        if component_id is None:
            if stage is None:
                stage = {'FAST_PREFIX': 'PREFIX', 'FAST_SUFFIX': 'EARLY',
                         'FAST_FULL': 'GPU_CONT', 'ACCURATE': 'GPU_CONT'}.get(
                             kind, 'GPU_CONT')
            component_id = task.get('component_' + stage)
        j = {'job_id': self.new_job_id(), 'task_id': i, 'kind': kind,
             'bank_index': task['bank_index'], 'enqueue_ns': self.now(),
             'affinity': gpu if kind == 'FAST_SUFFIX' else None, 'p_selected': psel,
             'component_id': component_id,
             'deadline_ns': int(task['deadline_abs_ns'])}
        if kind == 'FAST_PREFIX' and self.inline_fast_continuation:
            j['inline_fast_continuation'] = True
        self.jobs.append(j)
        if kind == 'FAST_SUFFIX':
            self.gpus[gpu]['pinned'].append(j)
        else:
            self.ready['Accurate' if kind == 'ACCURATE' else 'Fast'].append(j)
        event_ns = self.emit('ENQUEUE', **j)
        if self.controller is not None and kind in GPU_KINDS:
            self.controller.observer.component_enqueue(
                component_id, int(task['deadline_abs_ns']), event_ns,
                affinity=j['affinity'])
        return j

    def offer(self, i, kind, stage, retry=False):
        """First component request of a stage OFFERs once per root with ONE
        stable component_id; a real retry is the ONLY way to a new attempt."""
        task = self.tasks.get(i) or {}
        root = task.get('root_id')
        if root is None:
            root = 'root-%d' % i
            task['root_id'] = root
        key = (root, str(stage))
        count = self.component_seq.get(key, 0)
        if count and not retry:
            self.counters['offer_retry_not_recounted'] += 1
            return task.get('component_' + stage) or self._component_id(root, stage, 0)
        attempt = 0 if not count else count
        comp = self._component_id(root, stage, attempt)
        self.component_seq[key] = count + 1
        task['offered_' + stage] = True
        task['component_' + stage] = comp
        self.components[comp] = {'component_id': comp, 'root_id': root, 'task_id': i,
                                 'kind': kind, 'stage': str(stage), 'attempt': attempt,
                                 'state': 'offered', 'offer_ns': self.now()}
        event_ns = self.emit('OFFER', task_id=i, kind=kind, stage=stage, root_id=root,
                  component_id=comp)
        if self.controller is not None and kind in GPU_KINDS:
            self.controller.observer.component_offer(comp, kind, root_id=root,
                                                    now_ns=event_ns)
        return comp

    def eta(self, kind, gpu=None, task_id=None, horizon_ns=None):
        """Prediction INSIDE the horizon derived from the ABSOLUTE deadline
        (TASK 3.1); a missing horizon is a defect, never a 25 ms default.

        Retained for diagnostics and historical interfaces, not task admission.

        T014 s3.1.9: the 0.25 ms forecast wall budget is REAL.  Every call is
        timed with a fresh clock; a call that overran, returned no value or a
        non-finite/partial value is marked UNKNOWN (counted, with its wall time
        kept) and is NEVER consumed as a positive/complete result - the request
        is simply not admitted, which is the conservative keep side.
        """
        if horizon_ns is None:
            raise ValueError('eta requires horizon_ns from eta_horizon_ns(D, now, budget)')
        from quota_forecast_v8 import forecast as quota_forecast, LAST_FORECAST as FORECAST_LAST
        t0 = time.perf_counter_ns()
        value = quota_forecast(self.gpus, self.ready, self.tasks, self.mean_ns,
                               self.gpu_ipc_budget, self.now(), kind, gpu, task_id,
                               decision_ns=self.decision_budget * (1 + len(self.head_pending)),
                               horizon_ns=int(horizon_ns), gates=self.gate_ns,
                               wall_budget_ns=self.forecast_budget_ns,
                               prefix_terminal=kind == 'FAST_PREFIX')
        wall_ns = time.perf_counter_ns() - t0
        self._note_forecast(wall_ns)
        record = dict(FORECAST_LAST or {})
        self.last_forecast_record = record
        self.counters['forecast_expired_waiters_removed'] += int(
            record.get('expired_waiters_removed') or 0)
        partial = value is None or (isinstance(value, float) and math.isnan(value))
        if wall_ns > self.forecast_budget_ns:
            self.counters['forecast_over_budget'] += 1
        if partial:
            self.counters['forecast_empty'] += 1
        if wall_ns > self.forecast_budget_ns or partial:
            self.counters['forecast_unknown'] += 1
            self.last_forecast_unknown = True
            # T014 s3.1.5: a structured UNKNOWN, never ``inf``.  ``inf`` means
            # the prediction COMPLETED and found no feasible start inside the
            # horizon, which is a different fact and keeps its own path.
            reason = str(record.get('terminated')
                         or ('wall_budget' if wall_ns > self.forecast_budget_ns else 'empty'))
            self.counters['forecast_unknown_' + reason] += 1
            return ForecastUnknown(reason, record.get('elapsed_ns', wall_ns),
                                   record.get('transitions'), self.forecast_budget_ns)
        self.last_forecast_unknown = False
        return value + (self.network_return_ns if self.spec.get('real_network_return') else 0)

    def _note_forecast(self, ns):
        ns = int(ns)
        self.perf['forecast_ns'].append(ns)
        if ns > self.perf['max_forecast_ns']:
            self.perf['max_forecast_ns'] = ns
        self.last_forecast_wall_ns = ns

    def _admission(self, task, kind, gpu=None, stage=None, extra_ns=0):
        """OFFER first, then finite capacity and no-queue physical feasibility.

        No ETA, current GPU share, queue-completion prediction or future quality
        outcome can reject this request. Recheck physical budget at GPU start.
        """
        from execution_guards import GPU_ADMISSION_RULE
        i = task.get('task_id')
        D = int(task['deadline_abs_ns'])
        fresh = self.now()
        if D - fresh <= 0:
            self.counters['admission_expired_no_eta'] += 1
            return {'ok': False, 'reason': 'intrinsic_budget', 'remaining_ns': 0,
                    'estimate_ns': None, 'now_ns': fresh, 'horizon_ns': None,
                    'eta_called': False, 'budget_ns': None,
                    'evidence': {'expired_before_eta': True, 'remaining_ns': 0}}
        if (self.async_prefix_pipeline or self.fair_input_processing or
                self.spec.get('finite_queue_policy', False)):
            waiting = self._edge_waiting_ids()
            capacity = int(self.spec.get('edge_waiting_roots', 64))
            if i not in waiting and len(waiting) >= capacity:
                return {'ok': False, 'reason': 'queue_capacity', 'remaining_ns': D - fresh,
                        'now_ns': fresh, 'evidence': {'waiting_roots': len(waiting),
                                                     'capacity': capacity}}
        minimum = self._zero_wait_completion_ns(kind, gpu=gpu, extra_ns=extra_ns)
        fresh2 = self.now()
        enough = math.isfinite(minimum) and minimum <= D-fresh2
        source_remaining = D-int(task.get('source_publish_ns', task.get('arrival_ns', fresh2)))
        reason = (None if enough else 'unsupported_model' if not math.isfinite(minimum)
                  else 'intrinsic_budget' if source_remaining < minimum else 'receipt_delay')
        self.counters['admission_without_queue_eta'] += 1
        return {'ok': enough, 'reason': reason, 'remaining_ns': D-fresh2,
                'estimate_ns': None, 'now_ns': fresh2, 'horizon_ns': None,
                'eta_called': False, 'budget_ns': None,
                'budget_source': GPU_ADMISSION_RULE,
                'evidence': {'admission_rule': GPU_ADMISSION_RULE,
                    'zero_wait_ns': int(minimum) if math.isfinite(minimum) else None,
                    'source_rem_ns': source_remaining, 'remaining_ns': D-fresh2,
                    'queue_prediction_used': False}}

    def _zero_wait_completion_ns(self, kind, gpu=None, extra_ns=0):
        from execution_guards import minimum_inference_ns
        return minimum_inference_ns(kind, self.mean_ns, self.gpu_ipc_budget,
            self.network_return_ns, affinity=gpu if kind == 'FAST_SUFFIX' else None,
            inline=self.inline_fast_continuation, head_ns=self.decision_budget,
            extra_ns=extra_ns, model_only=self.spec.get('edge_model_only'))

    def _reject(self, task, kind, role, stage, component_id, outcome, now):
        """REJECT and ADMISSION_OUTCOME rejected are ALWAYS written together and
        are the only source of the observer's R (T012 C2)."""
        i = task['task_id']
        event_ns = self.emit('REJECT', task_id=i, job_id=None, kind=kind, role=role, stage=stage,
                  root_id=task.get('root_id'), component_id=component_id,
                  reason=outcome['reason'], remaining_ns=int(outcome['remaining_ns']),
                  head_waited_ns=0, evidence=outcome.get('evidence') or {})
        if self.controller is not None:
            self.controller.observer.component_reject(
                component_id, outcome['reason'], event_ns, kind=kind, role=role)
        self.emit('ADMISSION_OUTCOME', task_id=i, outcome='rejected',
                  reason=outcome['reason'], target_kind=kind, role=role,
                  remaining_ns=int(outcome['remaining_ns']), head_waited_ns=0,
                  evidence=outcome.get('evidence') or {},
                  root_id=task.get('root_id'), component_id=component_id)
        self.rejected_components += 1
        self._profile_update(role, kind, task, now, outcome.get('evidence') or {})

    def _admit(self, task, kind, role, stage, gpu=None, path='A', extra_ns=0,
               pre_push=None, pre_reject=None):
        """OFFER accounting -> admission -> ENQUEUE or REJECT (returns job/None);
        ``pre_push``/``pre_reject`` let a caller release its own commitment on
        the exact branch the old inline code used."""
        i = task['task_id']
        comp = self.offer(i, kind, stage)
        outcome = self._admission(task, kind, gpu=gpu, stage=stage, extra_ns=extra_ns)
        # T014 s3.2.2: freeze the DEADLINE OPPORTUNITY (deadline - arrival) of
        # every root offered to a GPU direction, per direction, so the window
        # shape can be compared without any execution-outcome bias.
        if self.controller is not None and role in ('Fast', 'Accurate'):
            try:
                self.controller.observer.note_deadline_opportunity(
                    role, int(task['deadline_abs_ns']) -
                    int(task.get('arrival_ns') or 0))
            except Exception:
                self.counters['deadline_opportunity_errors'] = \
                    int(self.counters.get('deadline_opportunity_errors', 0)) + 1
        if outcome['ok']:
            task['path'] = task.get('path') or path
            if pre_push is not None:
                pre_push()
            j = self.push(i, kind, gpu=gpu, stage=stage, component_id=comp)
            self.emit('ADMISSION_OUTCOME', task_id=i, outcome='enqueued',
                      target_kind=kind, role=role,
                      eta_called=False,
                      admission_rule=outcome.get('budget_source'),
                      zero_wait_ns=(outcome.get('evidence') or {}).get('zero_wait_ns'),
                      remaining_ns=int(outcome['remaining_ns']),
                      estimate_ns=(None if outcome['estimate_ns'] is None
                                   else float(outcome['estimate_ns'])),
                      horizon_ns=(None if outcome['horizon_ns'] is None
                                  else int(outcome['horizon_ns'])),
                      deadline_abs_ns=int(task['deadline_abs_ns']),
                      root_id=task.get('root_id'), component_id=comp)
            return j
        if pre_reject is not None:
            pre_reject()
        self._reject(task, kind, role, stage, comp, outcome, outcome['now_ns'])
        return None

    # -- warmup calibration (section 4) -----------------------------------
    def run_warmup(self):
        """One-start calibration through the NORMAL actor protocol: per card
        Full 8 discard + 32 keep, Accurate 8 + 32, Prefix->Suffix pairs 8 + 32
        (same negative task_id / bank_index 0 per pair, unique across pairs and
        cards); cards run in parallel, card-internal serial; no label read and
        warmup jobs never enter formal counts (negative ids)."""
        base = self.warmup_id_base
        plans = {g: [] for g in CARD_IDS}
        seq = 0
        for g in CARD_IDS:
            for kind in ('FAST_FULL', 'ACCURATE'):
                for k in range(WARMUP_DISCARD + WARMUP_KEEP):
                    seq += 1
                    plans[g].append({'kind': kind, 'task_id': base + seq,
                                     'discard': k < WARMUP_DISCARD})
            for p in range(WARMUP_DISCARD + WARMUP_KEEP):
                seq += 1
                tid = base + seq
                plans[g].append({'kind': 'FAST_PREFIX', 'task_id': tid,
                                 'pair': p, 'discard': p < WARMUP_DISCARD})
                plans[g].append({'kind': 'FAST_SUFFIX', 'task_id': tid,
                                 'pair': p, 'discard': p < WARMUP_DISCARD})
        records = {g: [] for g in CARD_IDS}
        done_count = {g: 0 for g in CARD_IDS}
        ptr = {g: 0 for g in CARD_IDS}
        reset_ok = {g: False for g in CARD_IDS}
        head_vector = None

        def push_next(g):
            if ptr[g] >= len(plans[g]):
                return
            item = plans[g][ptr[g]]
            j = {'job_id': -1, 'task_id': item['task_id'], 'kind': item['kind'],
                 'bank_index': 0, 'enqueue_ns': 0,
                 'affinity': g if item['kind'] == 'FAST_SUFFIX' else None,
                 'warmup': True, 'dispatch_ns': time.perf_counter_ns()}
            self.commands['gpu:%d' % g].put(j)
            ptr[g] += 1
            # after a Prefix the NEXT push (triggered by its DONE) sends the
            # matching Suffix with the same task_id

        for g in CARD_IDS:
            push_next(g)
        deadline = time.monotonic() + 600
        while any(done_count[g] < len(plans[g]) for g in CARD_IDS) \
                and time.monotonic() < deadline:
            self.pump_fake_actors()
            try:
                r = self.results.get(timeout=30)
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError('group %s warmup stalled: done=%r'
                                       % (self.name, done_count))
                continue
            if r.get('type') == 'ERROR':
                raise RuntimeError('warmup actor error: %r' % (r,))
            if r.get('type') == 'RESET':
                reset_ok[int(r['resource'].split(':')[1])] = True
                continue
            if r.get('type') != 'DONE':
                continue
            j = r['job']
            g = int(r['resource'].split(':')[1])
            if j['kind'] == 'FAST_PREFIX':
                head_vector = r.get('vector')
                item = next(it for it in plans[g]
                            if it['task_id'] == j['task_id'] and it['kind'] == 'FAST_PREFIX')
            else:
                item = plans[g][ptr[g] - 1]
            records[g].append({
                'card': g, 'uuid': self.gpu_uuids[g], 'kind': j['kind'],
                'start_ns': r['start_ns'], 'finish_ns': r['finish_ns'],
                'cuda_ms': r.get('cuda_ms'),
                'dispatch_to_start_ns': r['start_ns'] - j['dispatch_ns'],
                'service_ns': r['finish_ns'] - r['start_ns'],
                'task_id': j['task_id'], 'discard': bool(item['discard'])})
            done_count[g] += 1
            push_next(g)
        if any(done_count[g] < len(plans[g]) for g in CARD_IDS):
            raise RuntimeError('group %s warmup incomplete: done=%r'
                               % (self.name, done_count))
        # a clean RESET proves the warmup left no activation behind (s4)
        for g in CARD_IDS:
            self.commands['gpu:%d' % g].put({'kind': 'RESET'})
        while not all(reset_ok.values()) and time.monotonic() < deadline:
            self.pump_fake_actors()
            try:
                r = self.results.get(timeout=10)
            except Exception:
                continue
            if r.get('type') == 'ERROR':
                raise RuntimeError('warmup RESET failed (cache not empty?): %r' % (r,))
            if r.get('type') == 'RESET':
                reset_ok[int(r['resource'].split(':')[1])] = True
        if not all(reset_ok.values()):
            raise RuntimeError('group %s warmup RESET not confirmed' % self.name)
        # seed the cost book from KEEP samples only: floor = this card's P95
        # T012 R1: the stamp is the FORMAL relative clock origin (0), never an
        # absolute perf_counter_ns(); the run path feeds a relative now().
        now = 0
        seeded = {}
        for g in CARD_IDS:
            by_kind = {}
            for rec in records[g]:
                if rec['discard']:
                    continue
                by_kind.setdefault(rec['kind'], []).append(rec)
            for kind, recs in by_kind.items():
                services = [r['service_ns'] for r in recs]
                gaps = [r['dispatch_to_start_ns'] for r in recs]
                ok = self.cost_book.seed_warmup(g, kind, services, gaps, now)
                seeded['%d:%s' % (g, kind)] = {
                    'n': len(services),
                    'mean_ms': self.cost_book.mean_ns(g, kind) / 1e6,
                    'floor_ms': self.cost_book.floor_ns(g, kind) / 1e6,
                    'gate_ms': self.cost_book.gate_ns(g, kind) / 1e6,
                    'delta_ms': (self.cost_book.delta_ns(g, kind) or 0) / 1e6,
                    'seeded': bool(ok)}
        for g in CARD_IDS:
            for kind in GPU_KINDS:
                gate = self.cost_book.gate_ns(g, kind)
                if gate is None:
                    raise RuntimeError('warmup missing measured gate for card %d kind %s'
                                       % (g, kind))
                self.gate_ns[g][kind] = int(round(gate))
                self.mean_ns[g][kind] = int(round(self.cost_book.mean_ns(g, kind)))
                self.cost_ns[g][kind] = self.mean_ns[g][kind]
                self.cost_floor_ns[g][kind] = int(round(self.cost_book.floor_ns(g, kind)))
        self._update_delivery_budget()
        head_calibration = None
        if self.async_prefix_pipeline:
            durations = []
            for sample in range(WARMUP_DISCARD + WARMUP_KEEP):
                sent = time.perf_counter_ns()
                self.commands['decision:0'].put({'job_id': -1, 'task_id': -1,
                    'kind': 'FAST_DECISION', 'warmup': True, 'vector': head_vector,
                    'enqueue_ns': sent, 'dispatch_ns': sent})
                hr = self.results.get(timeout=30)
                if hr.get('type') != 'DECISION_DONE':
                    raise RuntimeError('CPU Head calibration failed: %r' % hr)
                if sample >= WARMUP_DISCARD:
                    durations.append(int(hr['finish_ns']) - int(hr['start_ns']))
            self.pipeline_head_samples.extend(durations)
            self.decision_budget = sorted(durations)[max(0, int(.95 * len(durations)) - 1)]
            head_calibration = {'samples': len(durations), 'service_ns': durations,
                                'p95_service_ns': self.decision_budget}
        elif self.gpu_quality_head:
            head_calibration = {'mode': 'GPU_EARLY', 'accounted_in': 'FAST_PREFIX',
                                'additional_cpu_head_budget_ns': 0}
        self.warmup_report = {
            'schema': 'T011_warmup_v1', 'component_calls_total': CARD_COUNT * 160,
            'component_calls_total_note': 'per arm: configured physical card count x 160 real component calls',
            'per_card_calls': {g: len(records[g]) for g in CARD_IDS},
            'seeded': seeded,
            'cache_reset_confirmed': {g: bool(reset_ok[g]) for g in CARD_IDS},
            'uuids': list(self.gpu_uuids),
            'source_prior_legacy_ms': dict(self.source_prior_legacy),
            'pipeline_head_calibration': head_calibration,
        }
        write(self.folder / 'warmup.json', self.warmup_report)
        write(self.folder / 'warmup_records.json', records)
        return self.warmup_report

    def build_controller(self):
        """Create the qf4 controller from the MEASURED warmup prices (s3.3:
        validation happens before any formal actor work)."""
        from queue_feedback_controller import (QueueFeedbackController,
                                               make_controller_inputs_qf4)
        # Initialize public demand accounting before the controller. Only
        # historical objectives create a candidate replay worker.
        if self.replay_worker is None and self.replay_worker_status == 'init':
            self._start_replay_worker()
        cost_records = self.cost_book.records_dict()
        c_star = self.cost_book.reference_c_star_ms()
        ref = dict(self.reference)
        # T012 s1 frozen identity: the T011 reference file still declares
        # qf4/QF-4, the factory only accepts the frozen qf4r identity.
        configuration = reference_configuration(ref)
        ref['configuration'] = configuration
        ref['c_ref_ms'] = c_star
        cfg = make_controller_inputs_qf4(
            ref, cost_records, self.deadline_ns, self.gpu_uuids, self.policy,
            start_ms=self.start_ms, initial_fast=self.initial_fast,
            folder=str(self.folder), cost_book=self.cost_book,
            worker=self.replay_worker)
        cfg['inline_fast_continuation'] = self.inline_fast_continuation
        self.controller = QueueFeedbackController(cfg)
        dc = self.controller.dual_clock
        if (dc is not None and dc.mode == 'FULL' and dc.c.get('net_gain_enabled')
                and not dc.c.get('independent_checks_enabled')):
            from net_gain_control import PublicHistory
            self.net_gain_history = PublicHistory(dc.c['net_gain_max_roots'])
        # T012 s3.4: ONE authoritative CostBook shared by the runtime, the
        # controller and its observer; nothing may keep a private copy.
        self.controller.cost_book = self.cost_book
        self.controller.observer.cost_book = self.cost_book
        seeded = self.controller.seed_costs(cost_records, 0)
        self.controller.cost_book = self.cost_book
        self.controller.observer.cost_book = self.cost_book
        self.c_star_ms = seeded
        self._refresh_costs(0)
        self.rebase_clock()          # no-op until the shared t0 is known
        return seeded

    def rebase_clock(self):
        """T012 R1: re-anchor every cost stamp onto the formal relative clock.

        Called as soon as the shared ``t0`` is known (the joint barrier release
        happens AFTER warmup), so the 250 ms price/gate refresh cannot be frozen
        by an absolute warmup stamp.
        """
        if self.t0_ns is None:
            return None
        self.cost_book.rebase_clock(self.t0_ns)
        self.cost_refresh_ns = 0
        return self.t0_ns

    # -- candidate replay worker (T012 C11) --------------------------------
    def _start_replay_worker(self):
        """One shared non-blocking replay worker per arm (same for the fixed
        arm); a missing module is an explicit prediction_unknown, never a
        synchronous 11-candidate replay in the scheduler."""
        self.replay_profile_ring = None
        self.net_gain_history = None
        self.replay_module = None
        self.replay_quality_gain_history = deque(maxlen=16)
        try:
            from candidate_replay_t012 import ReplayWorker, ProfileRing
            import candidate_replay_t012 as mod
        except Exception as exc:
            self.replay_worker = None
            self.replay_error = repr(exc)
            self.replay_worker_status = 'absent'
            return False
        try:
            self.replay_module = mod
            if self.delivery_objective == 'latency_timeout':
                from resource_axis import PublicDemandLedger
                configuration = reference_configuration(self.reference)
                self.replay_profile_ring = PublicDemandLedger(TICK_NS,
                    int(configuration.get('source_profile_ms', 2000) * 1_000_000))
                self.replay_worker = None
                self.replay_worker_status = 'direct_demand'
                self.replay_result = {'valid': False,
                                      'invalid_reason': 'disabled_direct_demand'}
                return True
            replay_cpu = os.environ.get('T014_REPLAY_CPU_%s' % self.policy)
            self.replay_worker = ReplayWorker(
                cpu=None if replay_cpu is None else int(replay_cpu),
                immutable_payload_reply=True)
            self.replay_profile_ring = ProfileRing()
            self.replay_worker_status = 'started'
            return True
        except Exception as exc:
            self.replay_worker = None
            self.replay_error = repr(exc)
            self.replay_worker_status = 'failed'
            return False

    def _replay_profile_snapshot(self, now):
        ring = getattr(self, 'replay_profile_ring', None)
        empty = {'roots': [], 'overflow': False, 'n': 0, 'span_ns': 0,
                 'first_publish_ns': None}
        if ring is None:
            return empty
        try:
            snap = ring.snapshot(int(now), include_diagnostics=False)
            return snap if isinstance(snap, dict) else empty
        except Exception:
            self.replay_errors += 1
            return empty

    def _replay_publish(self, root_id, publish_ns, deadline_abs_ns, kind, rem_ns,
                        entry_class=None, arrival_ns=None, revealed_role=None,
                        public_quality=None, input_receipt_ns=None):
        """One public root enters the replay ProfileRing in the EXACT shape the
        ring/kernel read back (source_publish_ns / original_arrival_ns /
        deadline_ns / entry_class / kind / budget_ns).  Re-adding the same
        root_id only refreshes its public record (the ring de-duplicates).

        T014 s3.2.5: the public quality probability is carried through, so Vhat
        is a real public-quality proxy and an absent probability stays None
        (never imputed as 0).
        """
        ring = getattr(self, 'replay_profile_ring', None)
        if ring is None or root_id is None:
            return
        record = {'root_id': root_id,
                  'source_publish_ns': int(publish_ns),
                  'original_arrival_ns': int(arrival_ns if arrival_ns is not None
                                             else publish_ns),
                  'deadline_ns': int(deadline_abs_ns),
                  'budget_ns': int(rem_ns),
                  'input_receipt_ns': max(0, int(input_receipt_ns if input_receipt_ns is not None
                                                else self.now() - int(publish_ns))),
                  'kind': kind,
                  'entry_class': entry_class,
                  'stage_rem_ns': int(rem_ns),
                  'source_seq': 0}
        if revealed_role is not None:
            record['revealed_role'] = revealed_role
        if public_quality is not None and not isinstance(public_quality, bool):
            try:
                record['public_quality'] = float(public_quality)
            except (TypeError, ValueError):
                pass
        history = getattr(self, 'net_gain_history', None)
        if history is not None:
            started = time.process_time_ns()
            history.observe(record, self.now())
            stats = self.controller.dual_clock.stats
            stats['net_gain_history_cpu_ns'] = stats.get('net_gain_history_cpu_ns',0)+time.process_time_ns()-started
        try:
            ring.add(record)
        except Exception:
            self.replay_errors += 1

    def _start_latency_by_card(self):
        """Measured dispatch->actor-start per card (never a silent 0)."""
        out = []
        for g in CARD_IDS:
            vals = []
            for kind in GPU_KINDS:
                try:
                    rec = self.cost_book.record_obj(g, kind)
                except Exception:
                    continue
                if rec.delta_ns:
                    vals.append(float(rec.delta_ns))
            out.append(int(max(vals)) if vals else 0)
        return out

    def _input_receipt_budget(self):
        """The measured HEALTHY SOURCE -> group input receipt delay (T014
        s3.2.5).  The output delivery budget is a different quantity and is
        never used to move the future input."""
        lags = list(self.perf.get('receipt_lag_ns') or ())
        if not lags:
            return 0, 'no_input_receipt_samples'
        p95 = _pctl(lags, 0.95)
        if p95 is None:
            return 0, 'no_input_receipt_samples'
        return int(p95), 'healthy_source_receipt_p95'

    def _direct_demand_snapshot(self, now, span):
        """Bounded bucket-count query; no whole-task scans or sorting."""
        ring = getattr(self, 'replay_profile_ring', None)
        if ring is None or not hasattr(ring, 'work_summary'):
            return {'valid': False, 'reason': 'direct_demand_ledger_unavailable'}
        costs = {g: {k: float(v) for k, v in self.mean_ns[g].items()}
                 for g in CARD_IDS}
        return ring.work_summary(max(0, int(now)-int(span)), int(now), costs)

    def _replay_payload(self, now):
        """The frozen s6.5 request payload.  The cost/gate/ipc/head tables are
        attached so the WORKER can actually run the kernel (they are part of the
        worker payload contract, never passed in the request only)."""
        grid = list(getattr(self.replay_module, 'CANDIDATE_GRID', ()) or ())
        if any(s['job'] is not None and s['job'].get('kind') == 'FAST_PREFIX'
               for s in self.gpus):
            scenario = 'unknown_F'
        elif any(s['job'] is not None and s['job'].get('kind') == 'ACCURATE'
                 for s in self.gpus):
            scenario = 'unknown_A'
        else:
            scenario = 'point'
        book = self.cost_book
        version = None
        if hasattr(book, 'global_cost_version'):
            try:
                version = book.global_cost_version()
            except Exception:
                version = None
        if version is None:
            version = book.cost_version()
        costs = {g: {k: float(v) for k, v in self.mean_ns[g].items()}
                 for g in CARD_IDS}
        gates = {g: {k: float(v) for k, v in self.gate_ns[g].items()}
                 for g in CARD_IDS}
        profile = self._replay_profile_snapshot(now)
        reference = demand_reference(profile, costs)
        if self.resource_accounting == 'service_debt':
            # Fractional service can accumulate across the old 50 ms boundary.
            # A floor(budget / whole-job cost) prefilter would hide real changes.
            grid = bounded_grid(self._share_fast(), reference['gF_ref'],
                                self.replay_requests, self.controller.grid_points)
        else:
            grid = local_opportunity_grid(self._share_fast(), gates, profile)
        if self.delivery_objective == 'latency_timeout':
            pending = self.controller.pending
            review = self.controller.held_review
            if pending is None and review:
                pending = review.get('item')
            if pending:
                grid = sorted(set(grid + [float(pending['command']['from_fast']),
                    float(pending['command']['target_fast'])]))
        # T014 s3.2.5: three DISTINCT latencies.  The input receipt budget is the
        # measured source->group lag; the output delivery budget and the
        # dispatch->actor-start latency are reported separately and are never
        # substituted for it.  There is NO axis-locked ``direction`` any more:
        # the candidate/trigger filter inside the controller decides the legal
        # axis (T014 s3.3.2, derivation defect 4).
        receipt_ns, receipt_src = self._input_receipt_budget()
        epoch = None
        try:
            epoch = int(self.controller.epoch)
        except Exception:
            epoch = None
        history_eligible = (self.resource_accounting == 'service_debt'
            and self.delivery_objective == 'quality_weighted_earliness'
            and self.controller.state == 'IDLE'
            and self.controller._freeze_pre_cohorts(now) is not None)
        return {'profile': profile,
                'demand_reference': reference,
                'grid': grid or None,
                'resource_accounting': self.resource_accounting,
                'allow_idle_endpoint_loans': self.allow_idle_endpoint_loans,
                'delivery_objective': self.delivery_objective,
                'protect_role_outcomes': self.resource_accounting == 'service_debt',
                'service_debt_balance': [dict((s.get('_service_debt') or {}).get(
                    'balance') or {'Fast': 0.0, 'Accurate': 0.0}) for s in self.gpus],
                'quality_gain_history': (list(getattr(self, 'replay_quality_gain_history', ()))
                    if self.resource_accounting == 'service_debt' else []),
                'quality_history_eligible': history_eligible,
                'cycle_origin_ns': int((int(now) // PERIOD_NS) * PERIOD_NS),
                'receipt_budget_ns': int(receipt_ns),
                'receipt_budget_source': receipt_src,
                'delivery_budget_ns': int(self.delivery_budget()),
                'delivery_budget_source': self.delivery_budget_source,
                'ipc_return_ns': [int(v) for v in self.gpu_ipc_budget],
                'scenario': 'point',
                'price_version': version,
                'epoch': epoch,
                'snapshot_ns': int(now),
                'current_gF': float(self._share_fast()),
                'direction': None,
                'costs': costs,
                'gates': gates,
                'start_ns': self._start_latency_by_card(),
                'ipc': [int(v) for v in self.gpu_ipc_budget],
                'head_ns': {'service_ns': int(self.decision_budget),
                            'message_ns': int(max(self.gpu_ipc_budget))}}

    def _maybe_replay(self, now):
        """ONE non-blocking request per NEW 500 ms window (never 11 candidates
        on the scheduler thread)."""
        if self.controller is None or self.delivery_objective == 'latency_timeout':
            return None
        window = (int(now) // REPLAY_TICK_NS) * REPLAY_TICK_NS
        if window == self.replay_window:
            return None
        self.replay_window = window
        w = self.replay_worker
        if w is None:
            self.replay_result = {'valid': False, 'invalid_reason': 'prediction_unknown'}
            return None
        # Payload tables are owned by this request. ProfileRing replaces
        # roots rather than mutating published records; history entries are
        # append-only snapshots. Retain this input for sequence-matched replies.
        payload = self._replay_payload(now)
        t0 = time.perf_counter_ns()
        try:
            w.request(payload, int(now))
            self.streams['events'].write({'event': 'REPLAY_REQUEST', 't_ns': int(now), 'payload': payload})
            self.replay_requests += 1
            self.replay_worker_status = 'requested'
        except Exception:
            self.replay_errors += 1
            self.replay_worker_status = 'request_failed'
            self.replay_result = {'valid': False, 'invalid_reason': 'prediction_unknown'}
        self._note_worker(time.perf_counter_ns() - t0)
        return self._poll_replay(now)

    def _poll_replay(self, now):
        w = self.replay_worker
        if w is None:
            self.replay_result = {'valid': False, 'invalid_reason': 'prediction_unknown'}
            return None
        t0 = time.perf_counter_ns()
        res = None
        try:
            res = w.poll(int(now))
        except Exception:
            self.replay_errors += 1
            self.replay_worker_status = 'poll_failed'
        self._note_worker(time.perf_counter_ns() - t0)
        stats = {}
        try:
            stats = w.stats() or {}
        except Exception:
            stats = {}
        if res is not None:
            # T012 R2-D: a result older than REPLAY_AGE_GATE_NS is not a current
            # prediction; it is reported explicitly instead of being consumed.
            age_ns = res.get('age_ns')
            if age_ns is not None and int(age_ns) > REPLAY_AGE_GATE_NS:
                self.replay_stale_results += 1
                res = dict(res)
                res['valid'] = False
                res['invalid_reason'] = 'stale_result'
                res['cause'] = 'age_gate'
                self.replay_result = res
                self.replay_worker_status = 'stale_result'
                return res
            self._remember_replay_quality_gain(res, now)
            self.replay_result = res
            self.replay_worker_status = 'result'
            self.streams['events'].write({'event': 'REPLAY_RESULT', 't_ns': int(now), 'result': res})
        elif stats and not stats.get('alive', True) and self.replay_worker_status != 'result':
            self.replay_worker_status = 'dead'
            self.replay_result = {'valid': False, 'invalid_reason': 'prediction_unknown'}
        return res

    def _remember_replay_quality_gain(self, res, now):
        """Store one fresh, healthy public gain table, never its large profile."""
        if self.resource_accounting != 'service_debt' or not res.get('valid') or \
                not res.get('quality_history_eligible'):
            return False
        epoch = res.get('epoch')
        if epoch != self.controller.epoch or \
                abs(float(res.get('reference_gF', -99))-self._share_fast())>1e-9:
            return False
        stamp = res.get('snapshot_ns')
        if stamp is None or not 0 <= int(now)-int(stamp) <= REPLAY_AGE_GATE_NS:
            return False
        gains = {str(r['gF']):r['net_gain_quality_earliness']
                 for r in res.get('candidates', ())
                 if r.get('net_gain_quality_earliness') is not None}
        if not gains:return False
        history = getattr(self, 'replay_quality_gain_history', None)
        if history is None:
            self.replay_quality_gain_history = history = deque(maxlen=16)
        if history and history[-1]['snapshot_ns'] == stamp:return False
        history.append({'epoch':epoch,'reference_gF':res['reference_gF'],
            'snapshot_ns':stamp,'profile_sha':res.get('profile_sha'),
            'signature':res['quality_gain_signature'],'H_gain':gains})
        return True

    def _stop_replay_worker(self):
        w = self.replay_worker
        if w is None:
            return False
        try:
            w.stop()
            self.replay_worker_status = 'stopped'
        except Exception:
            self.replay_errors += 1
            self.replay_worker_status = 'stop_failed'
        return True

    # -- frame handling ---------------------------------------------------
    def _receipt(self, frame, now):
        lag = now - int(frame.get('source_publish_ns', now))
        self.receipt_lag_max_ns = max(self.receipt_lag_max_ns, lag)
        self.perf['receipt_lag_ns'].append(max(int(lag), 0))
        if self.controller is not None:
            self.controller.observer.record_receipt_lag(lag, now)
        # T012 C7: the sequence/watermark check covers EVERY outcome, not only
        # ARRIVAL, so a dropped QUALITY/PREFIX frame is visible as a gap.
        outcome = frame.get('outcome')
        if outcome:
            key = 'source_outcome_' + str(outcome)
            self.counters[key] = self.counters.get(key, 0) + 1
        fid = frame.get('frame_id')
        if not isinstance(fid, int) or fid < 0:
            if outcome != 'SOURCE_END':
                self.counters['source_frames_unsequenced'] += 1
                self.frame_gap_seen = True
            return
        if fid in self.frame_publish_ns or fid <= self.last_contig_frame_id:
            self.counters['source_duplicate_frame_ids'] += 1
            self.frame_gap_seen = True
            return
        self.frame_publish_ns[fid] = int(frame.get('source_publish_ns', now))
        if fid == self.last_contig_frame_id + 1:
            self.last_contig_frame_id = fid
            self.watermark_ns = int(frame.get('source_publish_ns', now))
            while (self.last_contig_frame_id + 1) in self.frame_publish_ns:
                self.last_contig_frame_id += 1
                self.watermark_ns = self.frame_publish_ns[self.last_contig_frame_id]
        elif fid > self.last_contig_frame_id + 1:
            self.frame_gap_seen = True
        if len(self.frame_publish_ns) > 8192:
            floor = self.last_contig_frame_id - 4096
            self.frame_publish_ns = {k: v for k, v in self.frame_publish_ns.items()
                                     if k >= floor}

    def _later(self, due, kind, payload):
        self.timed_sequence += 1
        heapq.heappush(self.timed_events, (int(due), self.timed_sequence, kind, payload))

    def _drain_timed_events(self):
        while self.timed_events and self.timed_events[0][0] <= self.now():
            due, _, kind, payload = heapq.heappop(self.timed_events)
            if kind == 'NETWORK_RETURN':
                self.finish(**dict(payload, _network_delivered=True,
                                   terminal_available_ns=self.now()))
            elif payload['task_id'] not in self.ended:
                self._queue_input_outcome(payload, self.now(), receipt=False)

    def _edge_waiting_ids(self):
        waiting = {j['task_id'] for jobs in self.ready.values() for j in jobs}
        waiting.update(f['task_id'] for f in self.pending_input_outcomes)
        for state in self.gpus:
            waiting.update(j['task_id'] for j in state['pinned'])
            waiting.update(state['pending'])
            if state['busy'] and state['job'] is not None:
                waiting.discard(state['job']['task_id'])
        return waiting.difference(self.ended)

    def _queue_input_outcome(self, frame, now, receipt=True):
        """Read arrivals promptly; run expensive GPU admission in its own slice."""
        if self.fair_input_processing and frame.get('outcome') in (
                self.OUTCOME_PREFIX_READY, self.OUTCOME_GPU_READY):
            if receipt:
                self.counters['frames'] += 1
                self._receipt(frame, now)
            waiting = self._edge_waiting_ids()
            if (frame['task_id'] not in waiting and
                    len(waiting) >= int(self.spec.get('edge_waiting_roots', 64))):
                # Register the demand/OFFER and reject through the same capacity
                # guard; a full queue never runs the expensive ETA forecast.
                self._handle_frame(frame, self.now(), receipt=False)
                return
            self.pending_input_outcomes.append(dict(frame, _runtime_scheduler_received_ns=int(now)))
            self.counters['input_work_queue_max'] = max(
                self.counters['input_work_queue_max'], len(self.pending_input_outcomes))
            return
        self._handle_frame(frame, now, receipt=receipt)

    def _drain_input_outcomes(self):
        deadline = time.perf_counter_ns() + SOURCE_SLICE_NS
        handled = 0
        while self.pending_input_outcomes and handled < FRAME_BATCH_MAX:
            frame = self.pending_input_outcomes.popleft()
            fresh = self.now()
            waited = max(0, fresh - frame['_runtime_scheduler_received_ns'])
            self.input_admission_wait_samples.append(waited)
            self.input_admission_wait_max_ns = max(self.input_admission_wait_max_ns, waited)
            task = self.tasks.get(frame['task_id'])
            if task is not None:
                task['input_admission_wait_ns'] = waited
            self._handle_frame(frame, fresh, receipt=False)
            handled += 1
            if time.perf_counter_ns() >= deadline:
                break
        return handled

    def _handle_input_frame(self, frame, now):
        if self.spec.get('frozen_input_audit') and (
                frame.get('outcome') != 'SOURCE_END' or '_replay_raw_digest' in frame):
            from shared_local_t011 import source_frame_digest
            if self.spec.get('dedicated_ingress'):
                digest = frame.get('_replay_ingress_digest')
                if digest is None:
                    raise RuntimeError('Frame bypassed the dedicated ingress audit')
            else:
                digest = source_frame_digest(frame)
            if digest != frame.get('_replay_raw_digest'):
                raise RuntimeError('Raw source frame changed before policy routing')
            if frame['frame_id'] != self.raw_input_frames + 1:
                raise RuntimeError('Missing, duplicate or reordered raw source frame')
            self.raw_input_frames += 1
            self.raw_input_digest.update(bytes.fromhex(digest))
            ingress_at = int(frame.get('_replay_ingress_received_ns', now))
            ingress_lag = max(0, ingress_at - int(frame.get('source_publish_ns', ingress_at)))
            scheduler_wait = max(0, int(now) - ingress_at)
            self.ingress_lag_samples.append(ingress_lag)
            self.scheduler_input_wait_samples.append(scheduler_wait)
            self.ingress_lag_max_ns = max(self.ingress_lag_max_ns, ingress_lag)
            self.scheduler_input_wait_max_ns = max(self.scheduler_input_wait_max_ns, scheduler_wait)
            if frame['outcome'] == self.OUTCOME_ARRIVAL:
                identity = (frame['bank_index'], frame['origin_node'], frame['arrival_ns'], frame['deadline_ns'])
                if self.raw_input_plan.get(frame['task_id']) != identity:
                    raise RuntimeError('Task image/node/arrival/deadline differs from frozen plan')
                self.raw_input_arrivals += 1
            self.streams['input_receipts'].write({**{k: frame.get(k) for k in
                ('frame_id', 'task_id', 'bank_index', 'origin_node', 'arrival_ns', 'deadline_ns',
                 'source_publish_ns', '_replay_raw_digest', '_replay_actual_publish_ns',
                 '_replay_ingress_received_ns')},
                'received_ns': int(now), 'scheduler_received_ns': int(now),
                'ingress_lag_ns': ingress_lag, 'scheduler_input_wait_ns': scheduler_wait})
        only = self.spec.get('edge_model_only')
        if only and frame.get('outcome') != 'SOURCE_END':
            from quality_need_v24 import baseline_source_frame
            transformed = baseline_source_frame(frame, only, int(self.spec['network_upload_ns']),
                                                direct_fast=bool(self.spec.get('single_model_direct')))
            if transformed is not frame:
                self.counters['frames'] += 1
                self._receipt(frame, now)
                self.emit('BASELINE_ROUTE', task_id=frame['task_id'], model=only,
                          original_outcome=frame['outcome'], outcome=transformed['outcome'],
                          reason=transformed['reason'])
                if transformed['source_publish_ns'] > now:
                    self._later(transformed['source_publish_ns'], 'BASELINE_FRAME', transformed)
                else:
                    self._queue_input_outcome(transformed, now, receipt=False)
                return
        self._queue_input_outcome(frame, now)

    def _handle_frame(self, frame, now, receipt=True):
        dc = getattr(self.controller, 'dual_clock', None)
        if dc is not None and not dc.disabled and dc.mode != 'HYSTERESIS':
            source_cpu_start = time.process_time_ns()
            try:
                dc.source.observe(frame, now)
            finally:
                dc.source_event_cpu_ns += time.process_time_ns()-source_cpu_start
                dc.source_event_calls += 1
        if receipt:
            self.counters['frames'] += 1
            self._receipt(frame, now)
        outcome = frame.get('outcome')
        if outcome == 'SOURCE_END':
            self.source_end = True
            if self.controller is not None:
                self.controller.observer.close_source(int(self.duration_s * NS_PER_S))
            return
        i = frame.get('task_id')
        if outcome == self.OUTCOME_ARRIVAL:
            if i in self.seen_arrival:
                self.duplicate_frames += 1
                return
            self.seen_arrival.add(i)
            self.tasks[i] = {'task_id': i, 'arrival_ns': int(frame['arrival_ns']),
                             'deadline_abs_ns': int(frame['deadline_ns']),
                             'origin_node': int(frame['origin_node']),
                             'bank_index': int(frame['bank_index']),
                             'path': None, 'injected_ns': now,
                             'root_id': None, 'p_selected_final': None}
            self.live.add(i)
            self.emit('ARRIVAL', task_id=i, arrival_ns=int(frame['arrival_ns']),
                      deadline_ns=int(frame['deadline_ns']),
                      frame_id=frame.get('frame_id'),
                      source_publish_ns=int(frame.get('source_publish_ns', 0)),
                      injection_lag_ns=int(now - int(frame['arrival_ns'])),
                      receipt_lag_ns=int(now - int(frame['source_publish_ns'])))
            return
        task = self.tasks.get(i)
        if task is None:
            self.counters['orphan_frames'] += 1
            return
        if outcome == self.OUTCOME_QUALITY:
            task['probabilities'] = frame.get('quality_probabilities')
            task['accurate_need_probability'] = frame.get('accurate_need_probability')
            self.emit('QUALITY_INTENT', task_id=i, stage=frame.get('stage'),
                      quality_role=frame.get('quality_role'),
                      quality_probabilities=frame.get('quality_probabilities'),
                      accurate_need_probability=frame.get('accurate_need_probability'))
            if self.controller is not None and frame.get('quality_role') in \
                    ('Local', 'Fast', 'Accurate'):
                self.controller.observer.cur['intent'][frame['quality_role']] += 1
            return
        if outcome == self.OUTCOME_LOCAL_TERMINAL:
            # The common Local suffix runs on the shared source, not on a GPU:
            # it uses a shape-valid component_id but never touches the GPU-only
            # observer component API (that would count unknown_kind).
            with self.handler_timed('local_terminal'):
                self.common_local_completed += 1
                self.counters['common_local_terminal'] += 1
                prob = frame.get('quality_probabilities') or {}
                psel = prob.get('Local')
                if isinstance(psel, bool):
                    psel = None
                task['p_selected_final'] = psel
                if task.get('root_id') is None:
                    task['root_id'] = 'root-%d' % i
                comp = self._component_id(task['root_id'], 'LOCAL_SUFFIX', 0)
                task['component_LOCAL_SUFFIX'] = comp
                self.components[comp] = {
                    'component_id': comp, 'root_id': task['root_id'], 'task_id': i,
                    'kind': 'LOCAL_SUFFIX', 'stage': 'LOCAL_SUFFIX', 'attempt': 0,
                    'state': 'finished', 'offer_ns': now}
                actor_finish = int(frame.get('real_finish_ns', now))
                compute_end = int(frame.get('compute_end_ns', actor_finish))
                parent_recv = int(frame.get('parent_received_ns', now))
                terminal_available = int(frame.get('terminal_available_ns', parent_recv))
                # s2: job_id is integer EXECUTION positioning only; the common
                # Local terminal has no GPU job, so it stays None (identity is
                # carried by the stable component_id).
                self.emit('ENQUEUE', job_id=None, kind='LOCAL_SUFFIX',
                          task_id=i, p_selected=psel, affinity=None,
                          stage='LOCAL_SUFFIX', component_id=comp,
                          bank_index=int(task['bank_index']),
                          deadline_ns=int(task['deadline_abs_ns']),
                          enqueue_ns=int(task.get('injected_ns') or 0))
                self.emit('COMPONENT_FINISH', task_id=i, component_id=comp,
                          kind='LOCAL_SUFFIX', resource='source:local',
                          start_ns=int(frame.get('real_start_ns', actor_finish)),
                          finish_ns=actor_finish,
                          service_ms=(actor_finish - int(frame.get('real_start_ns',
                                                                  actor_finish))) / 1e6,
                          dispatch_to_start_ms=0.0, compute_end_ns=compute_end,
                          actor_finish_ns=actor_finish,
                          parent_received_ns=parent_recv)
                self.finish(i, 'Local', frame.get('prediction'),
                            reason=frame.get('reason') or 'common Local terminal',
                            terminal_finish_ns=actor_finish, component_id=comp,
                            compute_end_ns=compute_end, actor_finish_ns=actor_finish,
                            parent_received_ns=parent_recv,
                            terminal_available_ns=terminal_available)
            return
        if outcome == self.OUTCOME_ABSTAIN:
            self.counters['common_abstain'] += 1
            self.finish(i, 'Abstain', None, reason=frame.get('reason') or 'shared source abstain')
            return
        if outcome == self.OUTCOME_PREFIX_READY:
            with self.handler_timed('prefix_ready'):
                self._prefix_ready(task, frame, now)
            return
        if outcome == self.OUTCOME_GPU_READY:
            with self.handler_timed('gpu_ready'):
                self._gpu_ready(task, frame, now)
            return
        self.counters['unknown_frames'] += 1

    def _classify_rejection(self, task, kind, now, stage, gpu=None):
        """Classify an actor's actual remaining-inference-budget refusal."""
        remaining = int(task['deadline_abs_ns'] - now)
        zero = self._zero_wait_completion_ns(kind, gpu=gpu)
        source_rem = int(task['deadline_abs_ns'] - int(task.get('source_publish_ns', task['arrival_ns'])))
        if source_rem < zero:
            return 'intrinsic_budget', {'source_rem_ns': source_rem, 'zero_wait_ns': int(zero)}
        if remaining < zero:
            return 'receipt_delay', {'receipt_rem_ns': remaining, 'zero_wait_ns': int(zero)}
        # A live actor can have a more conservative measured gate than the
        # mean table. Report that observed refusal without inventing queue ETA.
        return 'execution_budget', {'remaining_ns': remaining,
            'zero_wait_ns': zero, 'source_rem_ns': source_rem,
            'actor_budget_refusal': True, 'stage': stage}

    def _prefix_ready(self, task, frame, now):
        i = task['task_id']
        task['source_publish_ns'] = int(frame.get('source_publish_ns', now))
        task['replay_input_receipt_ns'] = max(0, int(now - task['source_publish_ns']))
        task['root_id'] = 'root-%d' % i
        task['gpu_root_registered'] = False
        task['entry_class'] = 'Prefix-origin'
        if self.controller is not None:
            task['gpu_root_registered'] = bool(self.controller.observer.root_publish(
                task['root_id'], 'Prefix-origin', task['source_publish_ns'],
                int(task['deadline_abs_ns']), frame_id=frame.get('frame_id'),
                first_kind='FAST_PREFIX',
                original_arrival_ns=task.get('arrival_ns'),
                stage_rem_ns=int(task['deadline_abs_ns'] - task['source_publish_ns'])))
        self._replay_publish(task['root_id'], task['source_publish_ns'],
                             int(task['deadline_abs_ns']), 'FAST_PREFIX',
                             int(task['deadline_abs_ns'] - task['source_publish_ns']),
                             entry_class='Prefix-origin',
                             arrival_ns=task.get('arrival_ns'),
                             input_receipt_ns=task['replay_input_receipt_ns'])
        # Physical budget uses only publicly possible continuations. No future
        # Head outcome or queue ETA is consulted before accepting this Prefix.
        j = self._admit(task, 'FAST_PREFIX', 'Fast', 'PREFIX', path='B')
        if j is not None:
            return
        self.counters['prefix_rejected'] += 1
        self.finish(i, 'Abstain', None, reason='prefix continuation rejected by group admission')

    def _profile_update(self, role, kind, task, now, evidence):
        """Bounded recent rejected profile (2 s) for the candidate basis."""
        self.rejected_profiles['%s:%s' % (role, kind)] = {
            'role': role, 'kind': kind, 'task_id': task['task_id'],
            'observed_ns': now, 'stage_rem_ns': evidence.get('remaining_ns'),
            'affinity': evidence.get('affinity'), 'evidence': dict(evidence)}

    def _gpu_ready(self, task, frame, now):
        i = task['task_id']
        task['probabilities'] = frame.get('quality_probabilities')
        task['accurate_need_probability'] = frame.get('accurate_need_probability')
        task['source_publish_ns'] = int(frame.get('source_publish_ns', now))
        task['replay_input_receipt_ns'] = max(0, int(now - task['source_publish_ns']))
        candidates = list(frame.get('gpu_candidates') or [])
        kinds = dict(frame.get('gpu_candidate_kinds') or {})
        if not candidates:
            self.counters['gpu_no_candidate'] += 1
            self.finish(i, 'Abstain', None, reason='no GPU continuation candidate')
            return
        first_role = candidates[0]
        kind = kinds.get(first_role) or ('ACCURATE' if first_role == 'Accurate' else 'FAST_FULL')
        entry = 'directA' if kind == 'ACCURATE' else 'directF'
        task['root_id'] = 'root-%d' % i
        task['quality_first_role'] = first_role
        task['gpu_root_registered'] = False
        task['entry_class'] = entry
        if self.controller is not None:
            task['gpu_root_registered'] = bool(self.controller.observer.root_publish(
                task['root_id'], entry, task['source_publish_ns'],
                int(task['deadline_abs_ns']), frame_id=frame.get('frame_id'),
                quality_role=first_role, probabilities=task['probabilities'],
                gpu_candidates=candidates, first_kind=kind,
                original_arrival_ns=task.get('arrival_ns'),
                stage_rem_ns=int(task['deadline_abs_ns'] - task['source_publish_ns'])))
        self._replay_publish(task['root_id'], task['source_publish_ns'],
                             int(task['deadline_abs_ns']), kind,
                             int(task['deadline_abs_ns'] - task['source_publish_ns']),
                             entry_class=entry, arrival_ns=task.get('arrival_ns'),
                             input_receipt_ns=task['replay_input_receipt_ns'],
                             public_quality=((task['probabilities'] or {}).get(first_role)))
        j = self._admit(task, kind, first_role, 'GPU_CONT')
        if j is not None:
            self.counters['gpu_admitted'] += 1
            return
        self.counters['gpu_rejected'] += 1
        self.finish(i, 'Abstain', None, reason='GPU continuation rejected by group admission')

    # -- path B (own post-Head): ONE quality role, never resource-rerouted --
    def _route_b(self, task_id, probabilities, gpu, need_probability, inline_result=None):
        from quality_need_v24 import quality_order
        task = self.tasks.get(task_id)
        if task is None:
            return
        now = self.now()
        enabled = bool(self.spec.get('quality_need_enabled', False))
        only = self.spec.get('edge_model_only')
        order = quality_order(probabilities, 'B', need_probability, enabled,
                              allowed_roles=[only] if only else None)
        task['probabilities'] = probabilities
        task['accurate_need_probability'] = need_probability
        self.emit('QUALITY_INTENT', task_id=task_id, path='B', stage='EARLY',
                  quality_role=(order[0] if order else 'Abstain'),
                  quality_probabilities=probabilities,
                  accurate_need_probability=need_probability)
        early_role = order[0] if order else 'Abstain'
        dc = getattr(self.controller, 'dual_clock', None)
        if dc is not None and not dc.disabled and dc.c.get('independent_checks_enabled'):
            started = time.process_time_ns()
            dc.source.observe(dict(outcome='PREFIX_QUALITY_REVEALED', task_id=task_id,
                arrival_ns=task['arrival_ns'], source_publish_ns=now,
                quality_role=early_role), now)
            dc.source_event_cpu_ns += time.process_time_ns()-started
            dc.source_event_calls += 1
        root_id = task.get('root_id')
        # EARLY reveal is public the moment the head decided (s5)
        supported = None
        if self.controller is not None and root_id:
            root = self.controller.observer.roots.get(root_id)
            if root is not None:
                supported = root['budget_supported']
            self.controller.observer.reveal_early(
                root_id, early_role if early_role in ('Fast', 'Accurate') else 'Abstain',
                supported=supported, p_quality=(probabilities or {}).get(early_role))
            # the replay profile may carry ONLY public information: the EARLY
            # reveal refreshes the root's record in the ring (same root_id).
            self._replay_publish(root_id, task.get('source_publish_ns', now),
                                 int(task['deadline_abs_ns']), 'FAST_PREFIX',
                                 int(task['deadline_abs_ns']
                                     - task.get('source_publish_ns', now)),
                                 entry_class='Prefix-origin',
                                 arrival_ns=task.get('arrival_ns'),
                                 revealed_role=(early_role if early_role in
                                                ('Fast', 'Accurate') else 'Abstain'),
                                 input_receipt_ns=task.get('replay_input_receipt_ns'),
                                 public_quality=(probabilities or {}).get(early_role))
        if task_id in self.ended:
            # deferred cleanup: root expired while the Head was in flight -
            # release the original card, never dispatch a back segment (s3.2)
            self.counters['deferred_head_drops'] += 1
            self._release_prefix_commitment(task_id, gpu, reason='expired_root')
            return
        if not order:
            self._release_prefix_commitment(task_id, gpu, reason='no_quality_role')
            self.finish(task_id, 'Abstain', None, reason='no quality eligible path B role')
            return
        role = order[0]
        if inline_result is not None:
            if inline_result.get('inline_role') != role:
                raise RuntimeError('GPU inline choice disagrees with public quality routing')
            if role == 'Fast':
                task['p_selected_final'] = probabilities['Fast']
                if inline_result.get('inline_suffix_done'):
                    self.finish(task_id, 'Fast', inline_result['prediction'],
                        terminal_finish_ns=inline_result['finish_ns'],
                        compute_end_ns=inline_result['suffix_compute_end_ns'],
                        actor_finish_ns=inline_result['finish_ns'], parent_received_ns=self.now(),
                        component_id=task.get('component_INLINE_SUFFIX'))
                elif inline_result.get('inline_suffix_skip') == 'remaining_deadline_insufficient':
                    self.counters['inline_deadline_refusals'] += 1
                    comp = self.offer(task_id, 'FAST_SUFFIX', 'INLINE_SUFFIX')
                    reason, evidence = self._classify_rejection(task, 'FAST_SUFFIX', self.now(), 'INLINE_SUFFIX', gpu=gpu)
                    self._reject(task, 'FAST_SUFFIX', 'Fast', 'INLINE_SUFFIX', comp,
                        {'reason': reason, 'evidence': dict(evidence, inline_remaining_deadline=True),
                         'remaining_ns': task['deadline_abs_ns']-self.now()}, self.now())
                    self.finish(task_id, 'Abstain', None, reason='inline Fast remaining deadline insufficient')
                else:
                    raise RuntimeError('Inline Fast report has neither output nor deadline refusal')
                return
        kind = 'FAST_SUFFIX' if role == 'Fast' else 'ACCURATE'
        affinity = gpu if kind == 'FAST_SUFFIX' else None

        def _pre_push():
            if kind == 'FAST_SUFFIX':
                self.gpus[gpu]['pending'].discard(task_id)
            else:
                self._release_prefix_commitment(task_id, gpu, reason='chose_accurate')

        def _pre_reject():
            self._release_prefix_commitment(task_id, gpu, reason='early_rejected')

        j = self._admit(task, kind, role, 'EARLY', gpu=affinity,
                        pre_push=_pre_push, pre_reject=_pre_reject)
        if j is not None:
            return
        self.counters['early_rejected'] += 1
        self.finish(task_id, 'Abstain', None, reason='path B continuation rejected')

    def _release_prefix_commitment(self, task_id, gpu, reason):
        """A/reject/Head-failure path: RELEASE the original card's activation
        and clear the Suffix reserve exactly once (s3.2)."""
        self.gpus[gpu]['pending'].discard(task_id)
        self.prefix_card.pop(task_id, None)
        if not self.inline_fast_continuation:
            self.commands['gpu:%d' % gpu].put({'kind': 'RELEASE', 'task_id': task_id})
        self.counters['prefix_releases'] += 1
        self.emit('PREFIX_RELEASE', task_id=task_id, gpu=gpu, reason=reason)

    def _record_inline_suffix(self, task, gpu, result):
        """Physical stage telemetry; never enqueue/dispatch a second GPU job."""
        parent = result['job']
        i = task['task_id']
        cid = self._component_id(task['root_id'], 'INLINE_SUFFIX', 0)
        task['component_INLINE_SUFFIX'] = cid
        suffix_job = dict(parent, job_id=parent['inline_suffix_job_id'], kind='FAST_SUFFIX',
            component_id=cid, affinity=gpu, parent_job_id=parent['job_id'], inline=True,
            enqueue_ns=result['suffix_start_ns'], dispatch_ns=result['suffix_start_ns'])
        record = {'type': 'INLINE_DONE', 'resource': 'gpu:%d' % gpu,
            'job': suffix_job, 'task_id': i, 'kind': 'FAST_SUFFIX', 'inline': True,
            'start_ns': result['suffix_start_ns'], 'finish_ns': result['suffix_finish_ns'],
            'compute_start_ns': result['suffix_compute_start_ns'],
            'compute_end_ns': result['suffix_compute_end_ns'], 'cuda_ms': result['suffix_cuda_ms'],
            'upload_ns': 0, 'prediction': result['prediction'],
            'feature_or_result_transfer_ns': result['suffix_finish_ns']-result['suffix_compute_end_ns']}
        self.components[cid] = {'component_id': cid, 'task_id': i, 'root_id': task['root_id'],
            'kind': 'FAST_SUFFIX', 'state': 'finished', 'inline': True,
            'start_ns': record['start_ns'], 'finish_ns': record['finish_ns']}
        if self.controller is not None:
            self.controller.observer.component_inline_service(cid, 'FAST_SUFFIX', task['root_id'],
                self.now(), record['finish_ns'], gpu)
        self.component(record)
        self.emit('INLINE_SUFFIX_COMPLETE', task_id=i, component_id=cid, gpu=gpu,
            start_ns=record['start_ns'], finish_ns=record['finish_ns'],
            score_accounted_in=parent['component_id'], extra_dispatch=False)
        self.counters['inline_suffixes_completed'] += 1

    # -- GPU scheduling ---------------------------------------------------
    def _expire_ready(self, now):
        expiry_now = now + int(bool(self.spec.get('finite_queue_policy', False)))
        for role in ('Fast', 'Accurate'):
            keep = []
            for j in self.ready[role]:
                task = self.tasks.get(j['task_id'])
                if task is None or j['task_id'] in self.ended:
                    self._observer_remove(j, 'superseded', now)
                    continue
                if expiry_now > task['deadline_abs_ns']:
                    event_ns = self.emit('REMOVE', task_id=j['task_id'], job_id=j['job_id'],
                              kind=j['kind'], reason='expired GPU queue',
                              root_id=task.get('root_id'),
                              component_id=self._job_component(j))
                    self._observer_remove(j, 'resource', event_ns)
                    self.finish(j['task_id'], 'Abstain', None,
                                reason='expired GPU queue')
                    continue
                keep.append(j)
            self.ready[role] = keep
        for g in CARD_IDS:
            keep = []
            for j in self.gpus[g]['pinned']:
                task = self.tasks.get(j['task_id'])
                if task is None or j['task_id'] in self.ended:
                    self._observer_remove(j, 'superseded', now)
                    continue
                if expiry_now > task['deadline_abs_ns']:
                    event_ns = self.emit('REMOVE', task_id=j['task_id'], job_id=j['job_id'],
                              kind=j['kind'], reason='expired pinned Suffix queue',
                              root_id=task.get('root_id'),
                              component_id=self._job_component(j))
                    self._observer_remove(j, 'resource', event_ns)
                    # T012 s3.2: an expired pinned Suffix must release the card
                    # activation and the commitment exactly once, and must never
                    # be dispatched afterwards.
                    if j['task_id'] in self.prefix_card:
                        self._release_prefix_commitment(
                            j['task_id'], self.prefix_card[j['task_id']],
                            'expired pinned Suffix queue')
                    else:
                        self.gpus[g]['pending'].discard(j['task_id'])
                    self.finish(j['task_id'], 'Abstain', None,
                                reason='expired pinned Suffix queue')
                    continue
                keep.append(j)
            self.gpus[g]['pinned'] = keep
        # Head-pending roots whose deadline passed while the Head flies:
        # mark deferred cleanup, the decision result will only release (s3.2)
        for i in list(self.head_pending):
            task = self.tasks.get(i) or {}
            if i in self.ended:
                continue
            if expiry_now > task.get('deadline_abs_ns', 0):
                self.finish(i, 'Abstain', None, reason='expired awaiting Head decision')
                self.deferred_cleanup.add(i)

    def _observer_remove(self, j, reason, now):
        if self.controller is not None:
            self.controller.observer.component_remove(self._job_component(j), reason, now)

    def _job_component(self, j):
        """The observer key is the stable component_id, never the integer job_id."""
        cid = j.get('component_id')
        if not cid:
            cid = self._component_id('root-%s' % j.get('task_id'), str(j.get('kind')), 0)
            j['component_id'] = cid
        return cid

    def _drop_if_inference_budget_exhausted(self, job, now):
        """Drop before booking GPU service only if no physical card can finish.

        The minimum ignores current role shares and busy queues. This avoids
        turning temporary contention into an irreversible admission decision.
        Same-card Suffix affinity remains mandatory; no service debt is charged
        to a task that never starts. Frozen experiment cards share one timing
        table, so this also checks the selected device's inference duration.
        """
        now = max(int(now), int(self.now()))
        task = self.tasks[job['task_id']]
        minimum = self._zero_wait_completion_ns(job['kind'], gpu=job.get('affinity'))
        if minimum <= int(task['deadline_abs_ns'])-now:
            return False
        for queue in self.ready.values():
            if job in queue:
                queue.remove(job)
        for card in self.gpus:
            if job in card['pinned']:
                card['pinned'].remove(job)
        reason = 'insufficient inference time before GPU start'
        event_ns = self.emit('REMOVE', task_id=job['task_id'], job_id=job['job_id'],
            kind=job['kind'], reason=reason, root_id=task.get('root_id'),
            component_id=self._job_component(job))
        self._observer_remove(job, 'resource', event_ns)
        if job['kind'] == 'FAST_SUFFIX':
            gpu = job.get('affinity')
            if job['task_id'] in self.prefix_card:
                self._release_prefix_commitment(job['task_id'], self.prefix_card[job['task_id']], reason)
            elif gpu is not None:
                self.gpus[gpu]['pending'].discard(job['task_id'])
        self.counters['prestart_insufficient_inference_budget'] += 1
        self.emit('START_DEADLINE_REJECT', task_id=job['task_id'], kind=job['kind'],
            component_id=self._job_component(job), remaining_ns=int(task['deadline_abs_ns'])-now,
            zero_wait_ns=int(minimum) if math.isfinite(minimum) else None,
            queue_prediction_used=False)
        self.finish(job['task_id'], 'Abstain', None, reason=reason,
                    component_id=self._job_component(job))
        return True

    def dispatch(self, j, resource, now):
        j = dict(j, dispatch_ns=now, resource=resource)
        g = int(resource.split(':')[1])
        s = self.gpus[g]
        s['busy'] = True
        s['job'] = j
        s['busy_since'] = now
        # the dispatch deadline budget is the GATE (worst-case start), never the
        # mean prediction charge used inside forecast (T012 s3.4)
        s['budget_finish_ns'] = now + self.gate_ns[g][j['kind']] + self.gpu_ipc_budget[g]
        if j.get('inline_fast_continuation'):
            s['budget_finish_ns'] += self.gate_ns[g]['FAST_SUFFIX']
            j.update(inline_suffix_job_id=self.new_job_id(),
                inline_suffix_gate_ns=int(self.gate_ns[g]['FAST_SUFFIX']),
                inline_return_budget_ns=int(self.delivery_budget()) + self.network_return_ns,
                deadline_abs_ns=self.t0_ns + int(self.tasks[j['task_id']]['deadline_abs_ns']),
                quality_need_enabled=bool(self.spec.get('quality_need_enabled', False)))
        if j['kind'] == 'FAST_PREFIX':
            if not j.get('inline_fast_continuation'):
                s['pending'].add(j['task_id'])
                self.prefix_card[j['task_id']] = g
            if self.async_prefix_pipeline:
                j['decision_job_id'] = self.new_job_id()
                j['async_prefix_pipeline'] = True
        comp = j.get('component_id')
        if comp and comp in self.components:
            self.components[comp]['state'] = 'dispatched'
            self.components[comp]['start_ns'] = int(now)
        self.commands[resource].put(j)
        event_ns = self.emit('DISPATCH', **j)
        if self.controller is not None:
            self.controller.observer.component_dispatch(self._job_component(j), event_ns, card=g)
        return j

    def schedule(self, now):
        from quota_forecast_v8 import dispatch_choice, debt_dispatch
        measurement=profiler(self)
        for g in CARD_IDS:
            s = self.gpus[g]
            cycle = now // PERIOD_NS
            if s['cycle'] != cycle:
                s['cycle'] = cycle
                s['spent'] = Counter(s['cycle_charges'].get(cycle, {}))
            interior_change = ('_service_debt' in s and
                               0.0 < s['effective'] < 1.0 and
                               0.0 < s['target'] < 1.0)
            if (s['transition'] and not s['busy'] and
                    (interior_change or (not s['pending'] and not s['pinned']))):
                s['effective'] = s['target']
                s['transition'] = False
                self.emit('TARGET_EFFECTIVE', gpu=g, share=s['target'])
        self._expire_ready(now)
        for _ in range(200):
            with measurement.span('ours_choice'):
                g, j, role = dispatch_choice(self.gpus, self.ready['Fast'], self.ready['Accurate'],
                                         self.tasks, self.mean_ns, self.gpu_ipc_budget,
                                         gates=self.gate_ns, now=now,
                                         rescue_margin=self.rescue_margin,
                                         network_return_ns=self.network_return_ns)
            if j is None:
                break
            if self._drop_if_inference_budget_exhausted(j, now):
                continue
            with measurement.span('ours_book'):
                debt_dispatch(self.gpus[g], j, role, self.mean_ns[g],
                          self.ready['Fast'], self.ready['Accurate'],
                          thin_override=(dict(tasks=self.tasks, ipc=self.gpu_ipc_budget[g],
                                              now=now, network_return_ns=self.network_return_ns,
                                              margin=self.rescue_margin)
                                         if self.rescue_margin > 0 else None))
            for lst in (self.ready['Fast'], self.ready['Accurate']):
                if j in lst:
                    lst.remove(j)
                    break
            # Pinned is a waiting queue, just like the two shared ready queues.
            # Once dispatched, only the pending activation commitment remains;
            # an expiry sweep must not remove or terminalize this in-flight job.
            if j in self.gpus[g]['pinned']:
                self.gpus[g]['pinned'].remove(j)
            self.dispatch(j, 'gpu:%d' % g, now)

    # -- results ----------------------------------------------------------
    def _drain_results(self):
        got = 0
        deadline = (time.perf_counter_ns() + SOURCE_SLICE_NS
                    if self.fair_input_processing else None)
        while got < RESULT_BATCH_MAX:
            try:
                r = self.results.get_nowait()
            except Exception:
                break
            got += 1
            if r.get('type') == 'ERROR':
                raise RuntimeError(r)
            if r.get('type') == 'DECISION_DONE':
                self._on_decision(r)
                if deadline is not None and time.perf_counter_ns() >= deadline:
                    self.counters['result_slice_yields'] += 1
                    return
                continue
            if r.get('type') != 'DONE':
                raise RuntimeError('unexpected actor result %r' % (r,))
            self._on_done(r)
            if deadline is not None and time.perf_counter_ns() >= deadline:
                self.counters['result_slice_yields'] += 1
                return
        for q in self.continuation_queues.values():
            for _ in range(RESULT_BATCH_MAX):
                try:
                    r = q.get_nowait()
                except queue.Empty:
                    break
                self._on_decision(r)
                if deadline is not None and time.perf_counter_ns() >= deadline:
                    self.counters['result_slice_yields'] += 1
                    return

    def _on_decision(self, r):
        j = r['job']
        i = j['task_id']
        if j.get('async_prefix_pipeline') and i in self.pipeline_completed_heads:
            raise RuntimeError('duplicate completed pipeline decision')
        if j.get('async_prefix_pipeline') and i not in self.pipeline_prefix_seen:
            # Independent producer queues do not guarantee inter-process order.
            if i in self.pipeline_early_decisions:
                raise RuntimeError('duplicate early pipeline decision')
            self.pipeline_early_decisions[i] = r
            return
        r = dict(r)
        r['start_ns'] = int(r['start_ns']) - self.t0_ns
        r['finish_ns'] = int(r['finish_ns']) - self.t0_ns
        if j.get('async_prefix_pipeline'):
            j = dict(j)
            for key in ('enqueue_ns', 'dispatch_ns', 'vector_available_ns'):
                j[key] = int(j[key]) - self.t0_ns
            r['job'] = j
        self.head_pending.discard(i)
        self.emit('FAST_DECISION', task_id=i, job_id=j['job_id'], kind='FAST_DECISION',
                  resource='decision:0', start_ns=r['start_ns'], finish_ns=r['finish_ns'],
                  budget_ns=self.decision_budget, enqueue_ns=j['enqueue_ns'],
                  dispatch_ns=j['dispatch_ns'], gpu=j['gpu'],
                  vector_available_ns=j.get('vector_available_ns'))
        self.component(dict(r, task_id=i, component='FAST_DECISION'))
        if j.get('async_prefix_pipeline'):
            self.pipeline_prefix_seen.discard(i)
            self.pipeline_completed_heads.add(i)
            self.pipeline_head_samples.append(r['finish_ns'] - r['start_ns'])
            self.counters['pipeline_heads_completed'] += 1
            if self.counters['pipeline_heads_completed'] % 16 == 0:
                self.decision_budget = int(sorted(self.pipeline_head_samples)[
                    max(0, int(.95 * len(self.pipeline_head_samples)) - 1)])
            self.emit('PIPELINE_CONTINUATION', task_id=i, route_role=r.get('route_role'),
                      vector_available_ns=j['vector_available_ns'],
                      head_start_ns=r['start_ns'], head_finish_ns=r['finish_ns'],
                      received_ns=self.now(), gpu=j['gpu'])
            if i in self.pipeline_released:
                self.pipeline_released.discard(i)
                self.deferred_cleanup.discard(i)
                return
            from quality_need_v24 import quality_order
            only = self.spec.get('edge_model_only')
            expected = quality_order(r['probabilities'], 'B', r['accurate_need_probability'],
                                     bool(self.spec.get('quality_need_enabled', False)),
                                     allowed_roles=[only] if only else None)
            if r.get('route_role') != (expected[0] if expected else 'Abstain'):
                raise RuntimeError('CPU continuation queue disagrees with quality routing')
        self._route_b(i, r['probabilities'], j['gpu'], r['accurate_need_probability'])

    def _on_done(self, r):
        j = r['job']
        i = j['task_id']
        rid = r['resource']
        r = dict(r)
        r['start_ns'] = int(r['start_ns']) - self.t0_ns
        r['finish_ns'] = int(r['finish_ns']) - self.t0_ns
        if j.get('inline_fast_continuation'):
            for key in ('prefix_finish_ns', 'prefix_compute_end_ns', 'suffix_start_ns',
                        'suffix_compute_start_ns', 'suffix_compute_end_ns', 'suffix_finish_ns'):
                if key in r:
                    r[key] = int(r[key]) - self.t0_ns
        if not rid.startswith('gpu'):
            raise RuntimeError('group runtime got a non-GPU result %r' % (r,))
        g = int(rid.split(':')[1])
        s = self.gpus[g]
        s['busy'] = False
        s['job'] = None
        if '_service_debt' in s:
            from quota_forecast_v8 import debt_complete
            actual_ns = max(0, int(r['finish_ns']) - int(r['start_ns']))
            settlement = debt_complete(s, j, actual_ns)
            self.emit('SERVICE_DEBT_SETTLE', gpu=g, job_id=j['job_id'],
                      task_id=i, kind=j['kind'], actual_service_ns=actual_ns,
                      balance_ns=dict(s['_service_debt']['balance']),
                      effective_share=s['effective'], target_share=s['target'],
                      settlement=settlement)
        if s['busy_since'] is not None:
            s['busy_ns'] += r['finish_ns'] - s['busy_since']
            s['busy_since'] = None
        done_lag = self.now() - r['finish_ns']
        # T012 C5: every sample counts (bounded by the deque), the old
        # len<512 gate silently froze the max the report used.
        self.perf['done_lag_ns'].append(max(int(done_lag), 0))
        self.delivery_lag_ns.append(max(int(done_lag), 0))
        self._update_delivery_budget()
        r['compute_start_ns'] = int(r['compute_start_ns']) - self.t0_ns
        r['compute_end_ns'] = int(r['compute_end_ns']) - self.t0_ns
        if r.get('gpu_quality_head'):
            for key in ('head_start_ns', 'head_finish_ns'):
                r[key] = int(r[key]) - self.t0_ns
        role = 'Accurate' if j['kind'] == 'ACCURATE' else 'Fast'
        a, b = r['start_ns'], r['finish_ns']
        while a < b:
            cycle = a // PERIOD_NS
            e = min(b, (cycle + 1) * PERIOD_NS)
            ch = s['cycle_charges'].setdefault(cycle, Counter())
            ch[role] += e - a
            self.streams['quota'].write(
                {'gpu': g, 'role': role, 'cycle': cycle, 'start_ns': a, 'finish_ns': e,
                 'job_id': j['job_id'], 'group': self.name})
            a = e
        fresh_now = self.now()
        s['spent'] = Counter(s['cycle_charges'].get(fresh_now // PERIOD_NS, {}))
        comp = self._job_component(j)
        if self.controller is not None:
            # T012 C10: ONE authoritative book - the runtime, the controller and
            # the forecast all read/mutate self.cost_book (same object).
            stage_end = r.get('prefix_finish_ns', r['finish_ns'])
            self.cost_book.observe(g, j['kind'], stage_end - r['start_ns'],
                                   r['start_ns'] - j['dispatch_ns'], fresh_now)
            if r.get('inline_suffix_done'):
                self.cost_book.observe(g, 'FAST_SUFFIX', r['suffix_finish_ns']-r['suffix_start_ns'],
                                       0, fresh_now)
            self.controller.observer.component_finish(comp, fresh_now,
                                                      finish_ns=stage_end,
                                                      dispatch_ns=j['dispatch_ns'])
        if comp in self.components:
            self.components[comp]['state'] = 'finished'
            self.components[comp]['finish_ns'] = r.get('prefix_finish_ns', r['finish_ns'])
        if fresh_now - self.cost_refresh_ns >= 250_000_000:
            self._refresh_costs(fresh_now)
        component_record = dict(r, task_id=i, group=self.name, kind=j['kind'])
        if j.get('inline_fast_continuation'):
            component_record.update(finish_ns=r['prefix_finish_ns'],
                compute_end_ns=r['prefix_compute_end_ns'], cuda_ms=r['prefix_cuda_ms'],
                feature_or_result_transfer_ns=r['prefix_transfer_ns'],
                physical_bundle_finish_ns=r['finish_ns'], score_charge_ns=r['finish_ns']-r['start_ns'])
            component_record.pop('prediction', None)
        self.component(component_record)
        # T012 C3 four delivery endpoints on the COMPONENT_FINISH row: GPU math
        # end, actor report, and the instant the arm's parent could consume it.
        self.emit('COMPONENT_FINISH', task_id=i, component_id=comp, kind=j['kind'],
                  resource=rid, start_ns=r['start_ns'], finish_ns=component_record['finish_ns'],
                  service_ms=(component_record['finish_ns'] - r['start_ns']) / 1e6,
                  dispatch_to_start_ms=(r['start_ns'] - j['dispatch_ns']) / 1e6,
                  compute_end_ns=component_record['compute_end_ns'], actor_finish_ns=r['finish_ns'],
                  parent_received_ns=fresh_now, enqueue_ns=j.get('enqueue_ns'),
                  dispatch_ns=j['dispatch_ns'])
        task = self.tasks.get(i) or {}
        if r.get('inline_suffix_done'):
            self._record_inline_suffix(task, g, r)
        ended_root = i in self.ended
        if j['kind'] == 'FAST_PREFIX' and self.async_prefix_pipeline:
            self.pipeline_prefix_seen.add(i)
        if ended_root:
            # deferred cleanup (s3.2): the real GPU wall is charged above; the
            # cache is released; no second terminal; no resurrected Head
            self.counters['late_discarded_final_results'] += 1
            self.emit('LATE_DISCARDED_FINAL', task_id=i, kind=j['kind'], gpu=g,
                      finish_ns=r['finish_ns'])
            if j['kind'] == 'FAST_PREFIX':
                if not j.get('inline_fast_continuation'):
                    self.commands[rid].put({'kind': 'RELEASE', 'task_id': i})
                s['pending'].discard(i)
                if self.async_prefix_pipeline:
                    self.pipeline_released.add(i)
            elif j['kind'] == 'FAST_SUFFIX':
                s['pending'].discard(i)
                s['pinned'] = [p for p in s['pinned'] if p['job_id'] != j['job_id']]
            early = self.pipeline_early_decisions.pop(i, None)
            if early is not None:
                self._on_decision(early)
            return
        if j['kind'] == 'FAST_PREFIX':
            if self.gpu_quality_head:
                if not r.get('gpu_quality_head'):
                    raise RuntimeError('GPU Prefix completed without its quality decision')
                self.counters['gpu_heads_completed'] += 1
                self.emit('GPU_QUALITY_DECISION', task_id=i, gpu=g,
                          probabilities=r['probabilities'],
                          accurate_need_probability=r['accurate_need_probability'],
                          head_cuda_ms=r['head_cuda_ms'], accounted_in='FAST_PREFIX')
                self._route_b(i, r['probabilities'], g, r['accurate_need_probability'],
                              inline_result=r if j.get('inline_fast_continuation') else None)
                return
            # Prefix DONE ends the COMPONENT only: the Suffix reserve stays
            # until the Head decides (s3.2); dispatch the Head now
            now = fresh_now
            hj = r.get('pipeline_head_job') or {'job_id': self.new_job_id(), 'kind': 'FAST_DECISION', 'task_id': i,
                  'bank_index': j['bank_index'], 'gpu': g, 'vector': r['vector'],
                  'vector_available_ns': r['finish_ns'], 'enqueue_ns': now,
                  'dispatch_ns': now, 'resource': 'decision:0'}
            if self.async_prefix_pipeline:
                hj = dict(hj, resource='decision:0')
                for key in ('enqueue_ns', 'dispatch_ns', 'vector_available_ns'):
                    hj[key] = int(hj[key]) - self.t0_ns
            self.jobs.append(hj)
            self.head_pending.add(i)
            if not self.async_prefix_pipeline:
                self.commands['decision:0'].put(hj)
            self.emit('DISPATCH', **{k: v for k, v in hj.items() if k != 'vector'})
            early = self.pipeline_early_decisions.pop(i, None)
            if early is not None:
                self._on_decision(early)
        else:
            if j['kind'] == 'FAST_SUFFIX':
                s['pending'].discard(i)
                # T012: a completed pinned Suffix must leave the pinned list,
                # otherwise the expiry sweep re-removes a finished component
                s['pinned'] = [p for p in s['pinned'] if p['job_id'] != j['job_id']]
            # T012 C3: an in-flight job past its deadline still writes ONE
            # terminal (completion + charge kept); it is only marked late.
            self.finish(i, role, r.get('prediction'), terminal_finish_ns=r['finish_ns'],
                        component_id=comp, compute_end_ns=r['compute_end_ns'],
                        actor_finish_ns=r['finish_ns'], parent_received_ns=fresh_now,
                        terminal_available_ns=fresh_now)

    # -- controller view / apply ------------------------------------------
    def _head(self, j, now):
        task = self.tasks.get(j['task_id']) or {}
        return {'task_id': j['task_id'], 'kind': j['kind'], 'card': j.get('affinity'),
                'enqueue_ns': j.get('enqueue_ns', 0), 'arrival_ns': task.get('arrival_ns', 0),
                'deadline_ns': task.get('deadline_abs_ns', 0),
                'affinity': j.get('affinity'), 'rejected_request': False}

    def _view(self, now):
        from queue_feedback_controller import make_executor_view_qf4
        fast = [j for j in self.ready['Fast']
                if j['task_id'] not in self.ended
                and now <= (self.tasks.get(j['task_id']) or {}).get('deadline_abs_ns', 0)]
        acc = [j for j in self.ready['Accurate']
               if j['task_id'] not in self.ended
               and now <= (self.tasks.get(j['task_id']) or {}).get('deadline_abs_ns', 0)]
        fast.sort(key=lambda j: ((self.tasks.get(j['task_id']) or {}).get('arrival_ns', 0),
                                 j['task_id']))
        acc.sort(key=lambda j: (j.get('enqueue_ns', 0), j['task_id']))
        pinned = []
        for g in CARD_IDS:
            for j in self.gpus[g]['pinned']:
                task = self.tasks.get(j['task_id']) or {}
                if j['task_id'] in self.ended or now > task.get('deadline_abs_ns', 0):
                    continue
                pinned.append(j)
        heads = {'Fast': self._head(fast[0], now) if fast else None,
                 'Accurate': self._head(acc[0], now) if acc else None}
        cards = [{'card': g, 'target': self.gpus[g]['target'],
                  'effective': self.gpus[g]['effective'],
                  'transition': self.gpus[g]['transition'],
                  'busy': self.gpus[g]['busy'],
                  'pending': len(self.gpus[g]['pending']),
                  'pinned': len(self.gpus[g]['pinned']),
                  'spent_ms': {r: self.gpus[g]['spent'].get(r, 0) / 1e6
                               for r in ('Fast', 'Accurate')},
                  'service_debt_balance_ns': dict((self.gpus[g].get('_service_debt')
                      or {}).get('balance') or {})} for g in CARD_IDS]
        cost_records = self.cost_book.records_dict()
        obs = self.controller.observer if self.controller is not None else None
        lag = obs.latest_lag_p95() if obs is not None else None
        waiting = obs.live_counts() if obs is not None else {}
        snap = obs.snapshot() if obs is not None else {}
        idle_ns = {}
        horizon = 2 * WINDOW_NS
        for g in CARD_IDS:
            busy = self.gpus[g]['busy_ns']
            idle_ns[g] = max(horizon - (busy % horizon), 0.0) if busy else horizon
        view = make_executor_view_qf4(
            now, cards, heads, cost_records, self.deadline_ns,
            pinned=pinned, waiting=waiting,
            waiting_work_ms=snap.get('waiting_work_ms') or {},
            rejected_profiles=self.rejected_profiles,
            cost_version=self.cost_book.cost_version(),
            receipt_lag_p95_ns=(lag or {}).get('receipt_ns'),
            done_lag_p95_ns=(lag or {}).get('done_ns'),
            stall=bool((lag or {}).get('stall')),
            idle_ns=idle_ns, idle_horizon_ns=horizon,
            watermark_ns=self.watermark_ns,
            frame_gap_seen=self.frame_gap_seen,
            decision_budget_ns=self.decision_budget,
            gate_probe=self._gate_probe)
        # T012 s4: the executor view carries the frozen key set even where the
        # controller factory has no field for it yet.
        window_id = (int(now) // TICK_NS) * TICK_NS
        triggers = []
        if self.frame_gap_seen:
            triggers.append('frame_gap')
        log_backpressure = self._control_log_backpressure(now)
        if log_backpressure:
            triggers.append('log_backpressure')
        if (lag or {}).get('stall'):
            triggers.append('stall')
        if self.replay_worker_status not in ('result', 'started', 'requested', 'direct_demand'):
            triggers.append('replay_' + str(self.replay_worker_status))
        candidates = None
        if isinstance(self.replay_result, dict):
            candidates = self.replay_result.get('candidates') or self.replay_result.get('grid')
        view['snapshot_ns'] = int(now)
        view['window_id'] = window_id
        view['observer'] = snap
        view['demand'] = {'waiting': waiting,
                          'waiting_work_ms': snap.get('waiting_work_ms') or {},
                          'live': len(self.live), 'unfinished': len(self.tasks) - len(self.ended)}
        view['triggers'] = triggers
        view['candidates'] = candidates
        view['replay'] = self.replay_result
        if self.delivery_objective == 'latency_timeout':
            view['direct_demand'] = self._direct_demand_snapshot
            view['deadline_envelope'] = self._deadline_envelope
            view['deadline_role_envelope'] = self._deadline_envelope
            view['direct_budget_horizon_ns'] = self.deadline_ns
            view['direct_inline_fast_continuation'] = self.inline_fast_continuation
            # Use the queue heads already collected for this view. Pinned
            # Suffix deadlines come from the same public task records.
            timed_heads = [h for h in heads.values() if h]
            if pinned:
                earliest = min(pinned, key=lambda j: self.tasks[j['task_id']]['deadline_abs_ns'])
                timed_heads.append(self._head(earliest, now))
            slacks = []
            role_slacks = {'Fast': [], 'Accurate': []}
            for head in timed_heads:
                affinity = head.get('affinity')
                cards_for_head = CARD_IDS if affinity is None else (int(affinity),)
                service = min(float(self.gate_ns[g][head['kind']]) for g in cards_for_head)
                slacks.append(int(head['deadline_ns'])-int(now)-int(service)-int(self.delivery_budget()))
                role = 'Accurate' if head['kind'] == 'ACCURATE' else 'Fast'
                role_slacks[role].append(slacks[-1])
            view['direct_min_queue_slack_ns'] = min(slacks) if slacks else None
            view['direct_role_slack_ns'] = {r: min(s) if s else None for r,s in role_slacks.items()}
        view['log_backpressure'] = bool(log_backpressure)
        view['log_backpressure_windows'] = sorted(self.log_backpressure_windows)[-8:]
        view['source_end'] = self.source_end
        view['duration_ns'] = int(self.duration_s * NS_PER_S)
        view['delivery_budget_ns'] = int(self.delivery_budget())
        view['delivery_budget_source'] = self.delivery_budget_source
        view['resource_accounting'] = self.resource_accounting
        if self.resource_accounting == 'service_debt':
            view['live_candidate_guard'] = self._live_candidate_guard
            if getattr(self, 'net_gain_history', None) is not None:
                view['net_gain_evaluate'] = self._net_gain_evaluate
        return view, pinned

    def _deadline_envelope(self, role=None):
        """Observe public ready work without changing FIFO or admission."""
        from deadline_control import envelope
        now = self.now()
        dc = getattr(self.controller,'dual_clock',None)
        analytic = bool(dc is not None and dc.mode=='FULL' and dc.c.get('independent_checks_enabled'))
        jobs = [j for r in ('Fast', 'Accurate') if role is None or r==role for j in self.ready[r]]
        blocked = None
        if analytic:
            blocked=[]
            for g,s in enumerate(self.gpus):
                reserved=max(0,int(s.get('budget_finish_ns',now))-now) if s['busy'] else 0
                pinned_ids={j['task_id'] for j in s['pinned'] if j['task_id'] not in self.ended}
                committed=pinned_ids | {i for i in s['pending'] if i not in self.ended}
                if s['busy'] and s['job'] and s['job']['kind']=='FAST_SUFFIX':
                    committed.discard(s['job']['task_id'])
                reserved+=len(committed)*(int(self.mean_ns[g]['FAST_SUFFIX'])+int(self.gpu_ipc_budget[g]))
                if committed:reserved+=int(self.decision_budget)
                blocked.append(reserved)
        else:
            jobs += [j for s in self.gpus for j in s['pinned']
                     if role is None or role=='Fast']
        public = []
        for job in jobs:
            if job['task_id'] in self.ended:
                continue
            kind = job['kind']
            affinity = job.get('affinity') if kind == 'FAST_SUFFIX' else None
            devices = CARD_IDS if affinity is None else (int(affinity),)
            # Unrevealed Prefix retains its public F continuation contract.
            service = min(int(self.mean_ns[g][kind])+int(self.gpu_ipc_budget[g])+
                (int(self.mean_ns[g]['FAST_SUFFIX']) if kind == 'FAST_PREFIX'
                 and self.inline_fast_continuation else 0) for g in devices)
            public.append(dict(role='Accurate' if kind == 'ACCURATE' else 'Fast',
                deadline_ns=int(self.tasks[job['task_id']]['deadline_abs_ns']),
                service_ns=service))
        return envelope(now, public, int(self.network_return_ns)+int(self.delivery_budget()),
                        blocked_ns=blocked)

    def _net_gain_evaluate(self, target):
        """Only sanitized currently public state enters the two-ratio model."""
        from net_gain_control import evaluate, clean_job, Unknown
        now = self.now()
        started = time.process_time_ns()
        dc = self.controller.dual_clock
        try:
            windows = self.net_gain_history.windows(now)
            ready = {r:[clean_job(j) for j in self.ready[r] if j['task_id'] not in self.ended]
                     for r in ('Fast','Accurate')}
            cards = []
            for s in self.gpus:
                from quota_forecast_v8 import debt_copy
                cards.append(dict(target=s['target'],effective=s['effective'],transition=s['transition'],
                    busy=s['busy'],job=clean_job(s['job']) if s['job'] else None,
                    budget_finish_ns=s.get('budget_finish_ns',now),
                    pending=set(s['pending']),pinned=[clean_job(j) for j in s['pinned']],
                    spent=Counter(s['spent']),_service_debt=debt_copy(s)))
            ids={j['task_id'] for v in ready.values() for j in v}
            ids.update(s['job']['task_id'] for s in cards if s['busy'] and s['job'])
            ids.update(i for s in cards for i in s['pending'])
            ids.update(j['task_id'] for s in cards for j in s['pinned'])
            tasks={i:dict(arrival_ns=int(self.tasks[i]['arrival_ns']),
                deadline_ns=int(self.tasks[i]['deadline_abs_ns'])) for i in ids}
            routes={j['task_id']:('Accurate' if j['kind']=='ACCURATE' else 'Fast')
                    for v in ready.values() for j in v if j['kind']!='FAST_PREFIX'}
            routes.update({s['job']['task_id']:('Accurate' if s['job']['kind']=='ACCURATE' else 'Fast')
                           for s in cards if s['busy'] and s['job'] and s['job']['kind']!='FAST_PREFIX'})
            snapshot=dict(now=now,cards=cards,ready=ready,tasks=tasks,routes=routes,
                costs=self.mean_ns,gates=self.gate_ns,ipc=list(self.gpu_ipc_budget),
                terminal_return=int(self.network_return_ns)+int(self.delivery_budget()),
                physical_return=int(self.network_return_ns), head_ns=int(self.decision_budget),
                model_only=self.spec.get('edge_model_only'),
                rescue_margin=self.rescue_margin,
                capacity=getattr(self,'edge_capacity',None))
            result=evaluate(snapshot,windows,self._share_fast(),target,
                maximum=dc.c['net_gain_max_roots'],max_events=dc.c['net_gain_max_events'])
        except Unknown as exc:
            result=dict(valid=False,accept=False,reason=str(exc))
        result['cpu_ns']=time.process_time_ns()-started
        result['snapshot_ns']=now
        self.emit('NET_GAIN_EVALUATION',**result)
        return result

    def _live_candidate_guard(self, target):
        """Bounded current-state check with the existing dispatch calendar.

        Check every known ready terminal, including queue tails, up to four
        jobs (eight 0.25ms forecasts). A larger state is explicitly UNKNOWN.
        Running jobs and pending Head promises enter each calendar; unknown
        Prefix outcomes and future arrivals are not promised individual gains.
        """
        from quota_forecast_v8 import forecast, debt_copy, LAST_FORECAST
        now = self.now()
        jobs = []
        for role in ('Fast', 'Accurate'):
            choices = [j for j in self.ready[role]
                       if j['task_id'] not in self.ended and not self._is_expired(j, now)]
            if role == 'Fast':
                choices += [j for s in self.gpus for j in s['pinned']
                            if j['task_id'] not in self.ended and not self._is_expired(j, now)]
            if choices:
                choices.sort(key=lambda j: (
                    self.tasks[j['task_id']]['arrival_ns'] if role == 'Fast'
                    else j['enqueue_ns'], j['task_id']))
                jobs.extend((role, job) for job in choices if job['kind'] != 'FAST_PREFIX')
        result = {'ok': True, 'reason': 'current_ready_no_predicted_role_loss',
                  'snapshot_ns': now, 'heads': [], 'scope': 'known_ready_terminals',
                  'ready_terminal_count': len(jobs), 'max_checked_jobs': 4,
                  'busy': sum(bool(s['busy']) for s in self.gpus),
                  'pending': sum(len(s['pending']) for s in self.gpus),
                  'pinned': sum(len(s['pinned']) for s in self.gpus)}
        if len(jobs) > 4:
            return dict(result, ok=False, reason='live_calendar_population_unknown')
        scores = [0., 0.]
        early_scores = [0., 0.]
        y = [0, 0]
        resource_u = [0., 0.]
        from resource_objective import timely_utility
        known = True
        role_results = {r: {'Y': [0, 0], 'V': [0., 0.], 'H': [0., 0.], 'known': True}
                        for r in ('Fast', 'Accurate')}
        for role, job in jobs:
            tid = job['task_id']
            task = self.tasks[tid]
            quality = (task.get('probabilities') or {}).get(role)
            known = known and isinstance(quality, (int, float)) and not isinstance(quality, bool)
            role_results[role]['known'] &= isinstance(quality, (int, float)) and not isinstance(quality, bool)
            detail = {'task_id': tid, 'role': role, 'deadline_ns': task['deadline_abs_ns']}
            for index, gF in enumerate((self._share_fast(), target)):
                cards = [dict(s, pinned=[dict(j) for j in s['pinned'] if j is not job],
                              pending=set(s['pending']), _service_debt=debt_copy(s))
                         for s in self.gpus]
                if index:
                    for g, share in enumerate(self.targets(gF)):
                        cards[g]['target'] = share
                        cards[g]['transition'] = abs(cards[g]['effective']-share) > 1e-12
                ready = {r: [dict(j) for j in self.ready[r] if j is not job]
                         for r in ('Fast', 'Accurate')}
                started = time.perf_counter_ns()
                eta = forecast(cards, ready, self.tasks, self.mean_ns,
                    self.gpu_ipc_budget, now, job['kind'], job.get('affinity'), tid,
                    decision_ns=self.decision_budget*(1+len(self.head_pending)),
                    horizon_ns=max(1, int(task['deadline_abs_ns'])-now),
                    gates=self.gate_ns, wall_budget_ns=self.forecast_budget_ns,
                    candidate_job=job)
                wall = time.perf_counter_ns()-started
                if eta is None or wall > self.forecast_budget_ns:
                    return dict(result, ok=False, reason='live_calendar_unknown',
                                calendar=dict(LAST_FORECAST), wall_ns=wall)
                timely = now+eta <= task['deadline_abs_ns']
                detail['current_ontime' if index == 0 else 'target_ontime'] = timely
                detail['current_eta_ns' if index == 0 else 'target_eta_ns'] = eta if math.isfinite(eta) else None
                y[index] += int(timely)
                role_results[role]['Y'][index] += int(timely)
                if timely:
                    resource_u[index] += timely_utility(task['arrival_ns'],
                        task['deadline_abs_ns'], now + eta)
                if timely and isinstance(quality, (int, float)) and not isinstance(quality, bool):
                    scores[index] += float(quality)
                    early = float(quality) * max(0.,
                        (task['deadline_abs_ns']-now-eta) /
                        max(1, task['deadline_abs_ns']-task['arrival_ns']))
                    early_scores[index] += early
                    role_results[role]['V'][index] += float(quality)
                    role_results[role]['H'][index] += early
            result['heads'].append(detail)
        result.update(Y_delta=y[1]-y[0], V_delta=scores[1]-scores[0] if known else None,
                      H_delta=early_scores[1]-early_scores[0] if known else None)
        result['role_deltas'] = {r: {k: v[k][1]-v[k][0] if k == 'Y' or v['known'] else None
                                      for k in ('Y', 'V', 'H')} for r, v in role_results.items()}
        protected_metrics = ('V',) if getattr(self, 'delivery_objective', None) == 'correct_ontime' else ('V', 'H')
        result['selection_objective'] = getattr(self, 'delivery_objective', 'quality_weighted_earliness')
        if self.delivery_objective == 'latency_timeout':
            result['resource_utility_delta'] = resource_u[1]-resource_u[0]
            if any(v['Y'] < 0 for v in result['role_deltas'].values()):
                result.update(ok=False, reason='current_ready_predicted_timeout_increase')
            return result
        if any(v['Y'] < 0 or any(v[k] is not None and v[k] < -1e-12 for k in protected_metrics)
               for v in result['role_deltas'].values()):
            result.update(ok=False, reason='current_ready_predicted_role_loss')
        return result

    def _checked_ready_job(self, cid):
        """Find the live ready/pinned entry carrying this component_id."""
        if not cid:
            return None
        for j in list(self.ready['Fast']) + list(self.ready['Accurate']):
            if str(j.get('component_id')) == str(cid):
                return j
        for g in CARD_IDS:
            for j in self.gpus[g]['pinned']:
                if str(j.get('component_id')) == str(cid):
                    return j
        return None

    def _checked_job(self, checked, receiver, next_cycle, now):
        """T012 C9: resolve WHICH job the step is meant to serve.

        The probe must confirm a new start opportunity for this exact job, so the
        checked job is resolved first and its component_id is carried through.
        """
        spec = dict(checked) if isinstance(checked, dict) else None
        cid = None
        if spec is not None:
            cid = spec.get('component_id') or spec.get('comp_id')
        elif isinstance(checked, str) and checked:
            cid = checked
        role = (spec or {}).get('role')
        if role not in ('Fast', 'Accurate'):
            role = receiver if receiver in ('Fast', 'Accurate') else None
        basis = 'checked_component'
        job = self._checked_ready_job(cid)
        if job is None and cid and cid in self.components:
            rec = self.components[cid]
            kind = rec.get('kind')
            if kind:
                dl = (self.tasks.get(rec.get('task_id')) or {}).get('deadline_abs_ns')
                job = {'job_id': self.new_job_id(), 'task_id': rec.get('task_id'),
                       'kind': kind, 'bank_index': 0, 'enqueue_ns': next_cycle,
                       'affinity': rec.get('affinity'), 'p_selected': None,
                       'component_id': cid,
                       'deadline_ns': None if dl is None else int(dl)}
                role = role or ('Accurate' if kind == 'ACCURATE' else 'Fast')
        if job is None and role is not None:
            # TASK 3.4/6: an owed Suffix keeps its saved activation and
            # select_job prefers the pinned ledger over the ready queue, so the
            # pinned head IS the receiver's first real start opportunity.  An
            # all-expired ledger is still returned so the probe can report the
            # explicit expiry reason instead of a generic "no job".
            pin = [j for g in CARD_IDS for j in self.gpus[g]['pinned']
                   if j['task_id'] not in self.ended]
            fresh = [j for j in pin if not self._is_expired(j, now)]
            pin = fresh or pin
            if pin:
                job = min(pin, key=lambda j: (int(j.get('enqueue_ns') or 0),
                                              j['task_id']))
                basis = 'pinned_head'
        if job is None and role is not None:
            cands = [j for j in self.ready[role]
                     if j['task_id'] not in self.ended]
            fresh = [j for j in cands if not self._is_expired(j, now)]
            cands = fresh or cands
            job = cands[0] if cands else None
            basis = 'ordinary_head'
        if job is None and role is not None:
            profile = self.rejected_profiles.get('%s:%s' % (
                role, 'ACCURATE' if role == 'Accurate' else 'FAST_FULL'))
            if profile is not None:
                ev = profile.get('evidence') or {}
                dl = ev.get('deadline_ns')
                job = {'job_id': self.new_job_id(), 'task_id': profile['task_id'],
                       'kind': profile['kind'], 'bank_index': 0,
                       'enqueue_ns': next_cycle, 'affinity': profile.get('affinity'),
                       'p_selected': None, 'component_id': None,
                       'deadline_ns': None if dl is None else int(dl)}
                basis = 'rejected_profile'
        if job is None and role is None:
            for fallback in ('Fast', 'Accurate'):
                cands = [j for j in self.ready[fallback]
                         if j['task_id'] not in self.ended and
                         not self._is_expired(j, now)]
                if cands:
                    job, role, basis = cands[0], fallback, 'ordinary_head'
                    break
        return job, role, basis

    def _is_expired(self, j, now):
        d = j.get('deadline_ns')
        return d is not None and int(d) <= int(now)

    def _gate_probe(self, target_gF, current_gF, checked=None, receiver=None):
        """REAL select_job verification of the CHECKED job itself (T012 C9).

        TASK section 6: the probe must show that the job this step is meant to
        serve really gets a NEW start opportunity.  The cards keep their real
        busy state, the pinned ledger is restricted to the checked component, and
        ``ok`` is True only when the started job IS the checked component.
        """
        from quota_forecast_v8 import select_job, debt_copy
        from copy import deepcopy
        now = self.now()
        next_cycle = ((now // PERIOD_NS) + 1) * PERIOD_NS
        results = {'ok': False, 'basis': None, 'detail': {},
                   'checked_comp_id': None, 'started_comp_id': None,
                   'checked_task_id': None, 'checked_kind': None,
                   'pinned_start_ns': None, 'age_ns': None}
        job, role, basis = self._checked_job(checked, receiver, next_cycle, now)
        if job is None:
            results['basis'] = basis
            results['reason'] = 'no_checked_job'
            return results
        if job.get('component_id') is None:
            self._job_component(job)
        cid = str(job.get('component_id'))
        results['checked_comp_id'] = cid
        results['checked_task_id'] = job.get('task_id')
        results['checked_kind'] = job.get('kind')
        results['basis'] = basis
        rec = self.components.get(cid) or {}
        start_ns = rec.get('start_ns') or job.get('dispatch_ns') or job.get('enqueue_ns')
        if start_ns is not None:
            results['pinned_start_ns'] = int(start_ns)
            results['age_ns'] = max(0, int(now) - int(start_ns))
        # T012 C9: an already-expired checked job cannot start, so it is not
        # evidence of a new opportunity.
        if self._is_expired(job, now):
            results['reason'] = 'checked_job_expired'
            return results
        if role is None:
            role = 'Accurate' if job.get('kind') == 'ACCURATE' else 'Fast'
        shares = self.targets(float(target_gF))
        fast_q = [job] if role == 'Fast' else []
        acc_q = [job] if role == 'Accurate' else []
        fake_tasks = {job['task_id']: {
            'arrival_ns': (self.tasks.get(job['task_id']) or {}).get(
                'arrival_ns', next_cycle)}}
        started_comp = None
        for g in CARD_IDS:
            src = self.gpus[g]
            if bool(src.get('busy')):
                results['detail']['card%d' % g] = {
                    'ok': False, 'reason': 'card_busy', 'kind': None,
                    'started_comp_id': None}
                continue
            # the pinned ledger is restricted to the checked component: another
            # role's in-flight pinned start must never be read as evidence
            pinned = [deepcopy(p) for p in src['pinned']
                      if str(p.get('component_id')) == cid]
            state = {'busy': False, 'job': None, 'pinned': pinned,
                     'pending': set(src['pending']),
                     'target': float(shares[g]), 'effective': float(shares[g]),
                     'transition': bool(src.get('transition')),
                     'spent': Counter(), 'cycle_charges': {},
                     'cycle': next_cycle // PERIOD_NS, 'budget_finish_ns': 0}
            if '_service_debt' in src:
                state['_service_debt'] = debt_copy(src)
                # Interior share changes retain their same-card commitments.
                changing = abs(src['effective'] - shares[g]) > 1e-12
                interior = (0.0 < src['effective'] < 1.0 and
                            0.0 < shares[g] < 1.0)
                state['transition'] = (changing and not interior and
                                       bool(src['pending'] or src['pinned']))
            try:
                j2, _r2 = select_job(state, fast_q, acc_q, fake_tasks,
                                     self.mean_ns[g], gates=self.gate_ns[g],
                                     now=next_cycle, ipc=self.gpu_ipc_budget[g])
            except Exception as exc:            # noqa: BLE001 - record the reason
                results['detail']['card%d' % g] = {
                    'ok': False, 'reason': 'select_job_error:%r' % (exc,),
                    'kind': None, 'started_comp_id': None}
                continue
            got = None if j2 is None else str(j2.get('component_id'))
            ok = got is not None and got == cid
            results['detail']['card%d' % g] = {
                'ok': bool(ok), 'reason': 'checked_job_started' if ok
                else 'checked_job_not_selected', 'kind': None if j2 is None
                else j2.get('kind'), 'started_comp_id': got}
            if ok and started_comp is None:
                started_comp = got
        results['started_comp_id'] = started_comp
        results['ok'] = started_comp is not None
        if results['ok']:
            results['reason'] = 'checked_job_started'
        elif not any(v.get('ok') for v in results['detail'].values()):
            results['reason'] = 'no_new_start_opportunity'
        return results

    def _share_fast(self):
        return float(sum(self.gpus[g]['effective'] for g in CARD_IDS))

    def _apply(self, command):
        """The parent scheduler applies ONE command_id to BOTH card targets;
        effective is recorded by the scheduler, never faked by an actor ACK."""
        cid = command.get('command_id')
        target = float(command['target_fast'])
        now = self.now()
        if not (AXIS_MIN_FAST - 1e-9 <= target <= AXIS_MAX_FAST + 1e-9):
            self.apply_rejected += 1
            return {'accepted': False, 'command_id': cid, 'applied_ns': now,
                    'target_fast': self._share_fast(), 'reason': 'out_of_range'}
        old = [(self.gpus[g]['target'], self.gpus[g]['transition']) for g in CARD_IDS]
        try:
            new = self.targets(target)
            for g in CARD_IDS:
                self.gpus[g]['target'] = float(new[g])
                self.gpus[g]['transition'] = abs(self.gpus[g]['effective'] - float(new[g])) > 1e-12
        except Exception:
            for g in CARD_IDS:
                self.gpus[g]['target'], self.gpus[g]['transition'] = old[g]
            self.apply_rejected += 1
            return {'accepted': False, 'command_id': cid, 'applied_ns': now,
                    'target_fast': self._share_fast(), 'reason': 'apply_exception'}
        self.emit('CONFIG_REQUEST', target_fast=target, policy=self.policy,
                  command_id=cid, scheduled_s=now / NS_PER_S)
        self.apply_count += 1
        self.config_request_count += 1
        if cid is not None:
            self.seen_command_ids.add(cid)
        return {'accepted': True, 'command_id': cid, 'applied_ns': now,
                'target_fast': target}

    def _controller_tick(self, now):
        writer = self._log_writer
        if writer is not None:
            writer.control_active = True
        try:
            return self._controller_tick_core(now)
        finally:
            if writer is not None:
                writer.control_active = False

    def _controller_tick_core(self, now):
        """Light 100 ms control tick: the aligned value only labels the
        window_id; every view/decision call takes the fresh monotonic now.
        The T012 C5 timing covers the WHOLE synchronous path - view build,
        scoring/apply and the controller.jsonl serialisation into the queue."""
        aligned = (now // TICK_NS) * TICK_NS
        if aligned <= self.last_tick_aligned:
            return
        self.last_tick_aligned = aligned
        t_sync = time.perf_counter_ns()
        cpu_start_ns = time.process_time_ns()
        now_ns = self.now()   # FRESH monotonic clock for the view/decision
        score_before = self.controller.last_scored_window
        command_before = self.controller.command_seq
        self._refresh_costs(now_ns)
        view, pinned = self._view(now_ns)
        dc = getattr(self.controller, 'dual_clock', None)
        if dc is not None and not dc.disabled and dc.mode != 'HYSTERESIS':
            dc.safe_monitor(now_ns, view, self.controller.observer,
                       self.controller.effective_fast, self.controller.pending)
        decision_cpu_start_ns = time.process_time_ns()
        calc_before = self.controller.counters.get('direct_demand_calculations', 0)
        step = self.control_step(self.controller, now_ns, view, self._apply,
                                 full=(aligned % WINDOW_NS == 0))
        full_decision_cpu_ns = (time.process_time_ns()-decision_cpu_start_ns
            if self.controller.counters.get('direct_demand_calculations', 0)>calc_before else None)
        # T014 CPU_expected.probe_flag_or: a window that contained ANY probe
        # tick is flagged for the whole window (the observer OR-folds it).
        try:
            if self.controller.state == 'PROBE':
                self.controller.observer.mark_flags(probe=True)
        except Exception:
            self.counters['probe_flag_errors'] = int(
                self.counters.get('probe_flag_errors', 0)) + 1
        compute_ns = time.perf_counter_ns() - t_sync
        self.perf['tick_wall_ns'].append(compute_ns)
        self.perf['max_tick_wall_ns'] = max(self.perf['max_tick_wall_ns'], compute_ns)
        commands = list(step['commands'] or [])
        log_row = step.get('log') or {}
        ctl = self.controller
        # s4 keys: the frozen control_step adapter is the primary source, the
        # controller attributes and the Runtime's own apply ledger are the
        # fallbacks (they stay equal because the adapter delegates here).
        pending_command = step.get('pending_command')
        if pending_command is None:
            pending_command = log_row.get('pending_command')
        if pending_command is None:
            pending_command = getattr(ctl, 'pending_command', None)
        if isinstance(pending_command, dict):
            # keep the row compact and JSON-stable: the id, not the payload
            pending_command = pending_command.get('command_id')
        new_commands = []
        for c in commands:
            cid = c.get('command_id')
            # _apply has already registered accepted ids in seen_command_ids.
            # Logging needs its own cursor or every accepted command vanishes,
            # including the return commands issued during probe evaluation.
            if cid is None or cid not in self.logged_command_ids:
                new_commands.append(c)
                if cid is not None:
                    self.logged_command_ids.add(cid)
        new_command_count = step.get('new_command_count')
        if new_command_count is None:
            new_command_count = log_row.get('new_command_count')
        if new_command_count is None:
            new_command_count = len(new_commands)
        apply_count = log_row.get('apply_count')
        if apply_count is None:
            apply_count = getattr(ctl, 'apply_count', self.apply_count)
        config_request_count = log_row.get('config_request_count')
        if config_request_count is None:
            config_request_count = getattr(ctl, 'config_request_count',
                                           self.config_request_count)
        rec = {'schema': self.CONTROL_LOG_SCHEMA,
               'controller_version': self.controller_version,
               'implementation_revision': self.implementation_revision,
               'policy': self.policy, 'group': self.name,
               'tick_ns': int(now_ns), 'window_id': int(aligned),
               'compute_ns': int(compute_ns), 'decision': step['decision'],
               'controller_cpu_ns': time.process_time_ns()-cpu_start_ns,
               'dual_clock_fallback_reason': (getattr(self.controller,'dual_clock_fallback_reason',None)
                   or (dc.disabled_reason if dc is not None else None)),
               'direct_cheap_checked': bool(self.controller.direct_samples and self.controller.direct_samples[-1]['aligned_ns']==aligned),
               'full_decision_cpu_ns': full_decision_cpu_ns,
               'commands': new_commands, 'reason': log_row.get('reason'),
               'state': log_row.get('state'),
               'new_command_count': int(new_command_count),
               'pending_command': pending_command,
               'apply_count': int(apply_count),
               'config_request_count': int(config_request_count)}
        self.streams['controller'].write(rec)
        self.controller_rows += 1
        self.counters['commands'] += len(new_commands)
        self.cost_book.update_gates(now_ns)
        # A fresh result may score on a later 100ms tick, not just the 500ms
        # boundary. Do not combine that actual scoring/command work with the
        # next submission. The following tick retries against a fresh profile;
        # the original result age/epoch guards and per-window bound remain.
        scored = self.controller.last_scored_window != score_before
        issued = self.controller.command_seq != command_before
        if aligned % WINDOW_NS != 0 and not scored and not issued:
            self._maybe_replay(now_ns)
        self._poll_replay(now_ns)
        self._note_controller_sync(time.perf_counter_ns() - t_sync)

    # -- main loop --------------------------------------------------------
    def _drain_source_frames(self):
        """Public source consumption inside a bounded 0.5 ms wall slice or 32
        frames, whichever comes FIRST; a backlog is consumed in batches and
        never driven to the end in one go (s3.1).  The slot is timed in FULL:
        the old min(used, 4*slice) clamp hid the real 800 ms+ stalls."""
        slice_deadline = time.perf_counter_ns() + SOURCE_SLICE_NS
        batch = 0
        while batch < FRAME_BATCH_MAX:
            try:
                frame = self.frame_q.get_nowait()
            except Exception:
                break
            if frame.get('outcome') == 'SOURCE_END':
                self._handle_input_frame(frame, self.now())
                break
            self._handle_input_frame(frame, self.now())
            batch += 1
            if time.perf_counter_ns() >= slice_deadline:
                break
        used = time.perf_counter_ns() + SOURCE_SLICE_NS - slice_deadline
        self._note_source_slice(used)
        return batch

    def _tick_due(self, now):
        """ONE light control tick per 100 ms alignment (T014 s3.1.4)."""
        return (int(now) // TICK_NS) * TICK_NS > self.last_tick_aligned

    def run(self):
        self.open_streams()
        old_gc_thresholds = gc.get_threshold()
        # This finite source run retains its audit/task records. A full scan
        # of that growing heap stopped the complete-source CPU reproduction
        # for 759ms while collecting zero objects. Keep young-generation
        # cycle collection; defer the full scan until all deadlines/logs close.
        gc.set_threshold(old_gc_thresholds[0], old_gc_thresholds[1], 1_000_000)
        try:
            import resource
            rss_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except ImportError:
            resource = None
            rss_before_kib = None
        gc_started = {}
        gc_counts = Counter()
        gc_total_ns = Counter()
        gc_max_ns = Counter()
        gc_long = deque(maxlen=64)
        def note_gc(phase, info):
            generation = info['generation']
            if phase == 'start':
                gc_started[generation] = time.perf_counter_ns()
            else:
                elapsed = time.perf_counter_ns() - gc_started.pop(generation, time.perf_counter_ns())
                gc_counts[generation] += 1
                gc_total_ns[generation] += elapsed
                gc_max_ns[generation] = max(gc_max_ns[generation], elapsed)
                if elapsed >= 5_000_000:
                    gc_long.append({'generation': generation, 'at_ns': self.now(),
                                    'wall_ms': elapsed / 1e6,
                                    'collected': info.get('collected')})
        gc.callbacks.append(note_gc)
        old_switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(0.001)
        started = time.perf_counter_ns()
        duration_ns = int(self.duration_s * NS_PER_S)
        try:
            while True:
                now = self.now()
                if self.last_loop_ns is not None:
                    self._note_loop_gap(now - self.last_loop_ns)
                self.last_loop_ns = now
                if self.fair_input_processing:
                    t_input = time.perf_counter_ns()
                    self._drain_source_frames()
                    self._note_segment('source_receive', time.perf_counter_ns() - t_input)
                # -- segment 1: actor DONE read
                t0 = time.perf_counter_ns()
                if self.actor_mode == 'fake':
                    # fake actors are poll-driven; without this a controlled
                    # clock harness would never observe a DONE inside run().
                    self.pump_fake_actors()
                self._drain_results()
                self._drain_timed_events()
                self._note_segment('done_read', time.perf_counter_ns() - t0)
                # -- segment 2: expiry / source-side release / dispatch
                t0 = time.perf_counter_ns()
                with self.handler_timed('schedule'):
                    self.schedule(self.now())
                self._note_segment('expire_dispatch', time.perf_counter_ns() - t0)
                # -- segment 3: public source inside the bounded slice
                t0 = time.perf_counter_ns()
                if self.fair_input_processing:
                    self._drain_input_outcomes()
                else:
                    self._drain_source_frames()
                self._note_segment('input_admission' if self.fair_input_processing else 'source_slice',
                                   time.perf_counter_ns() - t0)
                # -- segment 4: observer cursors (incremental, never a full pass)
                t0 = time.perf_counter_ns()
                if self.controller is not None:
                    self.controller.observer.seal_source_cohorts(
                        self.watermark_ns, self.frame_gap_seen)
                    self.controller.observer.note_results_progress(self.now())
                self._note_segment('observer', time.perf_counter_ns() - t0)
                if self.source_end and now >= duration_ns and not self.live \
                        and not self.pending_input_outcomes \
                        and not self.head_pending and not self._busy() \
                        and not self.pipeline_prefix_seen and not self.pipeline_early_decisions \
                        and not self.timed_events \
                        and (self.controller is None or self.controller.pending is None):
                    break
                if (time.perf_counter_ns() - started) > (self.duration_s + 60) * NS_PER_S:
                    raise RuntimeError('group %s drain timeout, preserve evidence' % self.name)
                # -- segment 5: control tick; a full log queue PAUSES new control
                if self._log_paused():
                    self._note_log_pause()
                elif self._tick_due(now) and self.frame_q.empty() and not self.pending_input_outcomes:
                    t0 = time.perf_counter_ns()
                    self._controller_tick(self.now())
                    self._note_segment('controller', time.perf_counter_ns() - t0)
                # -- segment 6: log enqueue (never a disk write on this thread)
                t0 = time.perf_counter_ns()
                self._flush_logs()
                self._note_segment('log_enqueue', time.perf_counter_ns() - t0)
                if (not self.fair_input_processing or
                        (not self.pending_input_outcomes and self.frame_q.empty())):
                    time.sleep(0.00008)
            self.emit('RUN_END', group=self.name, ended=len(self.ended),
                      arrivals=len(self.seen_arrival))
            for q in self.commands.values():
                q.put({'kind': 'STOP'})
            self.pump_fake_actors()
        finally:
            gc.callbacks.remove(note_gc)
            self.gc_report = {'counts': dict(gc_counts),
                              'total_ms': {g: v / 1e6 for g, v in gc_total_ns.items()},
                              'max_ms': {g: v / 1e6 for g, v in gc_max_ns.items()},
                              'long_events': list(gc_long),
                              'full_collection_deferred': True,
                              'rss_before_kib': rss_before_kib,
                              'rss_peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                              if resource is not None else None}
            sys.setswitchinterval(old_switch_interval)
            try:
                self._stop_replay_worker()
                self._flush_logs(limit=None)
                self.close_streams()
            finally:
                gc.set_threshold(*old_gc_thresholds)
                if gc.isenabled():
                    cleanup_start = time.perf_counter_ns()
                    self.gc_report['post_run_collected'] = gc.collect(2)
                    self.gc_report['post_run_collection_ms'] = (
                        time.perf_counter_ns() - cleanup_start) / 1e6
        return self.summary()

    def stop_actors(self):
        for p in getattr(self, 'ps', []):
            p.join(timeout=30)
        alive = [p.pid for p in getattr(self, 'ps', []) if p.is_alive()]
        if alive:
            raise RuntimeError('group %s actors did not shut down: %r' % (self.name, alive))

    def _busy(self):
        if any(s['busy'] or s['pending'] or s['pinned'] for s in self.gpus):
            return True
        return bool(self.ready['Fast'] or self.ready['Accurate'])

    def summary(self):
        perf = {k: (list(v) if isinstance(v, deque) else v)
                for k, v in self.perf.items()}
        # T012 C5: P50/P95 plus the TRUE full-run max for every timed phase.
        perf_stats = {
            'handler': {name: _ms_stats(samples, self.max_handler_ns.get(name, 0))
                        for name, samples in sorted(self.handler_ns.items())},
            'source_slice': _ms_stats(self.source_slice_ns, self.max_source_slice_ns),
            'controller_sync': _ms_stats(self.controller_sync_ns, self.max_controller_sync_ns),
            'worker': _ms_stats(self.worker_ns, self.max_worker_ns),
            'main_loop_gap': _ms_stats(self.perf['main_loop_gap_ns'],
                                       self.perf['max_main_loop_gap_ns']),
            'tick_wall': _ms_stats(self.perf['tick_wall_ns'],
                                   self.perf['max_tick_wall_ns']),
            'forecast': _ms_stats(self.perf['forecast_ns'],
                                  self.perf['max_forecast_ns'])}
        return _jsonable({'group': self.name, 'policy': self.policy,
                'resource_accounting': self.resource_accounting,
                'allow_idle_endpoint_loans': self.allow_idle_endpoint_loans,
                'delivery_objective': self.delivery_objective,
                'service_debt_balance_ns': [dict((s.get('_service_debt') or {}).get(
                    'balance') or {}) for s in self.gpus],
                'service_debt': [service_debt_snapshot(s) for s in self.gpus],
                'gpu_uuids': list(self.gpu_uuids),
                't0_ns': self.t0_ns, 'deadline_ns': self.deadline_ns,
                'duration_s': self.duration_s,
                'arrivals': len(self.seen_arrival), 'ended': len(self.ended),
                'common_local_completed': self.common_local_completed,
                'controller_rows': self.controller_rows,
                'duplicate_frames': self.duplicate_frames,
                'source_duplicate_frames': int(
                    self.counters.get('source_duplicate_frame_ids', 0)),
                'source_frames_unsequenced': int(
                    self.counters.get('source_frames_unsequenced', 0)),
                'receipt_lag_max_ns': self.receipt_lag_max_ns,
                'watermark_ns': self.watermark_ns,
                'last_contig_frame_id': self.last_contig_frame_id,
                'frame_gap_seen': self.frame_gap_seen,
                'counters': {k: int(v) for k, v in sorted(self.counters.items())},
                'perf': perf,
                'handler_ms': {name: {'p50_ms': s['p50_ms'], 'p95_ms': s['p95_ms'],
                                      'max_ms': s['max_ms'], 'n': s['n']}
                               for name, s in perf_stats['handler'].items()},
                'source_slice_ms': perf_stats['source_slice'],
                'loop_gap_ms': perf_stats['main_loop_gap'],
                'controller_sync_ms': perf_stats['controller_sync'],
                'worker_ms': perf_stats['worker'],
                'tick_wall_ms': perf_stats['tick_wall'],
                'forecast_ms': perf_stats['forecast'],
                'forecast_budget_ns': int(self.forecast_budget_ns),
                'forecast_unknown': int(self.counters.get('forecast_unknown', 0)),
                'forecast_empty': int(self.counters.get('forecast_empty', 0)),
                'forecast_over_budget': int(self.counters.get('forecast_over_budget', 0)),
                'log_backpressure': bool(self.log_backpressure),
                'log_backpressure_events': int(self.log_backpressure_events),
                'log_backpressure_missing_rows': int(
                    getattr(self, 'log_backpressure_events_reported_missing', 0)),
                'log_dropped': int(self.log_dropped),
                'log_total': int(self.log_total),
                'log_backpressure_windows': sorted(self.log_backpressure_windows),
                'log_flushes': int(self.log_flushes),
                'log_writer': (self.log_writer_stats
                               if getattr(self, 'log_writer_stats', None) is not None
                               else (self._log_writer.stats() if self._log_writer else None)),
                'log_writer_joined': bool(getattr(self, 'log_writer_joined', False)),
                'log_pause_ticks': int(self.log_pause_ticks),
                'log_flush_batch': int(LOG_FLUSH_BATCH),
                'log_writer_yield_s': float(LOG_WRITER_YIELD_S),
                'gil_switch_interval_s': getattr(self, 'gil_switch_interval_s', None),
                'gc': getattr(self, 'gc_report', None),
                'loop_segments_ms': {name: _ms_stats(self.segments[name],
                                                     self.max_segment_ns.get(name, 0))
                                     for name in sorted(self.segments)
                                     if self.segments[name]},
                'replay': dict(self.replay_result) if isinstance(self.replay_result, dict)
                          else {'valid': False, 'invalid_reason': 'prediction_unknown'},
                'replay_worker_status': self.replay_worker_status,
                'replay_requests': int(self.replay_requests),
                'replay_errors': int(self.replay_errors),
                'delivery_budget_ns': int(self.delivery_budget()),
                'delivery_budget_source': self.delivery_budget_source,
                'apply': self.apply_stats(),
                'components': len(self.components),
                'rejected_components': int(self.rejected_components),
                'c_star_ms': getattr(self, 'c_star_ms', None),
                'final_cost_ns': {str(g): {k: float(v) for k, v in self.cost_ns[g].items()}
                                  for g in CARD_IDS},
                'final_gate_ns': {str(g): {k: float(v) for k, v in self.gate_ns[g].items()}
                                  for g in CARD_IDS},
                'final_cost_floor_ns': {str(g): {k: float(v) for k, v in self.cost_floor_ns[g].items()}
                                        for g in CARD_IDS},
                'cost_records_final': self.cost_book.records_dict(),
                'controller_snapshot': self.controller.snapshot() if self.controller else None,
                'warmup_cache_reset_confirmed': (self.warmup_report or {}).get(
                    'cache_reset_confirmed'),
                'actor_inventory': getattr(self, 'actor_inventory', []),
                'async_prefix_pipeline': self.async_prefix_pipeline,
                'gpu_quality_head': self.gpu_quality_head,
                'inline_fast_continuation': self.inline_fast_continuation,
                'fair_input_processing': self.fair_input_processing,
                'pending_input_outcomes': len(self.pending_input_outcomes),
                'ingress_lag_ms': _ms_stats(self.ingress_lag_samples, self.ingress_lag_max_ns),
                'scheduler_input_wait_ms': _ms_stats(self.scheduler_input_wait_samples,
                                                     self.scheduler_input_wait_max_ns),
                'input_admission_wait_ms': _ms_stats(self.input_admission_wait_samples,
                                                     self.input_admission_wait_max_ns),
                'input_audit': {'frames': self.raw_input_frames,
                    'arrivals': self.raw_input_arrivals,
                    'raw_sequence_sha256': self.raw_input_digest.hexdigest()},
                'pipeline_pending_decisions': len(self.pipeline_prefix_seen),
                'pipeline_early_decisions': len(self.pipeline_early_decisions),
                'pipeline_pending_releases': len(self.pipeline_released),
                'summary_path': str(self.folder / 'group_summary.json')})


def group_worker(payload):
    """Spawn target: set the CUDA mask before the first CUDA call.

    Joint barrier contract: the worker loads its two GPU models and its one
    GPU post-Head, runs the ONE-START warmup calibration (TASK.md section 4)
    through the normal actor protocol, announces READY, then waits for the
    shared source to release the ONE common t0.  Nothing here re-acquires the
    shared flock; the joint parent owns it.
    """
    if payload.get('isolated_session'):
        os.setsid()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(payload['gpu_uuids'])
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    cpu = payload.get('cpu') or {}
    cpu_pin = None
    if cpu:
        for g in CARD_IDS:
            os.environ['T011_GPU_CPU_%d' % g] = str(int(cpu['gpu'][g]))
            os.environ['T012_GPU_CPU_%d' % g] = str(int(cpu['gpu'][g]))
        os.environ['T011_DECISION_CPU'] = str(int(cpu['decision']))
        os.environ['T012_DECISION_CPU'] = str(int(cpu['decision']))
        os.environ['T011_MAIN_CPU'] = str(int(cpu['main']))
        cpu_pin = pin_cpu(cpu['main'])
    ready_q = payload.get('ready_q')
    go_q = payload.get('go_q')
    rt = None
    try:
        rt = SharedGroupRuntime(payload['name'], payload['spec'], payload['policy'],
                                payload['gpu_uuids'], payload['folder'],
                                payload.get('t0_ns'), payload['deadline_ns'],
                                payload['duration_s'], payload['bank'],
                                payload['frame_q'], payload.get('status_q'),
                                warmup_id_base=payload.get('warmup_id_base', -1_000_000),
                                reference_path=payload.get('reference_path'),
                                reference_sha256=payload.get('reference_sha256'))
        rt.start_actors()
        rt.run_warmup()
        rt.build_controller()
        if payload['spec'].get('log_process_cpu') is not None:
            # Start the disk process before READY/epoch, never during arrivals.
            rt.open_streams()
        if ready_q is not None:
            ready_q.put({'group': payload['name'], 'policy': payload['policy'], 'ready': True,
                         'pid': os.getpid(), 'gpu_uuids': list(payload['gpu_uuids']),
                         'cpu': dict(cpu), 'cpu_pin': cpu_pin,
                         'replay_worker': rt.replay_worker.stats() if rt.replay_worker else None,
                         'reference_path': rt.reference_path,
                         'reference_sha256': rt.reference_sha256,
                         'warmup': {k: v for k, v in (rt.warmup_report or {}).items()
                                    if k != 'records'},
                         'actor_inventory': rt.actor_inventory})
        if go_q is not None:
            release = go_q.get(timeout=1800)
            rt.t0_ns = int(release['t0_ns'])
            rt.rebase_clock()
        summary = rt.run()
        rt.stop_actors()
        logs = summary.get('log_writer') or {}
        if (not summary.get('log_writer_joined') or logs.get('error') or
                logs.get('written') != logs.get('enqueued') or logs.get('value_dropped')):
            raise RuntimeError('Disk log closure or completeness failed')
        if payload.get('status_q') is not None:
            payload['status_q'].put({'group': payload['name'], 'ok': True, 'summary': summary})
        write(Path(payload['folder']) / 'group_summary.json', summary)
    except Exception as exc:  # noqa: BLE001 - the parent records the real reason
        import traceback
        detail = {'group': payload['name'], 'ok': False, 'error': repr(exc),
                  'traceback': traceback.format_exc()}
        write(Path(payload['folder']) / 'group_failure.json', detail)
        if payload.get('status_q') is not None:
            payload['status_q'].put(detail)
        if ready_q is not None:
            ready_q.put(detail)
        if rt is not None and payload.get('isolated_session'):
            rt._stop_replay_worker()
            for process in getattr(rt,'ps',[]):
                if process.is_alive():
                    process.terminate()
            for process in getattr(rt,'ps',[]):
                process.join(timeout=2)
            rt.close_streams()
        raise
