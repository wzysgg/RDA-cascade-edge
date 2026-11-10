"""Independent actual-clock result delivery; no policy or private labels."""
import heapq
import os
import queue
import time
import traceback


class ReturnTimers:
    def __init__(self, delay_ns):
        self.delay_ns=delay_ns
        self.heap=[]
        self.sequence=0

    def accept(self, message, received_ns):
        if received_ns<message['ready_ns']:
            raise ValueError('Result received before GPU produced it')
        self.sequence+=1
        heapq.heappush(self.heap,(received_ns+self.delay_ns,self.sequence,
                                 received_ns,message))

    def pop_due(self, now):
        if not self.heap or self.heap[0][0]>now:return None
        due,_,received,message=heapq.heappop(self.heap)
        return dict(message,received_ns=received,due_ns=due,returned_ns=now)


def response_worker(incoming,delivered,logs,epoch,cfg,folder):
    from live_batch_runtime import cpu_snapshot,save_process_cpu
    os.sched_setaffinity(0,{cfg['response_cpu']})
    try:
        while not epoch.value:time.sleep(.001)
        while time.monotonic_ns()<epoch.value:time.sleep(.0005)
        baseline=cpu_snapshot()
        timers=ReturnTimers(cfg['network_return_ns']);stopping=False
        while not stopping or timers.heap:
            now=time.monotonic_ns()-epoch.value
            row=timers.pop_due(now)
            if row is not None:
                # This timestamp is the actual independent return event, never
                # a reconstructed GPU finish/deadline or a parent receipt time.
                delivered.put(dict(type='DELIVERED',**row))
                logs.put(('returns',row))
                continue
            # Queue.get(timeout) rounds sub-ms waits up to a whole ms on
            # Linux's pipe poll path. Keep it off the 0.5ms return timer.
            # Short sleeps yield the CPU; actual overshoot stays measured.
            try:
                message=incoming.get_nowait() if timers.heap else incoming.get(timeout=.01)
            except queue.Empty:
                if timers.heap:
                    remaining=timers.heap[0][0]-(time.monotonic_ns()-epoch.value)
                    if remaining>0:time.sleep(min(.0001,remaining/1e9))
                continue
            if message is None:stopping=True
            else:timers.accept(message,time.monotonic_ns()-epoch.value)
        delivered.put(dict(type='RESPONSE_STOPPED'))
    except Exception:
        delivered.put(dict(type='ERROR',error=traceback.format_exc()))
        raise
    finally:
        delivered.close();delivered.join_thread()
        logs.close();logs.join_thread()
        if 'baseline' in locals():save_process_cpu(folder,'response',baseline)
