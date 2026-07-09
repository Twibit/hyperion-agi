import math
import multiprocessing
import time
import ctypes
import threading
import numpy as np

class AstrocyteAttention:
    """Calculates biological attention weights using the Golden Ratio.

    PATCH: the original implementation used three nested pure-Python loops for
    matmul/transpose/softmax (O(m·n·p) at interpreter speed). That made the
    glial attention path ~1000× slower than necessary and effectively
    unusable for any sequence longer than ~32 tokens. Replaced with NumPy.
    Accepts either Python lists (legacy) or numpy arrays.
    """
    def __init__(self):
        self.golden_ratio = 1.618

    def forward(self, Q, K, V):
        Q = np.asarray(Q, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        # V kept for signature compatibility; weights returned by this method.
        scores = Q @ K.T / self.golden_ratio
        scores -= scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        return e / e.sum(axis=-1, keepdims=True)


class NeuralActor(multiprocessing.Process):
    """Isolated neural cluster."""
    def __init__(self, neuron_id, target_func, heartbeat_array, index):
        super().__init__()
        self.neuron_id = neuron_id
        self.target_func = target_func
        self.heartbeat_array = heartbeat_array
        self.index = index

    def run(self):
        def _beat():
            import os
            import sys
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.append(project_root)
            from cognitive_plugins.pineal_gland import PinealGland
            pineal = PinealGland()
            while True:
                self.heartbeat_array[self.index] = time.time()
                pineal.sleep(500)
        t = threading.Thread(target=_beat, daemon=True)
        t.start()
        try:
            self.target_func()
        except Exception:
            pass

class AstrocyteOversight:
    """Erlang Actor Model Glial Supervisor."""
    def __init__(self, max_neurons=1000, timeout=2.0):
        self.timeout = timeout
        self.heartbeats = multiprocessing.Array(ctypes.c_double, max_neurons)
        self.actors = {}
        self.next_index = 0

    def spawn(self, neuron_id, target_func):
        idx = self.next_index
        self.next_index += 1
        self.heartbeats[idx] = time.time()
        actor = NeuralActor(neuron_id, target_func, self.heartbeats, idx)
        actor.start()
        self.actors[neuron_id] = {'process': actor, 'index': idx, 'status': 'active'}

    def monitor(self):
        now = time.time()
        for nid, state in self.actors.items():
            if state['status'] == 'active':
                if now - self.heartbeats[state['index']] > self.timeout:
                    self.isolate(nid)

    def isolate(self, neuron_id):
        state = self.actors[neuron_id]
        state['status'] = 'isolated'
        proc = state['process']
        if proc.is_alive():
            proc.terminate()
            proc.join(1)
            if proc.is_alive():
                proc.kill()
