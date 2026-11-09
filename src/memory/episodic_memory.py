# src/memory/episodic_memory.py
import time
import numpy as np
from collections import deque


class EpisodicMemory:
    """
    Level 15: Episodic memory tối ưu bằng numpy.
    - States lưu dạng ndarray (N, dim) để similarity = vectorized
    - Không sort full: dùng argpartition cho top-k
    - Action-indexed để filter nhanh
    """

    def __init__(self, capacity=10_000, state_dim=8):
        self.capacity = capacity
        self.state_dim = state_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.timestamps = np.zeros(capacity, dtype=np.float64)
        self._size = 0
        self._ptr = 0   # vị trí ghi tiếp theo (circular)

    def _normalize_state(self, state) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32).ravel()
        if arr.shape[0] < self.state_dim:
            arr = np.pad(arr, (0, self.state_dim - arr.shape[0]))
        elif arr.shape[0] > self.state_dim:
            arr = arr[:self.state_dim]
        return arr

    def store(self, state, action, context, reward, next_state, done=False):
        s = self._normalize_state(state)
        ns = self._normalize_state(next_state)
        idx = self._ptr
        self.states[idx] = s
        self.actions[idx] = int(action) if action is not None else -1
        self.rewards[idx] = float(reward)
        self.next_states[idx] = ns
        self.dones[idx] = bool(done)
        self.timestamps[idx] = time.time()
        self._ptr = (self._ptr + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1
        return idx

    def query_similar(self, state, top_k=10, min_similarity=0.5):
        if self._size == 0:
            return []
        s = self._normalize_state(state)
        buf = self.states[:self._size]
        # Vectorized Euclidean distance
        diffs = buf - s
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        sims = 1.0 / (1.0 + dists)

        mask = sims >= min_similarity
        if not mask.any():
            return []

        valid_idx = np.where(mask)[0]
        valid_sims = sims[valid_idx]

        k = min(top_k, len(valid_idx))
        top_local = np.argpartition(-valid_sims, k - 1)[:k]
        top_idx = valid_idx[top_local]
        top_sims = valid_sims[top_local]

        order = np.argsort(-top_sims)
        result = []
        for i in order:
            gi = int(top_idx[i])
            ep = {
                "state": self.states[gi].tolist(),
                "action": int(self.actions[gi]),
                "reward": float(self.rewards[gi]),
                "next_state": self.next_states[gi].tolist(),
                "done": bool(self.dones[gi]),
                "timestamp": float(self.timestamps[gi]),
            }
            result.append((float(top_sims[i]), ep))
        return result

    def action_value(self, state, action, top_k=10):
        similar = self.query_similar(state, top_k=top_k)
        rewards = [
            ep["reward"] for _, ep in similar if ep["action"] == action
        ]
        if not rewards:
            return 0.0, 0
        return sum(rewards) / len(rewards), len(rewards)

    def statistics(self):
        return {
            "size": self._size,
            "capacity": self.capacity,
            "fill_ratio": self._size / self.capacity,
        }

    def clear(self):
        self._size = 0
        self._ptr = 0