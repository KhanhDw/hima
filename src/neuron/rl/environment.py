import random


class Environment:
    """
    Môi trường huấn luyện 2 nhánh cho SNN.

    State vector 8 chiều — KHỚP với Encoder(num_neurons=8):
      [0:3]  value one-hot:    0=[1,0,0], 1=[0,1,0], 2=[0,0,1]
      [3:6]  position binary:  0=000, 1=001, ..., 7=111
      [6:8]  branch one-hot:   A=[1,0], B=[0,1]

    Branches 8 bước — khớp 3-bit position (0-7).
    Mặc định terminate_on_wrong=False để SNN có đủ training signal
    (tránh episode kết thúc sau 1-2 step khi network còn random).
    """

    def __init__(self, terminate_on_wrong: bool = False):
        # 8 bước mỗi nhánh — pattern khác nhau để network phải dùng branch flag
        self.branch_a = [0, 1, 2, 0, 1, 2, 0, 1]
        self.branch_b = [1, 2, 0, 1, 2, 0, 1, 2]
        self.branches = [self.branch_a, self.branch_b]

        self.terminate_on_wrong = terminate_on_wrong

        self.branch_id = 0
        self.current_branch = self.branch_a
        self.target_sequence = self.current_branch
        self.current_step_idx = 0
        self.current_state_vector = self._encode_state(0, 0)

    # ──────────────────────────────────────────────────────────
    # Encoding — 8 dims, khớp input_neurons/Encoder
    # ──────────────────────────────────────────────────────────

    def _encode_state(self, target: int, step_idx: int) -> list:
        # 3 dims: value one-hot
        val = [0, 0, 0]
        if target in (0, 1, 2):
            val[target] = 1

        # 3 dims: step_idx as binary (0-7 vừa 3 bit)
        idx_clamped = step_idx & 0x7
        pos = [
            (idx_clamped >> 2) & 1,
            (idx_clamped >> 1) & 1,
            idx_clamped & 1,
        ]

        # 2 dims: branch one-hot
        branch = [1, 0] if self.branch_id == 0 else [0, 1]

        return val + pos + branch

    # ──────────────────────────────────────────────────────────
    # API
    # ──────────────────────────────────────────────────────────

    def reset(self):
        """Bắt đầu episode mới — random branch. Trả (state_vector, target)."""
        self.branch_id = random.choice([0, 1])
        self.current_branch = self.branches[self.branch_id]
        self.target_sequence = self.current_branch
        self.current_step_idx = 0

        target = self.target_sequence[0]
        self.current_state_vector = self._encode_state(target, 0)
        return list(self.current_state_vector), target

    def observe(self):
        """Trả (state_vector, target hiện tại) — có guard."""
        if not self.target_sequence:
            return list(self.current_state_vector), 0
        idx = min(self.current_step_idx, len(self.target_sequence) - 1)
        target = self.target_sequence[idx]
        return list(self.current_state_vector), target

    def step(self, action):
        """
        Trả (next_state_vector, reward, done).
        Mặc định: sai action → reward=-1, KHÔNG kết thúc episode
        (chỉ kết thúc khi hết sequence hoặc terminate_on_wrong=True).
        """
        if self.current_step_idx >= len(self.target_sequence):
            return list(self.current_state_vector), 0.0, True

        expected = self.target_sequence[self.current_step_idx]

        if action == expected:
            reward = 1.0
            self.current_step_idx += 1

            if self.current_step_idx >= len(self.target_sequence):
                done = True
                self.current_step_idx = 0
                self.current_state_vector = self._encode_state(0, 0)
            else:
                done = False
                next_target = self.target_sequence[self.current_step_idx]
                self.current_state_vector = self._encode_state(
                    next_target, self.current_step_idx
                )
        else:
            reward = -1.0
            if self.terminate_on_wrong:
                done = True
                self.current_step_idx = 0
                self.current_state_vector = self._encode_state(
                    self.target_sequence[0], 0
                )
            else:
                # Không kết thúc — chỉ phạt. State giữ nguyên để SNN thử lại.
                done = False
                target = self.target_sequence[self.current_step_idx]
                self.current_state_vector = self._encode_state(
                    target, self.current_step_idx
                )

        return list(self.current_state_vector), reward, done

    # ──────────────────────────────────────────────────────────
    # Legacy helpers (giữ để tương thích)
    # ──────────────────────────────────────────────────────────

    def reward(self, action, target):
        return 1.0 if action == target else -1.0

    def evaluate(self, action, target):
        reward = self.reward(action, target)
        return reward, reward > 0