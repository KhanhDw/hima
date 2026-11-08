from collections import deque


class Synapse:
    def __init__(
        self,
        weight=0.5,
        delay=1,
        pre=None,
        post=None
    ):
        self.weight = weight

        self.delay = max(1, delay)

        self.queue = deque()

        self.last_output = 0.0

        self.pre_spikes = deque(
            maxlen=32
        )
        self.pre = pre
        self.post = post

        # --- Bổ sung Eligibility Trace ---
        self.eligibility_trace = 0.0
        self.trace_decay = 0.95  # Tỷ lệ suy giảm theo thời gian/step
        self.learning_rate = 0.01

    def receive(
        self,
        spike,
        time
     ):

        if spike:

            self.pre_spikes.append(time)

            self.queue.append(
                (
                    time + self.delay,
                    spike
                )
            )

    def transmit(self, current_time):
        total_spike = 0

        while (
            self.queue
            and
            self.queue[0][0] <= current_time
        ):

            _, spike = self.queue.popleft()

            total_spike += spike

        self.last_output = (
            total_spike * self.weight
        )

        return self.last_output

    def update_trace(self, pre_spiked, post_spiked):
        """Cập nhật vết dấu hiệu năng dựa trên xung của pre và post neuron"""
        self.eligibility_trace *= self.trace_decay
        if pre_spiked or post_spiked:
            self.eligibility_trace += 1.0

    def apply_reward_modulation(self, reward):
        """Áp dụng phần thưởng dựa trên eligibility trace đã tích lũy"""
        if self.eligibility_trace != 0.0:
            self.weight += self.learning_rate * reward * self.eligibility_trace
            self.clamp()

    def clamp(self):

        self.weight = max(
            0.0,
            min(5.0, self.weight)
        )