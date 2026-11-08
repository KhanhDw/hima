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


    def clamp(self):

        self.weight = max(
            0.0,
            min(5.0, self.weight)
        )
