class RewardLearning:

    def __init__(
        self,
        lr=0.05
    ):
        self.lr = lr

    def update(
        self,
        synapse,
        reward
    ):

        synapse.weight += (
            reward
            * self.lr
            * synapse.last_output
        )

        synapse.clamp()