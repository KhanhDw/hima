from ..plasticity.rstdp import (
    RewardModulatedSTDP
)


class RewardManager:

    def __init__(
        self,
        positive_reward=1.0,
        negative_reward=-1.0
    ):
        self.positive_reward = (
            positive_reward
        )

        self.negative_reward = (
            negative_reward
        )

        self.rstdp = (
            RewardModulatedSTDP()
        )

    def compute_reward(
        self,
        action,
        target
    ):

        if action == target:
            return self.positive_reward

        return self.negative_reward

    def apply_reward(
        self,
        network,
        reward
    ):
        """
        Áp dụng reward lên toàn bộ
        synapse trong network.
        """

        for synapse in network.synapses:

            post = synapse.post

            if post is None:
                continue

            if (
                post.last_spike_time < 0
            ):
                continue

            self.rstdp.update(
                synapse=synapse,
                post_spike_time=
                post.last_spike_time,
                reward=reward
            )

    def evaluate(
        self,
        network,
        action,
        target
    ):

        reward = (
            self.compute_reward(
                action,
                target
            )
        )

        self.apply_reward(
            network,
            reward
        )

        return reward