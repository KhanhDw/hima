class ImaginationEngine:

    def __init__(self, world_model):
        self.world_model = world_model

    def rollout(
        self,
        initial_state,
        actions,
        max_depth=5
    ):
        state = list(initial_state)

        trajectory = []

        for action in actions[:max_depth]:

            predicted, conf = \
                self.world_model.predict(
                    state,
                    action
                )

            if predicted is None:
                break

            trajectory.append({
                "action": action,
                "predicted_state": predicted,
                "confidence": conf
            })

            state = predicted

        return trajectory