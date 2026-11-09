from src.prediction.prediction_error \
    import PredictionError


class ModelUpdater:

    def __init__(self):
        self.last_error = 0.0

    def update(
        self,
        predicted_state,
        actual_state
    ):
        if predicted_state is None:
            return 1.0

        error = PredictionError.calculate(
            predicted_state,
            actual_state
        )

        self.last_error = error

        return error

    def should_learn(
        self,
        threshold=0.2
    ):
        return self.last_error > threshold