class UncertaintyEstimator:

    @staticmethod
    def estimate(
        matches_count,
        max_reference=20
    ):
        if matches_count <= 0:
            return 1.0

        confidence = min(
            1.0,
            matches_count / max_reference
        )

        uncertainty = 1.0 - confidence

        return round(
            uncertainty,
            4
        )