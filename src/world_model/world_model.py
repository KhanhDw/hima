from src.world_model.transition_memory \
    import TransitionMemory

from src.world_model.state_predictor \
    import StatePredictor

from src.world_model.uncertainty_estimator \
    import UncertaintyEstimator

from src.world_model.causal_discovery \
    import CausalDiscovery

from src.world_model.concept_world_graph \
    import ConceptWorldGraph

from src.world_model.imagination_engine \
    import ImaginationEngine

from src.world_model.model_updater \
    import ModelUpdater


class WorldModel:

    def __init__(self):

        self.memory = \
            TransitionMemory()

        self.predictor = \
            StatePredictor(
                self.memory
            )

        self.causal_discovery = \
            CausalDiscovery()

        self.graph = \
            ConceptWorldGraph()

        self.updater = \
            ModelUpdater()

        self.imagination = \
            ImaginationEngine(
                self
            )

    def observe_transition(
        self,
        state,
        action,
        next_state,
        reward=0.0
    ):

        self.memory.store(
            state,
            action,
            next_state,
            reward
        )

        self.causal_discovery.observe(
            state,
            action,
            next_state
        )

    def predict(
        self,
        state,
        action
    ):
        return self.predictor.predict(
            state,
            action
        )

    def evaluate_prediction(
        self,
        predicted_state,
        actual_state
    ):
        return self.updater.update(
            predicted_state,
            actual_state
        )

    def uncertainty(
        self,
        state,
        action
    ):
        matches = self.memory.find_matches(
            state,
            action
        )

        return UncertaintyEstimator.estimate(
            len(matches)
        )

    def imagine(
        self,
        state,
        actions
    ):
        return self.imagination.rollout(
            state,
            actions
        )

    def statistics(self):

        return {
            "transitions":
                self.memory.size(),

            "last_error":
                self.updater.last_error
        }