from typing import cast
from src.integration.global_workspace import GlobalWorkspace
from src.integration.cognitive_orchestrator import CognitiveOrchestrator
from src.autonomous_goal.drive_motivator import DriveMotivator
from src.autonomous_goal.curiosity_engine import CuriosityEngine
from src.autonomous_goal.goal_generator import GoalGenerator
from src.meta.meta_learner import MetaLearner
from src.meta.performance_monitor import PerformanceMonitor

from src.neuron.network.network import Network
from src.neuron.core.input_neuron import InputNeuron
from src.neuron.core.neuron import Neuron
from src.neuron.learning.trainer import Trainer
from src.neuron.learning.reward_manager import RewardManager
from src.neuron.learning.sub_region_manager import SubRegionManager
from src.neuron.rl.agent import Agent
from src.neuron.rl.environment import Environment

from src.representation.encoder import Encoder
from src.memory.sequence_memory import SequenceMemory
from src.memory.context_memory import ContextMemory
from src.memory.working_memory import WorkingMemory

from src.prediction.predictor import Predictor
from src.prediction.prediction_error import PredictionError
from src.prediction.confidence import ConfidenceEstimator
from src.prediction.surprise import SurpriseCalculator

from src.hierarchy.chunker import Chunker
from src.hierarchy.abstraction_layer import AbstractionLayer
from src.hierarchy.meta_representation import MetaRepresentation

from src.planning.goal_manager import GoalManager
from src.planning.planner import Planner
from src.planning.intrinsic_reward import IntrinsicRewardCalculator
from src.world_model.world_model import WorldModel

# ── Optional imports — graceful fallback ─────────────────────
try:
    from src.memory.memory_consolidator import MemoryConsolidator
except ImportError:
    MemoryConsolidator = None  # type: ignore[assignment,misc]

try:
    from src.memory.episodic_memory import EpisodicMemory
except ImportError:
    EpisodicMemory = None  # type: ignore[assignment,misc]

try:
    from src.memory.temporal_trace import TemporalTrace
except ImportError:
    TemporalTrace = None  # type: ignore[assignment,misc]

try:
    from src.memory.memory_router import MemoryRouter
except ImportError:
    MemoryRouter = None  # type: ignore[assignment,misc]


CHAT_ACTION = -1


class MainCognitiveCore:
    def __init__(self, warmup_episodes: int = 0):
        """
        Args:
            warmup_episodes: số episode đầu chạy SNN thuần (không arbiter).
                            Hữu ích để world_model + episodic có data trước.
        """
        print("Đang khởi tạo Tích hợp Nhận thức Toàn cục (Cấp 15)...")

        # ── 1. SNN ───────────────────────────────────────────
        self.net = Network()
        self.input_neurons = [InputNeuron() for _ in range(8)]
        self.out_a = Neuron()
        self.out_b = Neuron()
        self.out_c = Neuron()

        for inp in self.input_neurons:
            self.net.add(inp)
        self.net.add(self.out_a)
        self.net.add(self.out_b)
        self.net.add(self.out_c)

        from src.neuron.network.connector import connect
        self.net.add_synapse(
            connect(self.input_neurons[0], self.out_a, weight=1.0, delay=1)
        )
        self.net.add_synapse(
            connect(self.input_neurons[1], self.out_b, weight=1.0, delay=1)
        )
        self.net.add_synapse(
            connect(self.input_neurons[2], self.out_c, weight=1.0, delay=1)
        )
        for i in range(3, 8):
            self.net.add_synapse(
                connect(self.input_neurons[i], self.out_a, weight=0.2, delay=1)
            )
            self.net.add_synapse(
                connect(self.input_neurons[i], self.out_b, weight=0.2, delay=1)
            )
            self.net.add_synapse(
                connect(self.input_neurons[i], self.out_c, weight=0.2, delay=1)
            )

        # ── 2. Nhận thức cơ sở ───────────────────────────────
        self.encoder = Encoder(num_neurons=8, min_val=0.0, max_val=2.0)
        self.sequence_memory = SequenceMemory(max_history_length=10)
        self.context_memory = ContextMemory()
        self.working_memory = WorkingMemory(capacity=5)
        self.predictor = Predictor()
        self.chunker = Chunker(frequency_threshold=3)
        self.abstraction_layer = AbstractionLayer()
        self.meta_rep = MetaRepresentation()
        self.goal_manager = GoalManager()
        self.planner = Planner()

        self.sub_region_manager = SubRegionManager(freeze_threshold=90.0)
        self.trainer = Trainer(
            input_neurons=self.input_neurons,
            output_neurons=[self.out_a, self.out_b, self.out_c],
            sub_region_manager=self.sub_region_manager,
        )
        self.reward_manager = RewardManager(
            positive_reward=1.0, negative_reward=-1.0
        )
        self.agent = Agent(output_neurons=[self.out_a, self.out_b, self.out_c])
        self.environment = Environment()

        # ── 3. Workspace ─────────────────────────────────────
        self.workspace = GlobalWorkspace()
        self.orchestrator = CognitiveOrchestrator(self.workspace)

        # ── 4. Drives / Meta ─────────────────────────────────
        self.drive_motivator = DriveMotivator()
        self.curiosity_engine = CuriosityEngine()
        self.goal_generator = GoalGenerator()
        self.meta_learner = MetaLearner()
        self.performance_monitor = PerformanceMonitor()

        # ── 5. Level 14: World Model ─────────────────────────
        self.world_model = WorldModel()
        self.last_state = None
        self.last_action = None
        self._last_cognitive_state = "IDLE"

        self.world_trust = 0.80
        self.arbiter_overrides = 0
        self.arbiter_total = 0
        self.episodic_hits = 0

        # Warm-up: đếm ngược số episode còn lại chạy SNN thuần
        self._warmup_episodes_remaining = max(0, int(warmup_episodes))

        # ── 6. Level 15: Memory layers ───────────────────────
        self.episodic_memory = None
        if EpisodicMemory is not None:
            try:
                self.episodic_memory = EpisodicMemory(
                    capacity=10_000, state_dim=8
                )
            except Exception:
                self.episodic_memory = None

        self.temporal_trace = None
        if TemporalTrace is not None:
            try:
                self.temporal_trace = TemporalTrace(
                    decay_rate=0.8, mode="ewma"
                )
            except TypeError:
                try:
                    self.temporal_trace = TemporalTrace(decay_rate=0.8)
                except Exception:
                    self.temporal_trace = None

        self.consolidator = None
        if (MemoryConsolidator is not None
                and self.episodic_memory is not None):
            try:
                self.consolidator = MemoryConsolidator(
                    episodic_memory=self.episodic_memory,
                    world_model=self.world_model,
                    trigger_every=50,
                )
            except TypeError:
                self.consolidator = None

        self.memory = None
        if (MemoryRouter is not None
                and self.temporal_trace is not None):
            try:
                self.memory = MemoryRouter(
                    sequence=self.sequence_memory,
                    working=self.working_memory,
                    context=self.context_memory,
                    temporal=self.temporal_trace,
                    episodic=self.episodic_memory,
                    consolidator=self.consolidator,
                )
            except TypeError:
                self.memory = None

        print(
            "Sẵn sàng: L13 + L14 WorldModel "
            f"+ L15 Episodic={'✓' if self.episodic_memory else '✗'} "
            f"+ Router={'✓' if self.memory else '✗'} "
            f"+ Warmup={self._warmup_episodes_remaining}"
        )

    # ══════════════════════════════════════════════════════════
    # Memory adapters
    # ══════════════════════════════════════════════════════════

    def _memory_observe_observation(self, latent):
        if self.memory is not None:
            for name in ("observe_observation", "observe_encode",
                         "encode_observation"):
                fn = getattr(self.memory, name, None)
                if callable(fn):
                    try:
                        fn(latent)
                        return
                    except Exception:
                        continue
        self.sequence_memory.append(latent)
        self.working_memory.load(latent)
        if self.temporal_trace is not None:
            try:
                self.temporal_trace.update(latent)
            except Exception:
                pass

    def _memory_observe_transition(self, state, action, context,
                                    reward, next_state, done):
        if self.memory is not None:
            fn = getattr(self.memory, "observe", None)
            if callable(fn):
                try:
                    fn(state, action, context, reward, next_state, done)
                    return
                except Exception:
                    pass
        if self.episodic_memory is not None:
            try:
                self.episodic_memory.store(
                    state, action, context, reward, next_state, done
                )
            except Exception:
                pass

    def _memory_recall(self, state, top_k=10) -> list:
        if self.memory is not None:
            fn = getattr(self.memory, "recall_similar", None)
            if callable(fn):
                try:
                    result = fn(state, top_k=top_k)
                    if isinstance(result, list):
                        return result
                    return []
                except Exception:
                    pass
        if self.episodic_memory is not None:
            fn = getattr(self.episodic_memory, "query_similar", None)
            if callable(fn):
                try:
                    result = fn(state, top_k=top_k)
                    if isinstance(result, list):
                        return result
                    return []
                except Exception:
                    pass
        return []

    def _memory_end_episode(self):
        if self.memory is not None:
            fn = getattr(self.memory, "end_episode", None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass
        if self.consolidator is not None:
            fn = getattr(self.consolidator, "maybe_consolidate", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    # ══════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════

    def _update_world_trust(self, world_error: float):
        acc = 1.0 / (1.0 + max(float(world_error), 0.0))
        self.world_trust = 0.95 * self.world_trust + 0.05 * acc
        self.world_trust = max(0.5, min(0.95, self.world_trust))

    def _episodic_reward_from_similar(self, similar, action):
        """Tính Q(s,a) từ cached similar episodes."""
        if not similar:
            return 0.0, 0
        rewards: list = []
        for item in similar:
            ep = None
            if isinstance(item, tuple) and len(item) == 2:
                cand = item[1]
                if isinstance(cand, dict):
                    ep = cand
            elif isinstance(item, dict):
                ep = item
            if ep is None or ep.get("action") != action:
                continue
            try:
                rewards.append(float(ep.get("reward", 0.0)))
            except (TypeError, ValueError):
                continue
        if not rewards:
            return 0.0, 0
        return sum(rewards) / len(rewards), len(rewards)

    def _arbitrate_action(self, state_vector, snn_action):
        # Warm-up: SNN tự quyết
        if self._warmup_episodes_remaining > 0:
            return snn_action, None, 0.0, False

        candidates = [0, 1, 2]

        # Cache: query một lần cho 3 candidate
        similar = self._memory_recall(state_vector, top_k=10)
        ep_vals = {
            c: self._episodic_reward_from_similar(similar, c)
            for c in candidates
        }

        best_action = snn_action
        best_score = float("-inf")
        best_pred = None
        best_conf = 0.0

        for c in candidates:
            try:
                pred, conf = self.world_model.predict(state_vector, c)
            except Exception:
                continue

            score = float(conf)
            if isinstance(pred, dict) and "reward" in pred:
                try:
                    score = float(pred["reward"]) * float(conf)
                except (TypeError, ValueError, KeyError):
                    pass

            ep_val, ep_count = ep_vals[c]
            if ep_count >= 3:
                score = 0.6 * score + 0.4 * ep_val
                self.episodic_hits += 1

            if score > best_score:
                best_score = score
                best_action = c
                best_pred = pred
                best_conf = float(conf)

        if best_conf >= self.world_trust:
            return best_action, best_pred, best_conf, True
        return snn_action, best_pred, best_conf, False

    def _current_context(self):
        snapshot = (
            self.context_memory.snapshot()
            if callable(getattr(self.context_memory, "snapshot", None))
            else {}
        )
        if not isinstance(snapshot, dict):
            snapshot = {}

        drives_raw = getattr(self.drive_motivator, "drives", {})
        drives_dict: dict = {}
        if isinstance(drives_raw, dict):
            for k, v in drives_raw.items():
                drives_dict[k] = getattr(v, "value", v)

        snapshot.update({
            "drives": drives_dict,
            "step": getattr(self.environment, "current_step_idx", 0),
            "cognitive_state": self._last_cognitive_state,
        })
        return snapshot

    def _safe_observe_transition(self, state, action, next_state, reward,
                                 context=None):
        observe = getattr(self.world_model, "observe_transition", None)
        if observe is None:
            return
        try:
            observe(state, action, next_state, reward, context=context)
        except TypeError:
            observe(state, action, next_state, reward)

    # ══════════════════════════════════════════════════════════
    # Episode
    # ══════════════════════════════════════════════════════════

    def run_cognitive_episode(self, learning=True, max_samples=None):
        """
        Args:
            learning: bật/tắt train SNN.
            max_samples: giới hạn số step trong episode (None = lấy hết
                        target_sequence của environment).
        """
        state_vector, target = self.environment.reset()
        done = False
        total_reward = 0.0
        steps_count = 0
        max_steps = len(self.environment.target_sequence)
        if max_samples is not None:
            max_steps = min(max_steps, int(max_samples))

        # Warmup decrement (mỗi episode giảm 1)
        if self._warmup_episodes_remaining > 0:
            self._warmup_episodes_remaining -= 1

        error = 0.0
        world_error = 0.0
        cognitive_status = {"state": "IDLE"}
        self.last_state = None
        self.last_action = None
        self.arbiter_overrides = 0
        self.arbiter_total = 0
        self.episodic_hits = 0

        while not done and steps_count < max_steps:
            steps_count += 1

            # A. Encode + Memory
            latent = self.encoder.encode_state(state_vector)
            self._memory_observe_observation(latent)
            self.workspace.broadcast(latent, source_module="MemoryCore")

            # B. Abstraction
            history = self.sequence_memory.get_sequence()
            if len(history) >= 2:
                chunk_id = self.chunker.observe_sequence(history[-2:])
                if chunk_id:
                    self.abstraction_layer.elevate(history[-1], chunk_id)
                    self.meta_rep.link("HighLevelConcept", chunk_id)

            # C. SNN tick — FIX: val > 0.5 (hỗ trợ cả binary và float input)
            for _ in range(30):
                for i, val in enumerate(state_vector):
                    if i >= len(self.input_neurons):
                        break
                    try:
                        if float(val) > 0.5:
                            self.input_neurons[i].stimulate(
                                spike=1, time=self.net.clock.time
                            )
                    except (TypeError, ValueError):
                        continue
                self.net.step()

            # D. Arbiter
            drives = self.drive_motivator.update_drives(activity_cost=0.02)
            snn_action = self.agent.act()
            (final_action, world_pred,
             world_conf, overridden) = self._arbitrate_action(
                state_vector, snn_action
            )
            self.arbiter_total += 1
            if overridden:
                self.arbiter_overrides += 1

            # E. Curiosity
            curiosity_score = self.curiosity_engine.calculate_curiosity(
                state_vector
            )
            if final_action is not None:
                try:
                    uncertainty = self.world_model.uncertainty(
                        state_vector, final_action
                    )
                except Exception:
                    uncertainty = 0.0
                curiosity_score = 0.7 * curiosity_score + 0.3 * uncertainty

            new_goal = self.goal_generator.evaluate_and_generate(
                curiosity_score, state_vector
            )
            if new_goal:
                self.goal_manager.set_goal([1, 0, 0, 0, 0, 1, 0, 0, 1])

            # F. Predictor Cấp 5
            predicted_latent = self.predictor.predict(history)

            # G. Imagination gated
            if steps_count % 5 == 0:
                imagine_fn = getattr(self.world_model, "imagine", None)
                update_fn = getattr(self.planner, "update_simulations", None)
                if callable(imagine_fn) and callable(update_fn):
                    try:
                        simulations = imagine_fn(state_vector, [0, 1, 2])
                        if simulations:
                            update_fn(simulations)
                    except (AttributeError, NotImplementedError, TypeError):
                        pass

            # H. Execute
            next_state_vector, ext_reward, done = \
                self.environment.step(final_action)

            # I. Store transition
            ctx = self._current_context()
            self._memory_observe_transition(
                state_vector, final_action, ctx,
                ext_reward, next_state_vector, done,
            )
            self._safe_observe_transition(
                state_vector, final_action, next_state_vector,
                ext_reward, context=ctx,
            )

            # FIX: guard world_pred None (khi warmup)
            if world_pred is not None:
                world_error = self.world_model.evaluate_prediction(
                    world_pred, next_state_vector
                )
                self._update_world_trust(world_error)
            else:
                world_error = 1.0

            self.last_state = list(state_vector)
            self.last_action = final_action

            # J. Prediction error
            next_latent = self.encoder.encode_state(next_state_vector)
            error = PredictionError.calculate(predicted_latent, next_latent)
            SurpriseCalculator.calculate(error)
            ConfidenceEstimator.estimate(error)

            # K. Workspace
            combined = {
                "latent": predicted_latent,
                "world": world_pred,
                "confidence": world_conf,
                "world_error": world_error,
                "source": "CombinedPredictor",
            }
            self.workspace.broadcast(
                combined, source_module="CombinedPredictor"
            )

            # L. Orchestrator
            cognitive_status = self.orchestrator.tick(
                state_vector, drives, curiosity_score
            )
            self._last_cognitive_state = cognitive_status["state"]

            # M. Meta-learn
            recent_acc = 1.0 if ext_reward > 0 else 0.0
            world_acc = 1.0 / (1.0 + world_error)
            self.meta_learner.adapt(
                recent_accuracy=0.7 * recent_acc + 0.3 * world_acc
            )

            # N. Train SNN
            total_reward += ext_reward
            is_step_correct = (ext_reward > 0)
            if learning and not overridden:
                self.trainer.train_step(self.net, snn_action, target)
                if self.sub_region_manager:
                    self.sub_region_manager.update_region_performance(
                        target, is_step_correct
                    )

            # O. Log
            self.performance_monitor.record_metrics(
                accuracy=1.0 if is_step_correct else 0.0,
                error=error,
                reward=total_reward,
            )

            target = self.environment.target_sequence[
                self.environment.current_step_idx
            ]
            state_vector = next_state_vector

        self._memory_end_episode()

        # FIX: accuracy từ reward (reward = correct - wrong)
        if steps_count > 0:
            accuracy = (total_reward + steps_count) / (2.0 * steps_count)
            accuracy = max(0.0, min(1.0, accuracy))
            is_correct = accuracy > 0.5
        else:
            accuracy = 0.0
            is_correct = False

        stats = self.world_model.statistics()

        # FIX: return statement nằm đúng vị trí, không nằm trong if
        ep_size = 0
        if self.episodic_memory is not None:
            stat_fn = getattr(self.episodic_memory, "statistics", None)
            if callable(stat_fn):
                try:
                    stats_result = stat_fn()
                    if isinstance(stats_result, dict):
                        raw_size = stats_result.get("size", 0)
                        if isinstance(raw_size, int):
                            ep_size = raw_size
                except Exception:
                    pass

        return {
            "correct": is_correct,
            "accuracy": accuracy,
            "steps_count": steps_count,
            "reward": total_reward,
            "synapses": len(self.net.synapses),
            "last_error": error,
            "chunk_count": len(self.chunker.get_chunks()),
            "world_transitions": stats["transitions"],
            "world_error": stats["last_error"],
            "world_trust": self.world_trust,
            "arbiter_override_rate":
                self.arbiter_overrides / max(self.arbiter_total, 1),
            "episodic_size": ep_size,
            "episodic_blend_rate":
                self.episodic_hits / max(self.arbiter_total, 1),
            "cognitive_state": cognitive_status["state"],
        }

    # ══════════════════════════════════════════════════════════
    # Chat step
    # ══════════════════════════════════════════════════════════

    def step(self, sensory_input):
        latent = self.encoder.encode_state(sensory_input)
        self._memory_observe_observation(latent)
        conscious_packet = self.workspace.broadcast(
            latent, source_module="LinguisticBridge"
        )

        if self.last_state is not None:
            chat_ctx = {"type": "chat"}
            self._memory_observe_transition(
                self.last_state, CHAT_ACTION, chat_ctx,
                0.0, sensory_input, False,
            )
            self._safe_observe_transition(
                self.last_state, CHAT_ACTION, sensory_input, 0.0,
                context=chat_ctx,
            )

        drives = self.drive_motivator.update_drives(activity_cost=0.02)
        curiosity_score = self.curiosity_engine.calculate_curiosity(
            sensory_input
        )

        cognitive_status = self.orchestrator.tick(
            sensory_input, drives, curiosity_score
        )

        new_goal = self.goal_generator.evaluate_and_generate(
            curiosity_score, sensory_input
        )
        if new_goal:
            self.goal_manager.set_goal([1, 0, 0, 0, 0, 1, 0, 0, 1])
            cognitive_status["new_goal"] = new_goal

        for _ in range(10):
            for i, val in enumerate(sensory_input):
                if i >= len(self.input_neurons):
                    break
                try:
                    if float(val) > 0.5:
                        self.input_neurons[i].stimulate(
                            spike=1, time=self.net.clock.time
                        )
                except (TypeError, ValueError):
                    continue
            self.net.step()

        current_lr = self.meta_learner.adapt(recent_accuracy=0.8)
        cognitive_status["adaptive_learning_rate"] = current_lr
        cognitive_status["focus"] = conscious_packet

        self.last_state = list(sensory_input)
        self.last_action = CHAT_ACTION
        self._last_cognitive_state = cognitive_status["state"]

        return cognitive_status


if __name__ == "__main__":
    core = MainCognitiveCore()
    print("\nChạy thử nghiệm Cấp 15...\n")

    for episode in range(20):
        res = core.run_cognitive_episode(learning=True)
        print(
            f"Ep {episode:02d} | "
            f"{res['cognitive_state']:<22} | "
            f"Acc:{res['accuracy']*100:5.1f}% | "
            f"R:{res['reward']:+6.1f} | "
            f"WErr:{res['world_error']:.3f} | "
            f"Trust:{res['world_trust']:.2f} | "
            f"Arb:{res['arbiter_override_rate']*100:3.0f}% | "
            f"EpBl:{res['episodic_blend_rate']*100:3.0f}% | "
            f"Syn:{res['synapses']}"
        )

    print("\n=== Chat mode ===")
    status = core.step([1, 0, 1, 0, 0, 1, 0, 0])
    print(f"State: {status['state']}")
    print(f"LR:    {status['adaptive_learning_rate']:.4f}")