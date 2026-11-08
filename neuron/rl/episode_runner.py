class EpisodeRunner:

    def __init__(
        self,
        environment,
        network,
        input_neurons,
        agent,
        trainer,
        reward_manager,
    ):
        self.environment = environment
        self.network = network
        self.input_neurons = input_neurons
        self.agent = agent
        self.trainer = trainer
        self.reward_manager = reward_manager

    def run_episode(self, steps_per_action=30, learning=True):
        state_vector, target = self.environment.reset()
        done = False
        total_reward = 0.0
        steps_count = 0
        last_outputs = []
        action = None

        while not done and steps_count < 3:
            steps_count += 1
            
            for _ in range(max(0, steps_per_action)):
                for i, val in enumerate(state_vector):
                    if val == 1 and i < len(self.input_neurons):
                        self.input_neurons[i].stimulate(
                            spike=1,
                            time=self.network.clock.time,
                        )

                last_outputs = self.network.step()

            action = self.agent.act()
            next_state_vector, reward, done = self.environment.step(action)
            total_reward += reward

            is_step_correct = (reward > 0)
            
            if learning:
                self.trainer.train_step(self.network, action, target)
                # Cập nhật hiệu suất cho SubRegionManager theo từng bước/mục tiêu
                if hasattr(self.trainer, 'sub_region_manager') and self.trainer.sub_region_manager:
                    self.trainer.sub_region_manager.update_region_performance(target, is_step_correct)
            
            target = getattr(self.environment, 'target_sequence', [target])[self.environment.current_step_idx] if hasattr(self.environment, 'current_step_idx') else target
            state_vector = next_state_vector

        is_correct = (total_reward > 0)

        return {
            "pattern": state_vector,
            "target": target,
            "action": action,
            "reward": total_reward,
            "correct": is_correct,
            "outputs": last_outputs,
            "synapses": len(self.network.synapses),
        }
    
    def train(self, episodes=150):
        history = []
        recent_correct = []

        for episode in range(max(0, episodes)):
            result = self.run_episode(steps_per_action=30, learning=True)
            history.append(result)
            recent_correct.append(1 if result["correct"] else 0)

            if len(recent_correct) > 20:
                recent_correct.pop(0)

            if episode % 10 == 0:
                acc = sum(recent_correct) / len(recent_correct) * 100
                print(
                    f"episode={episode:02d} | "
                    f"target={result['target']} | "
                    f"action={result['action']} | "
                    f"reward={result['reward']:+.1f} | "
                    f"acc_20={acc:.1f}% | "
                    f"syn={result['synapses']}"
                )

        return history

    def evaluate(self, num_tests=30):
        """Chạy kiểm thử tuần tự không có học tập (No-learning evaluation)"""
        correct = 0
        print("\n--- BẮT ĐẦU KIỂM THỬ TUẦN TỰ (EVALUATION MODE) ---")
        for _ in range(num_tests):
            result = self.run_episode(steps_per_action=30, learning=False)
            if result["correct"]:
                correct += 1
            print(f"Test Chuỗi -> Target: {result['target']} | Agent Action: {result['action']} ({'ĐÚNG' if result['correct'] else 'SAI'})")
        
        accuracy = (correct / num_tests) * 100
        print(f"Độ chính xác tổng kết chuỗi: {accuracy:.1f}% ({correct}/{num_tests})\n")
        return accuracy