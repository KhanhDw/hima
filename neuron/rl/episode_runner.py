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

    def run_episode(self, steps=50, learning=True):
        pattern_vector, target = self.environment.observe()
        last_outputs = []

        for _ in range(max(0, steps)):
            for i, val in enumerate(pattern_vector):
                if val == 1 and i < len(self.input_neurons):
                    self.input_neurons[i].stimulate(
                        spike=1,
                        time=self.network.clock.time,
                    )

            last_outputs = self.network.step()
            
            # Xóa dòng self.trainer.update_network(self.network) ở đây 
            # để không bị nhiễu STDP liên tục từng step.

        action = self.agent.act()
        
        # Chỉ cập nhật học tập (R-STDP và Homeostasis) dựa trên kết quả cuối episode nếu learning = True
        if learning:
            reward = self.reward_manager.evaluate(
                network=self.network,
                action=action,
                target=target,
            )
            for neuron in self.network.neurons:
                self.trainer.homeostasis.update(neuron)
        else:
            reward = 1.0 if action == target else -1.0

        is_correct = (action == target)

        return {
            "pattern": pattern_vector,
            "target": target,
            "action": action,
            "reward": reward,
            "correct": is_correct,
            "outputs": last_outputs,
            "synapses": len(self.network.synapses),
        }

    def train(self, episodes=100):
        history = []
        recent_correct = []

        for episode in range(max(0, episodes)):
            result = self.run_episode(learning=True)
            history.append(result)
            recent_correct.append(1 if result["correct"] else 0)

            if len(recent_correct) > 20:
                recent_correct.pop(0)

            if episode % 10 == 0:
                acc = sum(recent_correct) / len(recent_correct) * 100
                print(
                    f"episode={episode:02d} | "
                    f"pattern={result['pattern']} | "
                    f"target={result['target']} | "
                    f"action={result['action']} | "
                    f"reward={result['reward']:+.1f} | "
                    f"acc_20={acc:.1f}% | "
                    f"syn={result['synapses']}"
                )

        return history

    def evaluate(self, num_tests=30):
        """Chạy kiểm thử không có học tập (No-learning evaluation)"""
        correct = 0
        print("\n--- BẮT ĐẦU KIỂM THỬ (EVALUATION MODE) ---")
        for _ in range(num_tests):
            result = self.run_episode(steps=30, learning=False)
            if result["correct"]:
                correct += 1
            print(f"Test Pattern: {result['pattern']} -> Target: {result['target']} | Agent Action: {result['action']} ({'ĐÚNG' if result['correct'] else 'SAI'})")
        
        accuracy = (correct / num_tests) * 100
        print(f"Độ chính xác tổng kết: {accuracy:.1f}% ({correct}/{num_tests})\n")
        return accuracy