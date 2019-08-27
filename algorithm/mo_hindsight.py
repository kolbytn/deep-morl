import time
import numpy as np
import csv
import os

from env.lava_env import LavaEnv
from env.door_env import DoorEnv
from rl_util.functions import generate_weights, get_default_weights, visualize
from agent.dqn import DQNAgent


class MOHindsight:
    def __init__(self, name, env, agent, score_weight=None, multi_objective=True, weight_samples=True, test_freq=1,
                 epochs=20001, device='cuda'):
        self.name = name
        self.env = env
        self.agent = agent

        self.multi_objective = multi_objective
        self.weight_samples = weight_samples
        self.test_freq = test_freq
        self.epochs = epochs
        self.device = device

        self.weight_size = env.reward_dim if self.multi_objective else 0
        self.score_weight = env.score_weight
        self.models_path = os.path.join('log/weights', name, name)
        self.results_path = os.path.join('log/results', name + '.csv')
        self.graphs_path = os.path.join('log/graphs', name)

    def learn(self):

        scores = []
        test_signal_totals = []
        default_weights = get_default_weights(self.weight_size)
        test_totals = [[] for _ in range(len(default_weights))]
        global_step = 0
        start_time = time.time()

        with open(self.results_path, "w", 1) as file:
            csv_writer = csv.writer(file, delimiter=",")
            for e in range(self.epochs):
                done = False
                state = self.env.reset()
                score = 0
                episode_len = 0
                signal_totals = [0 for _ in range(self.weight_size)]
                reward_weights = generate_weights(alpha=self.agent.alpha)

                self.agent.train_mode()
                while not done:
                    action = self.agent.get_action(state, weights=reward_weights)

                    next_state, reward, done, signals = self.env.step(action)
                    # self.env.render()

                    self.agent.append_memory(state, action, reward, next_state, done, signals)

                    signal_totals = [sum(x) for x in zip(signal_totals, signals)]
                    score += reward
                    episode_len += 1
                    state = next_state

                    global_step += 1
                    if done:
                        break

                self.agent.log_episode(e)

                scores.append(score)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                print('Ep: {} step: {} weights: {} signals: {} score: {} time: {}:{}:{} alpha: {}'.format(
                    e, global_step, reward_weights, signal_totals, score, h, m, s, self.agent.alpha))

                if self.multi_objective and e % self.test_freq == 0:
                    test_signal_totals = []
                    self.agent.eval_mode()
                    for test_weights in default_weights:
                        done_test = False
                        state_test = self.env.reset()
                        total_reward = 0

                        while not done_test:
                            action = self.agent.get_action(state_test, weights=test_weights)
                            state_test, reward, done, signals = self.env.step(action)
                            total_reward += np.sum(np.asarray(signals) * test_weights)
                            if done:
                                break
                        test_signal_totals.append(total_reward)

                    for i in range(len(test_totals)):
                        test_totals[i].append(test_signal_totals[i])
                    visualize(self.graphs_path, test_totals, labels=default_weights)
                elif not self.multi_objective:
                    visualize(self.graphs_path, scores)

                csv_writer.writerow([e, global_step, reward_weights, signal_totals, score, h, m, s, test_signal_totals, episode_len, self.agent.alpha])
