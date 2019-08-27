import numpy as np
import random
import math
import os
from collections import deque
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rl_util.replay import RandomReplay
from rl_util.functions import generate_weights, compare_models
from rl_util.networks import LinearNetwork, DecoupledNetwork


class DQNAgent:
    def __init__(self, state_size, action_size, weight_size, dir_path, train_multi_objective=False, device='cuda',
                 load_model=False, load_episode=0, weight_samples=True, model_type='uvfa'):
        self.dir_path = dir_path
        self.load_model = load_model
        self.load_episode = load_episode
        self.device = device
        self.weight_samples = weight_samples
        self.state_size = state_size
        self.action_size = action_size
        self.weight_size = weight_size
        self.train_multi_objective = train_multi_objective
        self.alpha = np.asarray([1. for _ in range(weight_size)])
        self.max_variance = np.asarray([0 for _ in range(weight_size)])

        self.target_update = 10
        self.memory_augmentation = 5
        self.gamma = 0.99
        self.lr = 1e-4
        self.alpha_rate = 1e-2
        self.alpha_max = .5
        self.train_epochs = 2
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0
        self.save_epochs = 100
        self.batch_size = 32
        self.train_start = 500
        self.sample_size = 5000

        self.memory = RandomReplay(1000000)

        if model_type == 'uvfa':
            self.model = LinearNetwork(state_size, action_size, weight_size=weight_size, hidden_size=10).to(device)
            self.target_model = LinearNetwork(state_size, action_size, weight_size=weight_size, hidden_size=10).to(device)
        elif model_type == 'decoupled':
            self.model = DecoupledNetwork(state_size, action_size, weight_size=weight_size, hidden_size=10).to(device)
            self.target_model = DecoupledNetwork(state_size, action_size, weight_size=weight_size, hidden_size=10).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

        if self.load_model:
            self.model.load_state_dict(torch.load(self.dir_path + str(self.load_episode)))
            self.epsilon = self.epsilon * self.epsilon_decay ** self.load_episode
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def _update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def update_alpha(self, weights):
        alpha_target = weights * self.alpha_max + 1
        self.alpha += (alpha_target - self.alpha) * self.alpha_rate

    def eval_mode(self):
        self.model.eval()

    def train_mode(self):
        self.model.train()

    def get_action(self, state, weights=None):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_input = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            if self.train_multi_objective:
                if weights is None:
                    weights = np.full(self.weight_size, 1 / self.weight_size)
                weight_input = torch.from_numpy(weights).float().to(self.device).unsqueeze(0)
            else:
                weight_input = None

            q_values = self.model(state_input, weight_input)
            return torch.argmax(q_values.squeeze(0)).item()

    def append_memory(self, state, action, reward, next_state, done, signals):
        if self.train_multi_objective:
            for _ in range(self.memory_augmentation):
                weights = generate_weights(alpha=self.alpha)
                reward_vector = np.asarray(signals)
                reward_vector = reward_vector * weights
                reward = np.sum(reward_vector)
                self.memory.extend((state, action, reward, next_state, done, weights))
        else:
            self.memory.extend((state, action, reward, next_state, done, 0))

    def log_episode(self, n):
        if n % self.target_update == 0:
            self._update_target()

        if len(self.memory) >= self.train_start:
            self.train_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if n % self.save_epochs == 0:
            torch.save(self.model.state_dict(), self.dir_path + str(n))
            # to_compare = LinearNetwork(self.state_size, self.action_size, weight_size=self.weight_size, hidden_size=32).to(self.device)
            # to_compare.load_state_dict(torch.load(self.dir_path + str(n)))
            # compare_models(self.model, to_compare)

    def train_model(self):
        dataset = self.memory.sample(self.sample_size)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        total_loss = 0
        total_weights = np.zeros(self.weight_size)
        for _ in range(self.train_epochs):
            for state, action, reward, next_state, done, weights in loader:
                state = state.float().to(self.device)
                action = action.to(self.device)
                reward = reward.float().to(self.device)
                next_state = next_state.float().to(self.device)
                done = done.float().to(self.device)
                weights = weights.float().to(self.device) if self.train_multi_objective else None

                self.optimizer.zero_grad()

                model_out = self.model(state, weights)
                model_q = model_out.gather(1, action.unsqueeze(1)).squeeze(1)
                target_out = self.target_model(next_state, weights)
                target_out = torch.max(target_out, 1)[0]
                target_q = reward + self.gamma * target_out * (1 - done)

                losses = (target_q - model_q) ** 2
                loss = torch.mean(losses)

                loss.backward()
                self.optimizer.step()

                if self.weight_samples:
                    total_loss += torch.sum(losses).item()
                    total_weights = total_weights + torch.sum(weights * losses.unsqueeze(-1).expand(weights.shape), dim=0).detach().cpu().numpy()
        
        if self.weight_samples:
            self.update_alpha(total_weights / total_loss)
