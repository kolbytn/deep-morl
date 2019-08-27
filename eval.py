import time
import os
import numpy as np
import random

from env.lava_env import LavaEnv
from env.safety_env import SafetyEnv
from env.door_env import DoorEnv
from agent.dqn import DQNAgent
from rl_util.functions import get_default_weights


test_name = '19_07_26_small_position0'
episode = 17000
models_path = os.path.join('log/weights', test_name, test_name)
device = 'cuda'
multi_objective = True
epochs = 10
env_name = 'safety'
env_size = 'small'


if __name__ == '__main__':
    if env_name == 'door':
        env = DoorEnv()
    elif env_name == 'safety':
        env = SafetyEnv(size=env_size)
    elif env_name == 'lava':
        env = LavaEnv()
    else:
        raise Exception("Invalid environment type")
    weight_size = env.reward_dim if multi_objective else 0

    agent = DQNAgent(env.state_dim, env.action_dim, weight_size, models_path, train_multi_objective=multi_objective, device=device,
                     load_model=True, load_episode=episode)
            
    # weights = get_default_weights(env.reward_dim) if multi_objective else [[1 / env.reward_dim for _ in range(env.reward_dim)]]
    weights = [np.array([0, 0, 1, 0]), np.array([0, .2, .8, 0]), np.array([0, .4, .6, 0]), np.array([0, .6, .4, 0]), 
               np.array([0, .8, .2, 0]), np.array([0, 1, 0, 0])]
    for w in weights:
        print(w)
        signal_avg = [0 for _ in range(env.reward_dim)]
        for e in range(epochs):
            done = False
            state = env.reset()
            score = 0
            signal_totals = [0 for _ in range(env.reward_dim)]

            while not done:
                action = agent.get_action(state, weights=w)

                next_state, reward, done, signals = env.step(action)
                env.render()

                signal_totals = [sum(x) for x in zip(signal_totals, signals)]
                score += sum(np.asarray(signals) * w)
                state = next_state

                if done:
                    break
            
            signal_avg = [sum(x) for x in zip(signal_totals, signal_avg)]
            print('weights: {} signals: {} score: {}'.format(
                w, signal_totals, score))

        signal_avg = [x / epochs for x in signal_avg]
        print('average signals: {}'.format(signal_avg))
