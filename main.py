import time
import numpy as np
import csv
import os
import sys
import getopt

from env.lava_env import LavaEnv
from env.door_env import DoorEnv
from env.safety_env import SafetyEnv
from agent.dqn import DQNAgent
from algorithm.mo_hindsight import MOHindsight


multi_objective = True
weight_samples = True
env_name = 'safety'
env_size = 'small'

test_freq = 1
test_num = 10
epochs = 20001
device = 'cuda'


if __name__ == '__main__':
    name = '19_08_15_base'
    env_name = 'safety'
    model = 'uvfa'

    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:e:m:")
    except getopt.GetoptError:
        print("Invalid args")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-n':
            name = arg
            print(name)
        elif opt == '-e':
            env_name = arg
            print(env_name)
        elif opt == '-m':
            model = arg
            print(model)

    for test_count in range(test_num):

        test_name = name + str(test_count)
        models_path = os.path.join('log/weights', test_name, test_name)
        results_path = os.path.join('log/results', test_name + '.csv')
        graphs_path = os.path.join('log/graphs', test_name)
        if not os.path.exists(os.path.dirname(models_path)):
            os.makedirs(os.path.dirname(models_path))
        if not os.path.exists(os.path.dirname(results_path)):
            os.makedirs(os.path.dirname(results_path))
        if not os.path.exists(os.path.dirname(graphs_path)):
            os.makedirs(os.path.dirname(graphs_path))

        if env_name == 'door':
            env = DoorEnv()
        elif env_name == 'safety':
            env = SafetyEnv(size=env_size)
        elif env_name == 'lava':
            env = LavaEnv()
        else:
            raise Exception("Invalid environment type")

        weight_size = env.reward_dim if multi_objective else 0
        agent = DQNAgent(env.state_dim, env.action_dim, weight_size, models_path, model_type=model,
                         train_multi_objective=multi_objective, device=device, weight_samples=weight_samples)

        alg = MOHindsight(test_name, env, agent, multi_objective=multi_objective, weight_samples=weight_samples, test_freq=test_freq,
                          epochs=epochs, device=device)
        alg.learn()
