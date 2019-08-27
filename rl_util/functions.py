import random
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch

def generate_weights(n=None, alpha=None):
    if alpha is None:
        alpha = [1 for _ in range(n)]
    weights = np.random.dirichlet(alpha)
    return weights


def get_default_weights(n):
    weights = [np.array([1/n for _ in range(n)])]
    for i in range(n):
        weights.append(np.array([int(j == i) for j in range(n)]))

    # weights = []
    # for i in range(1, 2 ** n):
    #     binary_list = [int(x) for x in list('{0:0b}'.format(i))]
    #     padding_len = n - len(binary_list)
    #     binary_list = [0 for _ in range(padding_len)] + binary_list
    #     reward_weights = np.asarray(binary_list)
    #     reward_weights = reward_weights / np.sum(reward_weights)
    #     weights.append(reward_weights)

    return weights


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


def visualize(path, data, smooth=10, labels=None):
    plt.clf()
    if isinstance(data[0], list):
        for i, d in enumerate(data):
            if smooth > 0:
                d = [sum(d[x:x + smooth]) / smooth for x in range(0, len(d) - smooth)]

            if labels is None:
                plt.plot(d)
            else: 
                plt.plot(d, label=labels[i])

        if labels is not None:
            plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    else:
        if smooth > 0:
            data = [sum(data[x:x + smooth]) / smooth for x in range(0, len(data) - smooth)]
        plt.plot(data)
    plt.savefig(path, bbox_inches="tight")
