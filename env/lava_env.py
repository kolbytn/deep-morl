import gym_minigrid
import gym
import numpy as np

class LavaEnv:
    def __init__(self, max_length=100, score_weight=[.5, .5]):
        self.env = gym.make('MiniGrid-DistShift1-v0')
        self.max_length = max_length
        self.length = 0
        self.score_weight = score_weight

        self.lava_pos = []
        for i in range(self.env.grid.width):
            for j in range(self.env.grid.height):
                c = self.env.grid.get(i, j)
                if c is not None and c.type == 'lava':
                    self.lava_pos.append((i, j))
                elif c is not None and c.type == 'goal':
                    self.goal_pos = (i, j)

    @property
    def action_dim(self):
        return 3

    @property
    def state_dim(self):
        return (7, 7, 3)

    @property
    def reward_dim(self):
        return 2

    def render(self):
        self.env.render()

    def reset(self):
        obs = self.env.reset()
        self.length = 0

        return obs['image']

    def step(self, action):
        obs, _, _, _ = self.env.step(action)

        curr_pos = self.env.agent_pos
        min_lava_dist = self.env.grid.width + self.env.grid.height
        for lava in self.lava_pos:
            lava_dist = abs(lava[0] - curr_pos[0]) + abs(lava[1] - curr_pos[1])
            min_lava_dist = min(min_lava_dist, lava_dist)
        
        goal_dist = abs(curr_pos[0] - self.goal_pos[0]) + abs(curr_pos[1] - self.goal_pos[1])

        self.length += 1

        if min_lava_dist == 0:
            done = True
            goal_signal = 0
            lava_signal = -7
        elif goal_dist == 0:
            done = True
            goal_signal = 7
            lava_signal = min_lava_dist * .01
        elif self.length > self.max_length:
            done = True
            goal_signal = 0
            lava_signal = min_lava_dist * .01
        else:
            done = False
            goal_signal = 0
            lava_signal =  min_lava_dist * .01

        return obs['image'], goal_signal * self.score_weight[0] + lava_signal * self.score_weight[1], done, [goal_signal, lava_signal]


if __name__ == "__main__":
    env = LavaEnv()
    obs = env.reset()

    print(obs['image'].shape)

    for _ in range(10):
        env.render()
        obs, reward, done, info = env.step(env.env.action_space.sample()) # take a random action

    print("obs", obs)
    print("reward", reward)
    print("done", done)
    print("info", info)
