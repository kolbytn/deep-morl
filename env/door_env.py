import gym_minigrid
import gym
import numpy as np

class DoorEnv:
    def __init__(self, max_length=100, score_weight=None):
        self.env = gym.make('MiniGrid-GoToDoor-5x5-v0')
        self.max_length = max_length
        self.length = 0
        self.goal_colors = ['red', 'green', 'blue']
        self.score_weight = score_weight
        if self.score_weight is None:
            self.score_weight = [1 / self.reward_dim for _ in range(self.reward_dim)]

        self.doors = dict()

    @property
    def action_dim(self):
        return 3

    @property
    def state_dim(self):
        return (7, 7, 3)

    @property
    def reward_dim(self):
        return 3

    def render(self):
        self.env.render()

    def reset(self):
        obs = self.env.reset()
        self.length = 0

        self.doors = dict()
        for i in range(self.env.grid.width):
            for j in range(self.env.grid.height):
                c = self.env.grid.get(i, j)
                if c is not None and c.type == 'door':
                    self.doors[c.color] = (i, j)

        return obs['image']

    def step(self, action):
        obs, _, _, _ = self.env.step(action)

        curr_pos = tuple(self.env.agent_pos)

        for door in self.doors.values():
            if curr_pos[0] == door[0] and curr_pos[1] - door[1] == 1:
                door_dir = 1
            elif curr_pos[0] == door[0] and curr_pos[1] - door[1] == -1:
                door_dir = 3
            elif curr_pos[0] - door[0] == 1 and curr_pos[1] == door[1]:
                door_dir = 2
            elif curr_pos[0] - door[0] == -1 and curr_pos[1] == door[1]:
                door_dir = 0
            else:
                door_dir = -1

            if door_dir == self.env.agent_dir and action == 2:
                curr_pos = door

        self.length += 1
        if self.length > self.max_length or curr_pos in self.doors.values():
            done = True
        else:
            done = False
        signals = [int(curr_pos == self.doors[color]) if color in self.doors else 0 for color in self.goal_colors]

        return obs['image'], sum([x * y for x, y in zip(signals, self.score_weight)]), done, signals

    def get_state_id(self):
        return 0

if __name__ == "__main__":
    env = DoorEnv()
    obs = env.reset()

    print(obs['image'].shape)

    for _ in range(10):
        env.render()
        obs, reward, done, info = env.step(env.env.action_space.sample()) # take a random action

    print("obs", obs)
    print("reward", reward)
    print("done", done)
    print("info", info)
