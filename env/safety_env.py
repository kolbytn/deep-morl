from gym_minigrid.minigrid import *
import random
import itertools
import sys
import termios
import tty
import os
import time


class SmallSafetyEnv(MiniGridEnv):
    def __init__(self, width=9, height=7, agent_start_pos=(1,1), agent_start_dir=0, strip2_row=2, max_length=100):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = (width-2, 1)
        self.strip2_row = strip2_row
        
        super().__init__(
            width=width,
            height=height,
            max_steps=max_length,
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.set(*self.goal_pos, Goal())

        self.lava_pos = []
        for i in range(self.width - 6):
            self.grid.set(3+i, 1, Lava())
            self.grid.set(3+i, self.strip2_row, Lava())
            self.lava_pos.append((3+i, 1))
            self.lava_pos.append((3+i, self.strip2_row))

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


class MediumSafetyEnv(MiniGridEnv):
    def __init__(self, width=9, height=13, goal_pos=(7,4), agent_start_pos=(1,5), agent_start_dir=0, max_length=150):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos
        
        super().__init__(
            width=width,
            height=height,
            max_steps=max_length,
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.set(*self.goal_pos, Goal())

        self.lava_pos = []
        for i in range(3, width - 3):
            self.grid.set(i, 1, Lava())
            self.lava_pos.append((i, 1))

        obstacles = [(3, 3), (4, 3), (5, 3), (3, 4), 
                     (5, 5), (3, 6), (4, 6), (3, 7),
                     (3, 8)]
        for i, j in obstacles:
            self.grid.set(i, j, Wall())

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


class LargeSafetyEnv(MiniGridEnv):
    def __init__(self, width=13, height=17, goal_pos=(10,6), agent_start_pos=(2,6), agent_start_dir=0, max_length=200):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = goal_pos
        
        super().__init__(
            width=width,
            height=height,
            max_steps=max_length,
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.set(*self.goal_pos, Goal())

        self.lava_pos = []
        for i in range(4, width - 4):
            self.grid.set(i, 1, Lava())
            self.lava_pos.append((i, 1))

        obstacles = [(4, 7), (6, 6), (8, 7)]
        for i, j in obstacles:
            self.grid.set(i, j, Wall())
        self.grid.wall_rect(4, 3, 5, 3)
        self.grid.wall_rect(4, 8, 5, 5)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


class SafetyEnv:
    def __init__(self, size='small', max_length=200, lava_obj=True, obstable_obj=True, time_obj=True, score_weight=None,
                 goal_scale=200, lava_scale=1, obstacle_scale=1, time_scale=1, transition=.8):
        self.lava_obj = lava_obj
        self.obstacle_obj = obstable_obj
        self.time_obj = time_obj
        self.lava_scale = lava_scale
        self.obstacle_scale = obstacle_scale
        self.time_scale = time_scale
        self.goal_scale = goal_scale
        self.transition = transition

        if size == 'small':
            self.env = SmallSafetyEnv(max_length=max_length)
        elif size == 'medium':
            self.env = MediumSafetyEnv(max_length=max_length)
        elif size == 'large':
            self.env = LargeSafetyEnv(max_length=max_length)
        else:
            raise Exception("Invalid environment size")

        self.score_weight = [1 / self.reward_dim for _ in range(self.reward_dim)] if score_weight is None else score_weight
        self.max_lava_dist = 0
        for i in range(1, self.env.grid.width - 1):
            for j in range(1, self.env.grid.height - 1):
                if self.env.grid.get(i, j) is None:
                    self.max_lava_dist = max(self.max_lava_dist, self._get_lava_dist((i, j)))

    @property
    def action_dim(self):
        return 3

    @property
    def state_dim(self):
        max_dim = max(self.env.width, self.env.height)
        return (3)  # (max_dim, max_dim, 4)

    @property
    def reward_dim(self):
        return 1 + int(self.lava_obj) + int(self.obstacle_obj) + int(self.time_obj)

    def _get_lava_dist(self, position):
        min_lava_dist = self.env.grid.width + self.env.grid.height
        for lava in self.env.lava_pos:
            lava_dist = abs(lava[0] - position[0]) + abs(lava[1] - position[1])
            min_lava_dist = min(min_lava_dist, lava_dist)
        return min_lava_dist

    def _get_goal_dist(self):
        return abs(self.env.agent_pos[0] - self.env.goal_pos[0]) + abs(self.env.agent_pos[1] - self.env.goal_pos[1])

    def _is_facing_obj(self, obj='wall'):
        for direction, offset in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
            cell = self.env.grid.get(self.env.agent_pos[0] + offset[0], self.env.agent_pos[1] + offset[1])
            if cell is not None and cell.type == obj and self.env.agent_dir == direction:
                return True
        return False

    def get_loc_dir(self):
        return np.asarray([self.env.agent_pos[0], self.env.agent_pos[1], self.env.agent_dir])

    def _get_full_state(self):
        left_padding = int(max(self.env.height - self.env.width, 0) / 2)
        right_padding = left_padding if max(self.env.height - self.env.width, 0) % 2 == 0 else left_padding + 1
        top_padding = int(max(self.env.width - self.env.height, 0) / 2)
        bottom_padding = top_padding if max(self.env.width - self.env.height, 0) % 2 == 0 else top_padding + 1

        state = np.zeros(self.state_dim, dtype='uint8')
        for i in range(left_padding, self.env.width - right_padding):
            for j in range(top_padding, self.env.height - bottom_padding):
                cell = self.env.grid.get(i, j)

                if cell is not None:
                    if cell.type == 'wall':
                        state[i, j, 0] = 1
                    elif cell.type == 'lava':
                        state[i, j, 1] = 1
                    elif cell.type == 'goal':
                        state[i, j, 2] = 1
                    if i == self.env.agent_pos[0] and j == self.env.agent_pos[1]:
                        forward = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                        left = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                        right = [(0, -1), (-1, 0), (0, 1), (1, 0)]
                        f = forward[self.env.agent_dir]
                        l = left[self.env.agent_dir]
                        r = right[self.env.agent_dir]
                        state[i, j] = 1
                        state[i + f[0], j + f[1], 3] = 1
                        state[i + l[0], j + l[1], 3] = 1
                        state[i + r[0], j + r[1], 3] = 1

        # grid = self.env.grid.encode()
        # grid = np.delete(grid, 1, 2)
        # grid[self.env.agent_pos[0], self.env.agent_pos[1], 1] = self.env.agent_dir
        # state = np.pad(grid, ((left_padding, right_padding), (top_padding, bottom_padding), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))

        return state

    def render(self):
        self.env.render()

    def reset(self):
        obs = self.env.reset()

        return self.get_loc_dir()

    def step(self, action):
        if random.random() > self.transition:
            action_list = list(range(self.action_dim))
            action_list.remove(action)
            action = random.choice(action_list)
        obs, _, done, _ = self.env.step(action)

        reward_vec = [int(self._get_goal_dist() == 0) * self.goal_scale]
        
        if self.lava_obj:
            reward_vec.append(self._get_lava_dist(self.env.agent_pos) * self.lava_scale / self.max_lava_dist)
        if self.obstacle_obj:
            reward_vec.append(-int(self._is_facing_obj and action == self.env.actions.forward) * self.obstacle_scale)
        if self.time_obj:
            reward_vec.append(-self.time_scale)

        return self.get_loc_dir(), sum(x * y for x, y in zip(self.score_weight, reward_vec)), done, reward_vec


def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
 
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


if __name__ == "__main__":
    env = SafetyEnv(size='medium')

    while True:
        obs = env.reset()
        done = False
        while not done:
            env.render()
            char = getch()
        
            if char == 'w':
                action = 2
            elif char == 'a':
                action = 0
            elif char == 'd':
                action = 1
            else:
                print("Invalid")
                exit()

            obs, reward, done, info = env.step(action)
