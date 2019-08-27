import random
from collections import deque
import itertools
from torch.utils.data import Dataset


class ExperienceDataset(Dataset):
    def __init__(self, experience):
        super(ExperienceDataset, self).__init__()
        self._exp = []
        for x in experience:
            self._exp.append(x)

    def __getitem__(self, index):
        return self._exp[index]

    def __len__(self):
        return len(self._exp)

    def get_testset(self, percent):
        random.shuffle(self._exp)
        testset = ExperienceDataset(self._exp[:int(len(self._exp) * percent)])
        self._exp = self._exp[int(len(self._exp) * percent):]
        return testset


class RandomReplay:
    def __init__(self, capacity, epsilon=.01, alpha=.6):
        self._memory = deque(maxlen=capacity)
        self.max_error = 10000
        self.next_read = 0

    def extend(self, data):
        if isinstance(data, tuple):
            self._memory.append(data)
        else:
            for step in data:
                if isinstance(step, tuple):
                    self._memory.append(step)
                else:
                    raise Exception("Invalid data added to replay buffer.")

    def sample(self, n):
        # n = min(n, len(self._memory))
        # sample = list(itertools.islice(self._memory, self.next_read, len(self._memory) - 1))
        # self.next_read += len(sample)
        # print(len(self._memory), n - len(sample))
        # sample = sample + random.sample(self._memory, n - len(sample))

        return ExperienceDataset(random.sample(self._memory, n))

    def __len__(self):
        return len(self._memory)
