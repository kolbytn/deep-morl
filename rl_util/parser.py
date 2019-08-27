import random


class SpecLanguage:
    '''
        Examples:
        - o1 & ( o2 | ( o3 >= 1 ) )
        o2 | ( o2 >= 10 & ( - o1 ) )
        o1 | ( o2 | ( o1 & ( - o2 ) ) )
    '''
    def __init__(self, objective_count, min_op=False, curriculum=False):
        self.tokens = []
        self.reward_vec = []
        self.objective_count = objective_count
        self.min_op = min_op
        self.test_specs = []
        self.max_curriculum = not curriculum
        self.curr_curriculum = 0 if curriculum else self.objective_count

    def set_test_specs(self, test_specs):
        self.test_specs = test_specs

    def inc_curriculum(self):
        if not self.max_curriculum:
            self.curr_curriculum += 1
        if self.curr_curriculum == self.objective_count:
            self.max_curriculum = True

    def get_level(self):
        return random.randint(0, self.curr_curriculum)

    def generate(self, curriculum=None):
        spec = ''
        level = self.get_level() if curriculum is None else curriculum
        objective = random.randint(1, self.objective_count)

        constraint = random.random()
        c_value = random.sample(self.objective_max[objective], 1)[0]
        
        if self.min_op:
            if constraint < .25:
                spec += 'o' + str(objective)
            elif constraint < .5:
                spec += '- o' + str(objective)
            elif constraint < .75:
                spec += 'o' + str(objective) + ' >= ' + str(c_value)
            else:
                spec += 'o' + str(objective) + ' <= ' + str(c_value)
        else:
            if constraint < .5:
                spec += 'o' + str(objective)
            elif constraint < .75:
                spec += 'o' + str(objective) + ' >= ' + str(c_value)

        if random.random() < level / self.objective_count:
            if random.random() < .5:
                spec += ' & ( '
            else:
                spec += ' | ( '
            spec += self.generate(level - 1) + ' )'
        
        # while curriculum is None and spec in self.test_specs:
        #     spec = self.generate()
        return spec

    def parse(self, spec, reward_vec):
        self.tokens = spec.split(' ')
        self.reward_vec = reward_vec
        return self.get_value(self.tokens)

    def get_value(self, spec):
        curr = 0

        left, curr = self.get_side(spec, curr)

        if curr < len(spec):
            operation = spec[curr]
            curr += 1
            right, curr = self.get_side(spec, curr)

            if operation == '&':
                value = min(left, right)
            elif operation == '|':
                value = max(left, right)
            else:
                raise Exception("Unexpected token:", operation)

        else:
            value = left

        if curr < len(spec):
            raise Exception("Unexpected token:", spec[curr])

        return value

    def get_side(self, spec, curr):
        value = 0
        objective = None

        if spec[curr] == '(':
            paren_count = 0
            for i, t in enumerate(spec[curr+1:]):
                if t == '(':
                    paren_count += 1
                elif t == ')' and paren_count > 0:
                    paren_count -= 1
                elif t == ')' and paren_count == 0:
                    value = self.get_value(spec[curr+1:curr+i+1])
                    curr += i + 2

        elif spec[curr] == '-':
            curr += 1
            objective = int(spec[curr][-1]) - 1
            value = self.reward_vec[objective]
            if self.scaled:
                value = value / self.objective_max[objective] if self.objective_max[objective] > 0 else value
                value = 1 - value
            else:
                value = self.objective_max[objective] - value
            curr += 1

        elif spec[curr][0] == 'o':
            objective = int(spec[curr][-1]) - 1
            value = self.reward_vec[objective]
            value = value / self.objective_max[objective] if self.scaled and self.objective_max[objective] > 0 else value
            curr += 1

            if curr < len(spec) and spec[curr] == '>=':
                value = int(value >= float(spec[curr+1]))
                curr += 2
            elif curr < len(spec) and spec[curr] == '<=':
                value = int(value <= float(spec[curr+1]))
                curr += 2

        else:
            raise Exception("Unexpected token:", spec[curr])

        return value, curr


if __name__ == '__main__':
    lang = SpecLanguage(3, objective_max=[5, 5, 100, 200], scaled=True)
    reward_vec = [4, 1, 98, 2]
    print(reward_vec)
    spec = 'o3 & ( o3 >= 35 & ( o3 | ( - o1 ) ) )'
    value, _ = lang.parse(spec, reward_vec)

    for i in range(0, 10):
        spec = lang.generate(i)
        print(spec)
        value, _ = lang.parse(spec, reward_vec)
        print(value)
