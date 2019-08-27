import torch
import torch.nn as nn


class LinearNetwork(nn.Module):
    def __init__(self, state_size, action_size, weight_size=0, hidden_size=10):
        super().__init__()
        interval = 1
        if isinstance(state_size, tuple):
            if len(state_size) > 2:
                interval = state_size[2]
            state_size = torch.prod(torch.tensor(state_size))

        self.shared_layers = nn.Sequential(nn.Linear(state_size, hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(hidden_size, hidden_size),
                                           nn.ReLU())

        self.weighted_layers = nn.Sequential(nn.Linear(hidden_size + weight_size, hidden_size),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, action_size))

    def forward(self, state, weight=None):
        state = torch.flatten(state, start_dim=1)
        out = self.shared_layers(state)
        if weight is None:
            linear_in = out
        else:
            linear_in = torch.cat((out, weight), dim=-1)

        return self.weighted_layers(linear_in)


class DecoupledNetwork(nn.Module):
    def __init__(self, state_size, action_size, weight_size=0, hidden_size=10):
        super().__init__()
        interval = 1
        self.action_size = action_size
        self.weight_size = weight_size
        if isinstance(state_size, tuple):
            if len(state_size) > 2:
                interval = state_size[2]
            state_size = torch.prod(torch.tensor(state_size))

        self.state_layers = nn.Sequential(nn.Linear(state_size, hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(hidden_size, hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(hidden_size, weight_size * action_size))

        self.weight_layers = nn.Sequential(nn.Linear(weight_size, weight_size),
                                             nn.ReLU(),
                                             nn.Linear(weight_size, weight_size),
                                             nn.ReLU(),
                                             nn.Linear(weight_size, weight_size))

    def forward(self, state, weight=None):
        state = torch.flatten(state, start_dim=1)
        state_rep = self.state_layers(state)
        if weight is None:
            return out
        else:
            state_rep = state_rep.view((-1, self.action_size, self.weight_size))
            weight_rep = self.weight_layers(weight)
            return torch.bmm(state_rep, weight_rep.unsqueeze(-1)).squeeze(-1)
