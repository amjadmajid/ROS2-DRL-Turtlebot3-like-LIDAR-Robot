import torch
import torch.nn as nn
from torch.distributions import Normal

import math

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py


class Actor(nn.Module):
    def __init__(self, name, state_size, action_size, action_limit_v, action_limit_w, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.name = name

        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # linear 1
        self.fa1 = nn.Linear(state_size, 500)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0)
        #TODO: what should initial bias be
        # linear 2
        self.fa2 = nn.Linear(500, 500)
        nn.init.xavier_uniform_(self.fa2.weight)
        self.fa2.bias.data.fill_(0)

        self.mean_linear = nn.Linear(500, action_size)
        nn.init.xavier_uniform_(self.mean_linear.weight)
        self.mean_linear.bias.data.fill_(0)

        self.log_std_linear = nn.Linear(500, action_size)
        nn.init.xavier_uniform_(self.log_std_linear.weight)
        self.log_std_linear.bias.data.fill_(0)

    def forward(self, state):
        x = torch.relu(self.fa1(state))
        x = torch.relu(self.fa2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std
        # if state.shape <= torch.Size([self.state_size]):
        #     action[0] = torch.sigmoid(action[0])*self.action_limit_v
        #     action[1] = torc  h.tanh(action[1])*self.action_limit_w
        # else:
        #     action[:, 0] = torch.sigmoid(action[:, 0])*self.action_limit_v
        #     action[:, 1] = torch.tanh(action[:, 1])*self.action_limit_w
        return action

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.unsqueeze(log_prob, 0)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std


class Critic(nn.Module):

    def __init__(self, name, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.name = name

        # first critic
        self.q1_fc1 = nn.Linear(state_size, 500)
        nn.init.xavier_uniform_(self.q1_fc1.weight)
        self.q1_fc1.bias.data.fill_(0)

        self.q1_fca1 = nn.Linear(500 + action_size, 500)
        nn.init.xavier_uniform_(self.q1_fca1.weight)
        self.q1_fca1.bias.data.fill_(0)

        self.q1_fca2 = nn.Linear(500, 1)
        nn.init.xavier_uniform_(self.q1_fca2.weight)
        self.q1_fca2.bias.data.fill_(0)

        # second critic
        self.q2_fc1 = nn.Linear(state_size, 500)
        nn.init.xavier_uniform_(self.q2_fc1.weight)
        self.q2_fc1.bias.data.fill_(0)

        self.q2_fca1 = nn.Linear(500 + action_size, 500)
        nn.init.xavier_uniform_(self.q2_fca1.weight)
        self.q2_fca1.bias.data.fill_(0)

        self.q2_fca2 = nn.Linear(500, 1)
        nn.init.xavier_uniform_(self.q2_fca2.weight)
        self.q2_fca2.bias.data.fill_(0)


    def forward(self, states, actions):
        q1_states = torch.relu(self.q1_fc1(states))
        q1_merged = torch.cat((q1_states, actions), dim=1)
        q1_x = torch.relu(self.q1_fca1(q1_merged))
        q1_output = self.q1_fca2(q1_x)

        q2_states = torch.relu(self.q2_fc1(states))
        q2_merged = torch.cat((q2_states, actions), dim=1)
        q2_x = torch.relu(self.q2_fca1(q2_merged))
        q2_output = self.q2_fca2(q2_x)

        return q1_output, q2_output
