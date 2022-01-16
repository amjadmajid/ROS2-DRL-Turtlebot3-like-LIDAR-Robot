import torch
import torch.nn as nn

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py


class Actor(nn.Module):
    def __init__(self, name, state_size, action_size, action_limit_v, action_limit_w,):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.name = name

        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w

        self.fa1 = nn.Linear(state_size, 500)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)

        self.fa2 = nn.Linear(500, 500)
        nn.init.xavier_uniform_(self.fa2.weight)
        self.fa2.bias.data.fill_(0.01)

        self.fa3 = nn.Linear(500, action_size)
        nn.init.xavier_uniform_(self.fa3.weight)
        self.fa3.bias.data.fill_(0.01)

    def forward(self, states):
        x = torch.relu(self.fa1(states))
        x = torch.relu(self.fa2(x))
        action = self.fa3(x)
        if states.shape <= torch.Size([self.state_size]):
            action[0] = torch.sigmoid(action[0])*self.action_limit_v
            action[1] = torch.tanh(action[1])*self.action_limit_w
        else:
            action[:, 0] = torch.sigmoid(action[:, 0])*self.action_limit_v
            action[:, 1] = torch.tanh(action[:, 1])*self.action_limit_w
        return action


class Critic(nn.Module):

    def __init__(self, name, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.name = name

        self.fc1 = nn.Linear(state_size, 500)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        self.fca1 = nn.Linear(500 + action_size, 500)
        nn.init.xavier_uniform_(self.fca1.weight)
        self.fca1.bias.data.fill_(0.01)

        self.fca2 = nn.Linear(500, 1)
        nn.init.xavier_uniform_(self.fca2.weight)
        self.fca2.bias.data.fill_(0.01)

    def forward(self, states, actions):
        x_states = torch.relu(self.fc1(states))
        x_merged = torch.cat((x_states, actions), dim=1)
        x = torch.relu(self.fca1(x_merged))
        critic_output = self.fca2(x)
        return critic_output
