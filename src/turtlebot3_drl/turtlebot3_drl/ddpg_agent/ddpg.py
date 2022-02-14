import numpy
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn


# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py


ACTION_LINEAR_MAX = 0.22
ACTION_ANGULAR_MAX = 2.0

class Actor(nn.Module):
    def __init__(self, name, state_size, action_size, action_limit_v, action_limit_w,):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.name = name

        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w

        self.fa1 = nn.Linear(state_size, 512)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)

        self.fa2 = nn.Linear(512, 512)
        nn.init.xavier_uniform_(self.fa2.weight)
        self.fa2.bias.data.fill_(0.01)

        self.fa3 = nn.Linear(512, action_size)
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

        self.fc1 = nn.Linear(state_size, 256)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        self.fa1 = nn.Linear(action_size, 256)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)

        self.fca1 = nn.Linear(512, 512)
        nn.init.xavier_uniform_(self.fca1.weight)
        self.fca1.bias.data.fill_(0.01)

        self.fca2 = nn.Linear(512, 1)
        nn.init.xavier_uniform_(self.fca2.weight)
        self.fca2.bias.data.fill_(0.01)

    def forward(self, states, actions):
        xs = torch.relu(self.fc1(states))
        xa = torch.relu(self.fa1(actions))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.fca1(x))
        vs = self.fca2(x)
        return vs

class DDPG():
    def __init__(self, device, state_size, action_size, discount_factor, learning_rate, tau, actor_noise):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.tau = tau
        self.actor_noise = actor_noise

        self.actor = Actor("actor", self.state_size, self.action_size, ACTION_LINEAR_MAX, ACTION_ANGULAR_MAX)
        self.target_actor = Actor("target_actor", self.state_size, self.action_size, ACTION_LINEAR_MAX, ACTION_ANGULAR_MAX)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)

        self.critic = Critic("critic", self.state_size, self.action_size)
        self.target_critic = Critic("target_critic", self.state_size, self.action_size)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)

        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

    def get_action(self, state, step):
        state = torch.from_numpy(numpy.asarray(state, numpy.float32))
        action = self.actor.forward(state).detach().data.numpy().tolist()
        N = copy.deepcopy(self.actor_noise.get_noise(t=step))
        N[0] = N[0]*ACTION_LINEAR_MAX/2
        N[1] = N[1]*ACTION_ANGULAR_MAX
        action[0] = numpy.clip(action[0] + N[0], 0., ACTION_LINEAR_MAX)
        action[1] = numpy.clip(action[1] + N[1], -ACTION_ANGULAR_MAX, ACTION_ANGULAR_MAX)
        return action

    def train(self, batch):

        s_sample, a_sample, r_sample, new_s_sample, done_sample = batch

        s_sample = torch.from_numpy(s_sample)
        a_sample = torch.from_numpy(a_sample)
        r_sample = torch.from_numpy(r_sample)
        new_s_sample = torch.from_numpy(new_s_sample)
        done_sample = torch.from_numpy(done_sample)

        # optimize critic
        a_target = self.target_actor.forward(new_s_sample).detach()
        next_value = torch.squeeze(self.target_critic.forward(new_s_sample, a_target).detach())
        y_expected = r_sample + (1 - done_sample)*self.discount_factor*next_value               # y_exp = r _ gamma*Q'(s', P'(s'))
        y_predicted = torch.squeeze(self.critic.forward(s_sample, a_sample))                    # y_pred = Q(s,a)
        self.qvalue = y_predicted.detach()

        #TODO: use l1 loss or MSE or other?
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # optimize actor
        pred_a_sample = self.actor.forward(s_sample)
        loss_actor = -1*torch.sum(self.critic.forward(s_sample, pred_a_sample))

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # Soft update all target networks
        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)
        return [loss_critic.detach(), loss_actor.detach()]

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)        