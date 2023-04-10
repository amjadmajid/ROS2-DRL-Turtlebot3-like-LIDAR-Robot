import numpy
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn


from ..turtlebot3_drl.common.visual import DrlVisual
from ..turtlebot3_drl.common.ounoise import OUNoise
from ..turtlebot3_drl.common.settings import ENABLE_STACKING
from ..turtlebot3_drl.drl_environment.drl_environment import NUM_SCAN_SAMPLES

LINEAR = 0
ANGULAR = 1

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py

class Actor(nn.Module):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.name = name
        self.iteration = 0

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=3)
        self.conv1_bn = nn.BatchNorm1d(32)
        self.conv1_mp = nn.MaxPool1d(3, 2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, stride=2)
        self.conv2_bn = nn.BatchNorm1d(32)
        self.conv2_mp = nn.MaxPool1d(3, 2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        # self.conv3_bn = nn.BatchNorm1d(16)
        self.flatten = nn.Flatten(1, 2)

        self.conv_sizes = self.get_conv_sizes(state_size)
        conv_out_size = self.conv_sizes[-1]
        print(f"conv out sizes: {self.conv_sizes}")

        self.fa1 = nn.Linear(conv_out_size + 4, hidden_size)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)

        self.fa2 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.fa2.weight)
        self.fa2.bias.data.fill_(0.01)

        self.fa3 = nn.Linear(hidden_size, action_size)
        nn.init.xavier_uniform_(self.fa3.weight)
        self.fa3.bias.data.fill_(0.01)

        self.visual = None

    def forward_conv(self, states, visualize=False, size=False):
        # batch_size, channels, length
        states = states.unsqueeze(1)
        c1 = torch.relu((self.conv1(states)))
        c1 = self.conv1_bn(c1)
        c2 = self.conv1_mp(c1)
        c3 = torch.relu((self.conv2(c2)))
        c3 = self.conv2_bn(c3)
        c4 = self.conv2_mp(c3)
        c5 = torch.relu((self.conv3(c4)))
        # c5 = self.conv3_bn(c5)
        c5 = self.flatten(c5)
        # if visualize:
        #     vc1 = self.flatten(c1)
        #     vc3 = self.flatten(c3)
            # self.visual.update_conv([vc1, vc3, c5])
        if size:
            # sc1 = self.flatten(c1)
            # sc3 = self.flatten(c3)
            # return [int(sc1.size(dim=1)), int(sc3.size(dim=1)), int(c5.size(dim=1))]
            return [int(c5.size(dim=1))]
        return c5

    def get_conv_sizes(self, input_size, device=None):
        if device is not None:
            return self.forward_conv(torch.zeros(1, input_size - 4).to(device), False, True)
        return self.forward_conv(torch.zeros(1, input_size - 4), False, True)

    def forward(self, states, visualize=False):
        laser_states = torch.narrow(states, 1, 0, NUM_SCAN_SAMPLES)
        other_states = torch.narrow(states, 1, NUM_SCAN_SAMPLES, 4)
        conv_output = self.forward_conv(laser_states, visualize)
        # print(f"a: {laser_states.shape}")
        # print(f"b: {other_states.shape}")
        # print(f"c: {conv_output.shape}")
        combined = torch.cat((conv_output, other_states), dim=1)
        # print(f"b: {combined.shape}")
        x1 = torch.relu(self.fa1(combined))
        x2 = torch.relu(self.fa2(x1))
        action = self.fa3(x2)

        if visualize:
            self.visual.update_hidden(states, action, [x1, x2])
            if self.iteration % 100:
                self.visual.update_bias([self.fa1.bias, self.fa2.bias])
            self.iteration += 1

        action[:, LINEAR] = torch.tanh(action[:, LINEAR])
        action[:, ANGULAR] = torch.tanh(action[:, ANGULAR])
        return action

class Critic(nn.Module):

    def __init__(self, name, state_size, action_size, hidden_size, conv_function, device):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        self.conv_function = conv_function
        self.device = device

        conv_out_size = self.get_conv_out_size(self.state_size)

        self.fc1 = nn.Linear(conv_out_size + 4, int(hidden_size / 2))
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        self.fa1 = nn.Linear(action_size, int(hidden_size / 2))
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)

        self.fca1 = nn.Linear(hidden_size, hidden_size)
        nn.init.xavier_uniform_(self.fca1.weight)
        self.fca1.bias.data.fill_(0.01)

        self.fca2 = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.fca2.weight)
        self.fca2.bias.data.fill_(0.01)

    def forward(self, states, actions):
        laser_states = torch.narrow(states, 1, 0, NUM_SCAN_SAMPLES)
        other_states = torch.narrow(states, 1, NUM_SCAN_SAMPLES, 4)
        conv_states = self.conv_function(laser_states)
        states = torch.cat((conv_states, other_states), dim=1)

        xs = torch.relu(self.fc1(states))
        xa = torch.relu(self.fa1(actions))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.fca1(x))
        vs = self.fca2(x)
        return vs

    def get_conv_out_size(self, input_size):
        c = self.conv_function(torch.zeros(1, input_size - 4).to(self.device))
        return int(c.size(dim=1))


class DDPGConv():
    def __init__(self, device, sim_speed):
        self.device = device

        self.batch_size      = 1024
        self.buffer_size     = 1000000
        self.state_size      = NUM_SCAN_SAMPLES + 4    # 10 laser readings
        self.action_size     = 2
        self.hidden_size     = 512
        self.discount_factor = 0.99
        self.learning_rate   = 0.0001
        self.tau             = 0.0001
        self.step_time       = 0
        self.loss_function   = F.smooth_l1_loss
        self.actor_noise = OUNoise(action_space=self.action_size, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)
        # Stacking
        self.stack_depth = 3
        self.frame_skip = 15
        self.input_size = self.state_size
        if ENABLE_STACKING:
            self.input_size *= self.stack_depth

        self.actor = Actor("actor", self.input_size, self.action_size, self.hidden_size).to(self.device)
        self.target_actor = Actor("target_actor", self.input_size, self.action_size, self.hidden_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)

        self.critic = Critic("critic", self.input_size, self.action_size, self.hidden_size, self.actor.forward_conv, self.device).to(self.device)
        self.target_critic = Critic("target_critic", self.input_size, self.action_size, self.hidden_size, self.actor.forward_conv, self.device).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)

        self.networks = [self.actor, self.target_actor, self.critic, self.target_critic]

        self.parameters = [self.batch_size, self.buffer_size, self.state_size, self.action_size, self.hidden_size, self.discount_factor,
                                self.learning_rate, self.tau, self.step_time, self.actor_optimizer.__class__.__name__, self.critic_optimizer.__class__.__name__,
                                self.loss_function.__name__]

        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

    def get_action(self, state, is_training, step=0, visualize=False):
        state = torch.from_numpy(numpy.asarray(state, numpy.float32)).to(self.device).unsqueeze(0)
        action = self.actor.forward(state, visualize).detach().cpu().data.numpy().tolist()[0]
        if is_training:
            N = copy.deepcopy(self.actor_noise.get_noise(t=step))
            action[LINEAR] = numpy.clip(action[LINEAR] + N[LINEAR], -1.0, 1.0)
            action[ANGULAR] = numpy.clip(action[ANGULAR] + N[ANGULAR], -1.0, 1.0)
        return action

    def train(self, replaybuffer):
        batch = replaybuffer.sample(self.batch_size)
        s_sample, a_sample, r_sample, new_s_sample, done_sample = batch

        s_sample = torch.from_numpy(s_sample).to(self.device)
        a_sample = torch.from_numpy(a_sample).to(self.device)
        r_sample = torch.from_numpy(r_sample).to(self.device)
        new_s_sample = torch.from_numpy(new_s_sample).to(self.device)
        done_sample = torch.from_numpy(done_sample).to(self.device)

        # optimize critic
        a_target = self.target_actor.forward(new_s_sample).detach()
        next_value = torch.squeeze(self.target_critic.forward(new_s_sample, a_target).detach())
        y_expected = r_sample + (1 - done_sample)*self.discount_factor*next_value               # y_exp = r _ gamma*Q'(s', P'(s'))
        y_predicted = torch.squeeze(self.critic.forward(s_sample, a_sample))                    # y_pred = Q(s,a)
        self.qvalue = y_predicted.detach()

        #TODO: use l1 loss or MSE or other?
        loss_critic = self.loss_function(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        # optimize actor
        pred_a_sample = self.actor.forward(s_sample)
        loss_actor = -1 * (self.critic.forward(s_sample, pred_a_sample)).mean()

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
        self.actor_optimizer.step()

        # Soft update all target networks
        self.soft_update(self.target_actor, self.actor, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)
        return [loss_critic.mean().detach().cpu(), loss_actor.mean().detach().cpu()]

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)