import numpy
import time

import torch
import torch.nn.functional as F
import torch.nn as nn

from ..turtlebot3_drl.common.visual import DrlVisual
from ..turtlebot3_drl.common.settings import ENABLE_STACKING
from ..turtlebot3_drl.drl_environment.drl_environment import NUM_SCAN_SAMPLES


LINEAR = 0
ANGULAR = 1

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py
# https://github.com/djbyrne/TD3

def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)

class Actor(nn.Module):
    def __init__(self, name, state_size, action_size, hidden_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.name = name
        self.iteration = 0

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=3)
        # self.conv1_bn = nn.BatchNorm1d(64)
        self.conv1_mp = nn.MaxPool1d(3, 2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=3)
        # self.conv2_bn = nn.BatchNorm1d(32)
        self.conv2_mp = nn.MaxPool1d(3, 2)

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=2)
        # self.conv3_bn = nn.BatchNorm1d(16)
        self.flatten = nn.Flatten(1, 2)

        self.conv_sizes = self.get_conv_sizes(state_size)
        conv_out_size = self.conv_sizes[-1]
        print(f"conv out sizes: {self.conv_sizes}")

        self.fc1 = nn.Linear(conv_out_size, 124)
        self.fc2 = nn.Linear(128, action_size)
        # self.fc3 = nn.Linear(hidden_size, action_size)

        self.visual = None

        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        # self.fc3.apply(init_weights)

    def forward_conv(self, states, visualize=False, size=False):
        # batch_size, channels, length
        states = states.unsqueeze(1)
        c1 = torch.relu((self.conv1(states)))
        # c2 = self.conv1_mp(c1)
        c3 = torch.relu((self.conv2(c1)))
        # c4 = self.conv2_mp(c3)
        c5 = self.flatten(c3)
        # c5 = self.flatten(c1)
        if visualize:
            vc1 = self.flatten(c1)
            # vc3 = self.flatten(c3)
            # self.visual.update_conv([vc1, vc3, c5])
        if size:
            sc1 = self.flatten(c1)
            sc3 = self.flatten(c3)
            return [int(sc1.size(dim=1)), int(sc3.size(dim=1)), int(c5.size(dim=1))]
            # return [int(c5.size(dim=1))]
        return c5

    def get_conv_sizes(self, input_size, device=None):
        if device is not None:
            return self.forward_conv(torch.zeros(1, input_size - 4).to(device), False, True)
        return self.forward_conv(torch.zeros(1, input_size - 4), False, True)

    def forward(self, states, visualize=False):
        laser_states = torch.narrow(states, 1, 0, NUM_SCAN_SAMPLES)
        other_states = torch.narrow(states, 1, NUM_SCAN_SAMPLES, 4)

        conv_output = self.forward_conv(laser_states, visualize)
        combined = torch.cat((conv_output, other_states), dim=1)

        x1 = torch.relu(self.fc1(combined))
        x2 = torch.relu(self.fc2(x1))
        action = torch.tanh(self.fc3(x2))

        if visualize:
            self.visual.update_hidden(states, action.squeeze(), [x1, x2])
            if self.iteration % 100:
                self.visual.update_bias([self.fc1.bias, self.fc2.bias])
            self.iteration += 1
        return action


class Critic(nn.Module):

    def __init__(self, name, state_size, action_size, hidden_size, conv_function, device):
        super(Critic, self).__init__()
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        self.conv_function = conv_function

        # conv_out_size = self.get_conv_out_size(self.state_size)

        self.l1 = nn.Linear(128, 128)
        self.l2 = nn.Linear(128, 1)

        self.l5 = nn.Linear(128, 128)
        self.l4 = nn.Linear(128, 1)

        # self.l1 = nn.Linear(conv_out_size + 4, int(hidden_size / 2))
        # self.l2 = nn.Linear(action_size, int(hidden_size / 2))
        # self.l3 = nn.Linear(hidden_size, hidden_size)
        # self.l4 = nn.Linear(hidden_size, 1)

        # self.l5 = nn.Linear(conv_out_size + 4, int(hidden_size / 2))
        # self.l6 = nn.Linear(action_size, int(hidden_size / 2))
        # self.l7 = nn.Linear(hidden_size, hidden_size)
        # self.l8 = nn.Linear(hidden_size, 1)

        self.l1.apply(init_weights)
        self.l2.apply(init_weights)
        # self.l3.apply(init_weights)
        self.l4.apply(init_weights)
        self.l5.apply(init_weights)
        # self.l6.apply(init_weights)
        # self.l7.apply(init_weights)
        # self.l8.apply(init_weights)

    def forward(self, states, actions):
        laser_states = torch.narrow(states, 1, 0, NUM_SCAN_SAMPLES)
        other_states = torch.narrow(states, 1, NUM_SCAN_SAMPLES, 4)
        conv_states = self.conv_function(laser_states)
        l1 = torch.relu(self.l1(conv_states))
        states = torch.cat((l1, other_states), dim=1)
        x1 = torch.relu(self.l2(states))

        l3 = torch.relu(self.l3(conv_states))
        states = torch.cat((l3, other_states), dim=1)
        x2 = torch.relu(self.l4(states))


        # states = torch.cat((conv_states, other_states), dim=1)

        # xs = torch.relu(self.l1(states))
        # xa = torch.relu(self.l2(actions))
        # x = torch.cat((xs, xa), dim=1)
        # x = torch.relu(self.l3(x))
        # x1 = self.l4(x)

        # xs = torch.relu(self.l5(states))
        # xa = torch.relu(self.l6(actions))
        # x = torch.cat((xs, xa), dim=1)
        # x = torch.relu(self.l7(x))
        # x2 = self.l8(x)

        return x1, x2

    def get_Q(self, states, actions):
        laser_states = torch.narrow(states, 1, 0, NUM_SCAN_SAMPLES)
        other_states = torch.narrow(states, 1, NUM_SCAN_SAMPLES, 4)
        conv_states = self.conv_function(laser_states)

        l1 = torch.relu(self.l1(conv_states))
        states = torch.cat((l1, other_states), dim=1)
        x1 = torch.relu(self.l2(states))

        # states = torch.cat((conv_states, other_states), dim=1)

        # xs = torch.relu(self.l1(states))
        # xa = torch.relu(self.l2(actions))
        # x = torch.cat((xs, xa), dim=1)
        # x = torch.relu(self.l3(x))
        # x1 = self.l4(x)
        return x1

    def get_conv_out_size(self, input_size):
        c = self.conv_function(torch.zeros(1, input_size - 4).to(self.device))
        return int(c.size(dim=1))

class TD3Conv():
    def __init__(self, device, sim_speed):
        self.device = device
        self.iteration = 0

        # DRL parameters
        self.batch_size      = 1024
        self.buffer_size     = 1000000
        self.state_size      = NUM_SCAN_SAMPLES + 4
        self.action_size     = 2
        self.hidden_size     = 256
        self.discount_factor = 0.99
        self.learning_rate   = 0.0001
        self.tau             = 0.0001
        self.step_time       = 0.0
        # Stacking
        self.stack_depth = 3
        self.frame_skip = 15
        self.input_size = self.state_size
        if ENABLE_STACKING:
            self.input_size *= self.stack_depth
        # TD3 parameters
        self.policy_noise   = 0.2
        self.noise_clip     = 0.5
        self.policy_freq    = 2
        self.loss_function  = F.smooth_l1_loss

        self.last_actor_loss = 0

        self.actor = Actor("actor", self.input_size, self.action_size, self.hidden_size).to(self.device)
        self.target_actor = Actor("target_actor", self.input_size, self.action_size, self.hidden_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)

        self.critic = Critic("critic", self.input_size, self.action_size, self.hidden_size, self.actor.forward_conv, self.device).to(self.device)
        self.target_critic = Critic("target_critic", self.input_size, self.action_size, self.hidden_size, self.actor.forward_conv, self.device).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)

        self.networks = [self.actor, self.target_actor, self.critic, self.target_critic]

        self.parameters = [self.batch_size, self.buffer_size, self.state_size, self.action_size, self.hidden_size, self.discount_factor,
                                self.learning_rate, self.tau, self.step_time, self.actor_optimizer.__class__.__name__, self.critic_optimizer.__class__.__name__,
                                self.loss_function.__name__, self.policy_noise, self.noise_clip, self.policy_freq]

        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)

    def get_action(self, state, is_training, step=0, visualize=False):
        state = torch.from_numpy(numpy.asarray(state, numpy.float32)).to(self.device).unsqueeze(0)
        action = self.actor.forward(state, visualize).detach().cpu().data.numpy().tolist()[0]
        if is_training:
            # Add Gaussian noise as per original TD3 paper
            action[LINEAR] = numpy.clip(action[LINEAR] + numpy.random.normal(0.0, 0.1), -1.0, 1.0)
            action[ANGULAR] = numpy.clip(action[ANGULAR] + numpy.random.normal(0.0, 0.1), -1.0, 1.0)
        return action

    def train(self, replaybuffer):
        self.iteration += 1
        batch = replaybuffer.sample(self.batch_size)
        s_sample, a_sample, r_sample, new_s_sample, done_sample = batch
        s_sample = torch.from_numpy(s_sample).to(self.device)
        a_sample = torch.from_numpy(a_sample).to(self.device)
        r_sample = torch.from_numpy(r_sample).to(self.device)
        new_s_sample = torch.from_numpy(new_s_sample).to(self.device)
        done_sample = torch.from_numpy(done_sample).to(self.device)

        # optimize critic
        with torch.no_grad():
            # TD3 principle: add noise during train step
            noise = (torch.randn_like(a_sample) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.target_actor.forward(new_s_sample) + noise).clamp(-1.0, 1.0)

            target_q1, target_q2 = self.target_critic.forward(new_s_sample, next_action)
            target_q1 = torch.squeeze(target_q1)
            target_q2 = torch.squeeze(target_q2)
            target_Q = torch.min(target_q1, target_q2)
            y_expected = r_sample + (1 - done_sample) * self.discount_factor * target_Q

        current_q1, current_q2 = self.critic.forward(s_sample, a_sample)
        current_q1 = torch.squeeze(current_q1)
        current_q2 = torch.squeeze(current_q2)
        #TODO: use l1 loss or MSE or other?
        loss_critic = self.loss_function(current_q1, y_expected) + self.loss_function(current_q2, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2.0, norm_type=2)
        self.critic_optimizer.step()

        if self.iteration % self.policy_freq == 0:

            # optimize actor
            loss_actor = -1 * self.critic.get_Q(s_sample, self.actor.forward(s_sample)).mean()
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2.0, norm_type=2)
            self.actor_optimizer.step()

            # Soft update all target networks
            self.soft_update(self.target_actor, self.actor, self.tau)
            self.soft_update(self.target_critic, self.critic, self.tau)

            self.last_actor_loss = loss_actor.mean().detach().cpu()
        return [loss_critic.mean().detach().cpu(), self.last_actor_loss]

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)