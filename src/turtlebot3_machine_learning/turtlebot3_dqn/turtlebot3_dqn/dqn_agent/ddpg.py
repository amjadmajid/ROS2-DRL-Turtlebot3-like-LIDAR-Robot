import numpy
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import random

import torch
import torch.nn.functional as F
import torch.nn as nn

import os.path
import timeit
import csv
import math
import time
import matplotlib.pyplot as plt


class Actor:
    def __init__(self, name, state_size, action_size, action_limit_v, action_limit_w,):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.name = name

        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w

        self.fa1 = nn.Linear(state_size, 250)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        # self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())

        self.fa2 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fa2.weight)
        self.fa2.bias.data.fill_(0.01)
        # self.fa2.weight.data = fanin_init(self.fa2.weight.data.size())

        self.fa3 = nn.Linear(250, action_size)
        nn.init.xavier_uniform_(self.fa3.weight)
        self.fa3.bias.data.fill_(0.01)
        # self.fa3.weight.data.uniform_(-EPS,EPS)

    def forward_pass(self, states):
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
    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i] * \
                self.tau + actor_target_weights[i]*(1-self.tau)
        self.target_actor_model.set_weights(actor_target_weights)

    def update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i] * \
                self.tau + critic_target_weights[i]*(1-self.tau)
        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()


class Critic(nn.Module):

    def __init__(self, name, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.name = name

        self.fc1 = nn.Linear(state_size, 125)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        # self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fa1 = nn.Linear(action_size, 125)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        # self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())

        self.fca1 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fca1.weight)
        self.fca1.bias.data.fill_(0.01)
        # self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fca2 = nn.Linear(250, 1)
        nn.init.xavier_uniform_(self.fca2.weight)
        self.fca2.bias.data.fill_(0.01)
        # self.fca2.weight.data.uniform_(-EPS, EPS)

    def forward_pass(self, states, actions):
        xs = torch.relu(self.fc1(states))
        xa = torch.relu(self.fa1(actions))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.fca1(x))
        vs = self.fca2(x)
        return vs
