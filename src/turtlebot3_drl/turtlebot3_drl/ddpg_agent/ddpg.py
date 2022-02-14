import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal

import math
import numpy

# Reference for network structure: https://arxiv.org/pdf/2102.10711.pdf
# https://github.com/hanlinniu/turtlebot3_ddpg_collision_avoidance/blob/main/turtlebot_ddpg/scripts/original_ddpg/ddpg_network_turtlebot3_original_ddpg.py


ACTION_LINEAR_MAX = 0.22
ACTION_ANGULAR_MAX = 2.0

class Actor(nn.Module):
    def __init__(self, name, state_size, action_size, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.name = name

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

class DDPG():
    def __init__(self, device, state_size, action_size, discount_factor, learning_rate, tau):
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.tau = tau

        self.alpha = torch.tensor(0.2)

        self.actor = Actor("actor", self.state_size, self.action_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)

        self.critic = Critic("critic", self.state_size, self.action_size)
        self.target_critic = Critic("target_critic", self.state_size, self.action_size)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
        self.hard_update(self.target_critic, self.critic)

        self.target_entropy = -torch.prod(torch.Tensor([self.action_size]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

    def get_action(self, state):
        action, _, _, _ = self.actor.sample(torch.FloatTensor(state).unsqueeze(0))
        action = action.detach().data.numpy().tolist()[0]
        # print(action) TODO: check output saturation
        action[0] = numpy.clip(action[0] * ACTION_LINEAR_MAX, 0.0, ACTION_LINEAR_MAX)
        action[1] = numpy.clip(action[1] * ACTION_ANGULAR_MAX, -ACTION_ANGULAR_MAX, ACTION_ANGULAR_MAX)
        return action

    def train(self, batch):
        s_sample, a_sample, r_sample, new_s_sample, done_sample = batch
        s_sample = torch.from_numpy(s_sample)
        a_sample = torch.from_numpy(a_sample)
        r_sample = torch.from_numpy(r_sample).unsqueeze(1)
        new_s_sample = torch.from_numpy(new_s_sample)
        done_sample = torch.from_numpy(done_sample).unsqueeze(1)

        # optimize critic (q function)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.actor.sample(new_s_sample)
            qf1_next_target, qf2_next_target = self.target_critic.forward(new_s_sample, next_state_action)
            min_qf_next_target = torch.minimum(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = r_sample + (1 - done_sample) * self.discount_factor * (min_qf_next_target)

        qf1, qf2 = self.critic.forward(s_sample, a_sample)
        q1_loss_critic = F.mse_loss(qf1, next_q_value)
        q2_loss_critic = F.mse_loss(qf2, next_q_value)
        loss_critic = q1_loss_critic + q2_loss_critic

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        self.critic_optimizer.step()

        # optimize actor (policy)
        pi, log_pi, mean, log_std = self.actor.sample(s_sample)

        # todo: use sum?
        # q1_loss_actor, q2_loss_actor = -1*torch.sum(self.critic.forward(s_sample, pred_a_sample))
        qf1_pi, qf2_pi = self.critic.forward(s_sample, pi)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # Regularization Loss
        #reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
        #policy_loss += reg_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        self.soft_update(self.critic, self.target_critic, self.tau)
        return [loss_critic.detach(), actor_loss.detach(), alpha_loss.detach()]

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)        