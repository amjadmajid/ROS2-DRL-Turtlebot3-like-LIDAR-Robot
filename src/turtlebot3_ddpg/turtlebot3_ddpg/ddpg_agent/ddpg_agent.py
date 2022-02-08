#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert

import copy
import numpy
import os
import sys
import time
import math

import torch
import torch.nn.functional as F

from .sac import Critic, Actor
from .replaybuffer import ReplayBuffer
from . import storagemanager as sm
from .ounoise import OUNoise

from turtlebot3_msgs.srv import Ddpg
from turtlebot3_msgs.srv import Goal
from std_srvs.srv import Empty

import rclpy
from rclpy.node import Node

import matplotlib.pyplot as plt

# Constants
ACTION_LINEAR_MAX = 0.22
ACTION_ANGULAR_MAX = 2.0

INDEX_LIN = 0
INDEX_ANG = 1

PLOT_INTERVAL = 3

class SACAgent(Node):
    def __init__(self, stage, agent, episode):
        super().__init__('sac_agent')
        self.stage = int(stage)
        # Specify which model and episode to load from models_directory or Change to False for new session
        self.load_session = agent  # example: 'ddpg_0'
        self.load_episode = int(episode)

        # ===================================================================== #
        #                       parameter initalization                         #
        # ===================================================================== #

        # 10 laser readings, distance to goal, angle to goal, previous linear action, previous angular action
        self.state_size = 10 + 4
        self.action_size = 2
        self.episode_size = 10000

        # General hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 0.0003
        self.batch_size = 256

        # SAC hyperparameters
        self.tau = 0.01
        self.alpha = torch.tensor(0.2)

        # Replay Buffer
        self.buffer_size = 50000
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # logging
        self.loss_critic_sum = 0.0
        self.loss_actor_sum = 0.0
        self.loss_alpha_sum = 0.0
        self.rewards_data = []
        self.avg_critic_loss_data = []
        self.avg_actor_loss_data = []
        self.avg_alpha_loss_data = []

        # ===================================================================== #
        #                          GPU initalization                            #
        # ===================================================================== #

        print("gpu torch available: ", torch.cuda.is_available())
        if (torch.cuda.is_available()):
            print("device name: ", torch.cuda.get_device_name(0))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        # ===================================================================== #
        #                        Models initialization                          #
        # ===================================================================== #

        self.actor = Actor("actor", self.state_size, self.action_size, ACTION_LINEAR_MAX, ACTION_ANGULAR_MAX)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)

        self.critic = Critic("critic", self.state_size, self.action_size)
        self.target_critic = Critic("target_critic", self.state_size, self.action_size)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)
        self.hard_update(self.target_critic, self.critic)

        self.target_entropy = -torch.prod(torch.Tensor([self.action_size]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)


        #TODO: The authors of the original DDPG paper recommended time-correlated OU noise, but more recent results suggest 
        # that uncorrelated, mean-zero Gaussian noise works perfectly well. Since the latter is simpler, it is preferred. 
        self.actor_noise = OUNoise(self.action_size, theta=0.5, max_sigma=0.5, min_sigma=0.5, decay_period=8000000)

        # ===================================================================== #
        #                             Model loading                             #
        # ===================================================================== #

        # Directory where your models will be stored and loaded from
        models_directory = (os.path.dirname(os.path.realpath(__file__))).replace(
            'install/turtlebot3_ddpg/lib/python3.8/site-packages/turtlebot3_ddpg/ddpg_agent',
            'src/turtlebot3_ddpg/model')

        # Specify whether model is being trained or only evaluated
        self.training = True
        self.record_results = True
        # store model every N episodes
        self.store_interval = 200

        if self.load_session:
            self.session_dir = os.path.join(models_directory, self.load_session)
            sm.load_session(self, self.session_dir, self.load_episode)
        else:
            self.session_dir = sm.new_session_dir(models_directory)

        # File where results per episode are written
        if self.record_results:
            self.results_file = open(os.path.join(self.session_dir, time.strftime("%Y%m%d-%H%M%S") + '.txt'), 'w+')

        # ===================================================================== #
        #                             Start Process                             #
        # ===================================================================== #

        self.ddpg_com_client = self.create_client(Ddpg, 'ddpg_com')
        self.goal_com_client = self.create_client(Goal, 'goal_com')
        self.process()

    # ===================================================================== #
    #                           Class functions                             #
    # ===================================================================== #

    #TODO: move below to ddpg (model) file

    def get_action(self, state):
        action, _, _, _ = self.actor.sample(torch.FloatTensor(state).unsqueeze(0))
        action = action.detach().data.numpy().tolist()[0]
        # print(action) TODO: check output saturation
        action[0] = numpy.clip(action[0] * ACTION_LINEAR_MAX, 0.0, ACTION_LINEAR_MAX)
        action[1] = numpy.clip(action[1] * ACTION_ANGULAR_MAX, -ACTION_ANGULAR_MAX, ACTION_ANGULAR_MAX)
        return action

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def train(self):
        if self.replay_buffer.get_length() < self.batch_size:
            return 0, 0

        s_sample, a_sample, r_sample, new_s_sample, done_sample = self.replay_buffer.sample(self.batch_size)

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
        self.loss_critic_sum += loss_critic.detach()

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

        self.loss_actor_sum += actor_loss.detach()
        # Regularization Loss
        #reg_loss = 0.001 * (mean.pow(2).mean() + log_std.pow(2).mean())
        #policy_loss += reg_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() 

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.loss_alpha_sum += alpha_loss.detach()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        self.soft_update(self.critic, self.target_critic, self.tau)

    def step(self, action, previous_action):
        req = Ddpg.Request()
        req.action = action
        req.previous_action = previous_action

        while not self.ddpg_com_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('env step service not available, waiting again...')
        future = self.ddpg_com_client.call_async(req)

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                if future.result() is not None:
                    res = future.result()
                    return res.state, res.reward, res.done, res.success
                else:
                    self.get_logger().error(
                        'Exception while calling service: {0}'.format(future.exception()))
                    print("ERROR getting ddpg service response!")

    def get_goal_status(self):
        req = Goal.Request()
        while not self.goal_com_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('new goal service not available, waiting again...')
        future = self.goal_com_client.call_async(req)

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                if future.result() is not None:
                    res = future.result()
                    return res.new_goal
                else:
                    self.get_logger().error(
                        'Exception while calling service: {0}'.format(future.exception()))
                    print("ERROR getting new_goal service response!")

    def update_plots(self, episode):
        # plot 1:
        xaxis = numpy.array(range(episode))
        # x = xaxis
        # y = self.rewards_data
        # plt.subplot(2, 3, 1)
        # plt.gca().set_title('reward')
        # plt.plot(x, y)

        # plot 2:
        x = xaxis
        y = numpy.array(self.avg_critic_loss_data)

        plt.subplot(2, 2, 1)
        plt.gca().set_title('avg critic loss over episode')
        plt.plot(x, y)

        # plot 3:
        x = xaxis
        y = numpy.array(self.avg_actor_loss_data)

        plt.subplot(2, 2, 2)
        plt.gca().set_title('avg actor loss over episode')
        plt.plot(x, y)

        # plot 4:
        x = xaxis
        y = numpy.array(self.avg_alpha_loss_data)

        plt.subplot(2, 2, 3)
        plt.gca().set_title('avg actor loss over episode')
        plt.plot(x, y)


        # plot 5:
        count = int(episode / PLOT_INTERVAL)
        if count > 0:
            x = numpy.array(range(PLOT_INTERVAL, episode+1, PLOT_INTERVAL))
            averages = list()
            for i in range(count):
                avg_sum = 0
                for j in range(PLOT_INTERVAL):
                    avg_sum += self.rewards_data[i * PLOT_INTERVAL + j]
                averages.append(avg_sum / PLOT_INTERVAL)
            y = numpy.array(averages)
            plt.subplot(2, 2, 4)
            plt.gca().set_title('avg reward over 10 episodes')
            plt.plot(x, y)

        plt.draw()
        plt.pause(0.001)
        plt.show()
        plt.savefig(os.path.join(self.session_dir, "_figure.png"))

    def process(self):
        success_count = 0

        if self.record_results:
            self.results_file.write(
                "episode, reward, success, duration, n_steps, success_count, memory length, avg_critic_loss, avg_actor_loss\n")

        # for episode in range(self.load_episode+1, self.episode_size):
        episode = self.load_episode

        plt.figure(figsize=(14,10))
        plt.axis([-50, 50, 0, 10000])
        plt.ion()
        plt.show()

        self.update_plots(episode)

        while (True):
            past_action = [0.0, 0.0]
            state, _, _, _ = self.step([], past_action)
            next_state = list()
            episode_done = False
            step = 0
            reward_sum = 0.0
            time.sleep(1.0)
            episode_start = time.time()
            self.loss_critic_sum = 0.0
            self.loss_actor_sum = 0.0

            while not episode_done:
                action = self.get_action(state)
                next_state, reward, episode_done, success = self.step(action, past_action)
                past_action = copy.deepcopy(action)
                reward_sum += reward
                state = copy.deepcopy(next_state)
                step += 1

                if self.training == True:
                    self.replay_buffer.add_sample(state, action, reward, next_state, episode_done)
                    self.train()

                # time.sleep(0.01)  # While loop rate

            avg_critic_loss = self.loss_critic_sum / step
            avg_actor_loss = self.loss_actor_sum / step
            avg_alpha_loss = self.loss_alpha_sum / step
            episode += 1
            episode_duration = time.time() - episode_start
            print(f"Episode: {episode} score: {reward_sum} success: {success} n_steps: {step} memory length: {self.replay_buffer.get_length()} episode duration: {episode_duration}")
            self.results_file.write(f"{episode}, {reward_sum}, {success}, {episode_duration}, {step}, {success_count}, {self.replay_buffer.get_length()}, {avg_critic_loss}, {avg_actor_loss}\n")

            self.rewards_data.append(reward_sum)
            self.avg_critic_loss_data.append(avg_critic_loss)
            self.avg_actor_loss_data.append(avg_actor_loss)
            self.avg_alpha_loss_data.append(avg_alpha_loss)

            # self.update_plots(episode)
            if (self.training == True):
                if (episode % self.store_interval == 0) or (episode == 1):
                    # self.update_plots(episode)
                    sm.save_session(self, self.session_dir, episode)

            if self.training != True:
                while(self.get_goal_status() == False):
                    print("Waiting for new goal...")
                    time.sleep(1.0)

def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    ddpg_agent = SACAgent(args[0], args[1], args[2])
    rclpy.spin(ddpg_agent)

    ddpg_agent.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
