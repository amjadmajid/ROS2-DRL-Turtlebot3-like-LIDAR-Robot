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

import torch.nn as nn
import torch.nn.functional as F
import torch

from .ddpg import Critic, Actor
from .replaybuffer import ReplayBuffer
from . import storagemanager as sm
from .ounoise import OUNoise

from turtlebot3_msgs.srv import Ddpg
from std_srvs.srv import Empty

import rclpy
from rclpy.node import Node

ACTION_LINEAR_MAX = 0.22
ACTION_ANGULAR_MAX = 2.0

INDEX_LIN = 0
INDEX_ANG = 1


class DDPGAgent(Node):
    def __init__(self, stage):
        super().__init__('ddpg_agent')

        # ===================================================================== #
        #                       parameter initalization                         #
        # ===================================================================== #
        self.stage = int(stage)

        # State size and action size
        self.state_size = 14
        self.action_size = 2
        self.episode_size = 50000

        # General hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_minimum = 0.05
        self.batch_size = 256
        # self.target_update_interval = 5  # in episodes

        # DDPG hyperparameters
        self.tau = 0.001

        # Replay memory
        self.memory_size = 100000
        self.memory = ReplayBuffer(self.memory_size)

        self.graph_build = False

        # ===================================================================== #
        #                          GPU initalization                            #
        # ===================================================================== #
        # print("GPU INITALIZATION")
        # gpu_devices = tf.config.experimental.list_physical_devices(
        #     'GPU')
        # print("GPU devices ({}): {}".format(len(gpu_devices),  gpu_devices))

        # ===================================================================== #
        #                        Models initialization                          #
        # ===================================================================== #

        self.actor = Actor("actor", self.state_size, self.action_size, ACTION_LINEAR_MAX, ACTION_ANGULAR_MAX)
        self.target_actor = Actor("target_actor", self.state_size, self.action_size, ACTION_LINEAR_MAX, ACTION_ANGULAR_MAX)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.learning_rate)

        self.critic = Critic("critic", self.state_size, self.action_size)
        self.target_critic = Critic("target_critic", self.state_size, self.action_size)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.learning_rate)

        # TODO: initalize same weights between model and target model?
        self.update_network_parameters(1)

        # ===================================================================== #
        #                             Model loading                             #
        # ===================================================================== #

        models_directory = (os.path.dirname(os.path.realpath(__file__))).replace(
            'install/turtlebot3_dqn/lib/python3.8/site-packages/turtlebot3_dqn/dqn_agent',
            'src/turtlebot3_machine_learning/turtlebot3_dqn/model')
        # models_dir = '/media/tomas/JURAJ\'S USB'

        # Change load_model to load desired model (e.g. 'ddpg_0') or False for new session
        self.load_session = False  # example: 'ddpg_0'
        self.load_episode = 1 if self.load_session else 0

        if self.load_session:
            self.session_dir = os.path.join(models_directory, self.load_session)
            sm.load_session(self, self.session_dir, self.load_episode)
        else:  # New train session
            self.session_dir = sm.new_model_dir(models_directory)

        # Determine summary file name
        self.summary_file = open(os.path.join(self.session_dir, time.strftime("%Y%m%d-%H%M%S") + '.txt'), 'w+')

        # ===================================================================== #
        #                             Start Process                             #
        # ===================================================================== #

        self.actor_noise = OUNoise(self.action_size, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)
        self.ddpg_com_client = self.create_client(Ddpg, 'ddpg_com')
        self.pause_simulation_client = self.create_client(Empty, '/pause_physics')
        self.unpause_simulation_client = self.create_client(Empty, '/unpause_physics')
        self.process()

    # ===================================================================== #
    #                           Class functions                             #
    # ===================================================================== #

    # TODO: move this elsewhere
    def pause_physics(self):
        req = Empty.Request()
        while not self.pause_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        print("pausing simulation!")
        self.pause_simulation_client.call_async(req)

    def unpause_physics(self):
        req = Empty.Request()
        while not self.unpause_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        print("unpausing simulation!")
        self.unpause_simulation_client.call_async(req)

    def get_action(self, state, step):

        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        action = action.data.numpy()
        N = copy.deepcopy(self.actor_noise.get_noise(t=step))
        N[0] = N[0]*ACTION_LINEAR_MAX/2
        N[1] = N[1]*ACTION_ANGULAR_MAX
        action[0] = numpy.clip(action[0] + N[0], 0., ACTION_LINEAR_MAX)
        action[1] = numpy.clip(action[1] + N[1], -ACTION_ANGULAR_MAX, ACTION_ANGULAR_MAX)
        return action

        state_np = numpy.asarray(state, numpy.float32)
        state_np = state_np.reshape(1, len(state_np))
        state_tensor = tf.convert_to_tensor(state_np, numpy.float32)
        action = self.actor.forward_pass(state_tensor)
        print("action: ", action)
        action = action.numpy()
        action = action.tolist()
        action = action[0]
        linear = action[INDEX_LIN] * ACTION_LINEAR_MAX
        angular = action[INDEX_ANG] * ACTION_ANGULAR_MAX
        noise_lin = 0
        noise_ang = 0

        # OUNoise
        # TODO: allow backwards linear movement?
        # noise = self.actor_noise.get_noise(step)
        # noise_lin = noise[0] * ACTION_LINEAR_MAX/2
        # noise_ang = noise[1] * ACTION_ANGULAR_MAX

        # normal noise
        # if numpy.random.random() < self.epsilon:
        noise_lin = (numpy.random.random()-0.5)*0.4 * self.epsilon
        noise_ang = (numpy.random.random()-0.5) * 4 * self.epsilon

        linear = numpy.clip(linear + noise_lin, 0, ACTION_LINEAR_MAX)
        angular = numpy.clip(angular + noise_ang, -ACTION_ANGULAR_MAX, ACTION_ANGULAR_MAX)
        return [linear, angular]

    def update_network_parameters(self, tau):

        # update target actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data*(1.0 - tau) + param.data*tau)

        # update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data*(1.0 - tau) + param.data*tau)

    def train(self):
        if self.memory.get_length() < self.batch_size:  # batch_size:
            return 0, 0

        s_sample, a_sample, r_sample, new_s_sample, done_sample = self.memory.sample(self.batch_size)

        s_sample = torch.from_numpy(s_sample)
        a_sample = torch.from_numpy(a_sample)
        r_sample = torch.from_numpy(r_sample)
        new_s_sample = torch.from_numpy(new_s_sample)
        done_sample = torch.from_numpy(done_sample)

        # optimize critic
        a_target = self.target_actor.forward(new_s_sample).detach()
        next_value = torch.squeeze(self.target_critic.forward(new_s_sample, a_target).detach())
        # y_exp = r _ gamma*Q'(s', P'(s'))
        y_expected = r_sample + (1 - done_sample)*self.discount_factor*next_value
        # y_pred = Q(s,a)
        y_predicted = torch.squeeze(self.critic.forward(s_sample, a_sample))
        self.qvalue = y_predicted.detach()
        # self.pub_qvalue.publish(torch.max(self.qvalue))
        print(self.qvalue, torch.max(self.qvalue))

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
        self.update_network_parameters(self.tau)

    def step(self, action, previous_action):
        req = Ddpg.Request()
        req.action = action
        req.previous_action = previous_action

        while not self.ddpg_com_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        future = self.ddpg_com_client.call_async(req)

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                if future.result() is not None:
                    res = future.result()
                    return res.state, res.reward, res.done
                else:
                    self.get_logger().error(
                        'Exception while calling service: {0}'.format(future.exception()))
                    print("ERROR getting ddpg service response!")

    def process(self):
        success_count = 0

        self.summary_file.write(
            "episode, reward, duration, n_steps, epsilon, success_count, memory length, avg_critic_loss, avg_actor_loss\n")

        for episode in range(self.load_episode+1, self.episode_size):
            past_action = numpy.zeros(self.action_size)
            state, _, _ = self.step([], past_action)
            next_state = list()
            done = False
            step = 0
            reward_sum = 0.0
            time.sleep(1.0)
            episode_start = time.time()
            sum_critic_loss = 0.0
            sum_actor_loss = 0.0

            while not done:
                step_start = time.time()
                # Send action and receive next state and reward
                action = self.get_action(state, step)
                next_state, reward, done = self.step(action, past_action)
                past_action = copy.deepcopy(action)
                reward_sum += reward

                if step > 1:
                    self.memory.add_sample(state, action, reward, next_state, done)
                    train_start = time.time()
                    critic_loss, actor_loss = self.train()  # TODO: alternate experience gathering and training?
                    # sum_critic_loss += critic_loss
                    # sum_actor_loss += actor_loss
                    train_time = (time.time() - train_start)

                    if done:
                        # avg_critic_loss = sum_critic_loss / step
                        # avg_actor_loss = sum_actor_loss / step
                        episode_duration = time.time() - episode_start
                        print("Episode: {} score: {} n_steps: {} memory length: {} epsilon: {} episode duration: {}".format(
                              episode, reward_sum, step, self.memory.get_length(), self.epsilon, episode_duration))
                        self.summary_file.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(  # todo: remove format
                            episode, reward_sum, episode_duration, step, self.epsilon, success_count, self.memory.get_length()))  # , avg_critic_loss, avg_actor_loss))

                # Prepare for next step
                state = next_state
                step += 1
                # print("step time: ", time.time() - step_start)
                # time.sleep(0.01)  # While loop rate

            # Update result and save model every 100 episodes
            if (episode % 100 == 0) or (episode == 1):
                sm.save_session(self, self.session_dir, episode)

            # Epsilon
            if self.epsilon > self.epsilon_minimum:
                self.epsilon *= self.epsilon_decay


def main(args=sys.argv[1]):
    rclpy.init(args=args)
    ddpg_agent = DDPGAgent(args)
    rclpy.spin(ddpg_agent)

    ddpg_agent.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
