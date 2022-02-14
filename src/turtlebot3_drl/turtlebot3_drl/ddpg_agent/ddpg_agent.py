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
import os
import sys
import time

from .ddpg import DDPG
from .replaybuffer import ReplayBuffer
from . import storagemanager as sm
from .utilities import DDPGplot
from . import utilities
from .ounoise import OUNoise

from turtlebot3_msgs.srv import DrlStep
from turtlebot3_msgs.srv import Goal

import rclpy
from rclpy.node import Node

PLOT_INTERVAL = 2

class DDPGagent(Node):
    def __init__(self, stage, agent, episode):
        super().__init__('ddpg_agent')
        self.stage = int(stage)
        self.load_session = agent
        self.load_episode = int(episode)

        # 10 laser readings, distance to goal, angle to goal, previous linear action, previous angular action
        self.state_size = 10 + 4
        self.action_size = 2

        # General hyperparameters
        self.discount_factor = 0.998
        self.learning_rate = 0.001
        self.batch_size = 256

        # DDPG hyperparameters
        self.tau = 0.01

        self.actor_noise = OUNoise(self.action_size, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)

        # Replay Buffer
        self.buffer_size = 50000
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # logging
        self.rewards_data = []
        self.avg_critic_loss_data = []
        self.avg_actor_loss_data = []

        self.device = utilities.check_gpu()
        self.agent = DDPG(self.device, self.state_size, self.action_size, self.discount_factor, self.learning_rate, self.tau, self.actor_noise)

        self.is_training = True
        self.store_interval = 200 # save simulation state every N episodes

        # ===================================================================== #
        #                             Model loading                             #
        # ===================================================================== #

        # Directory where your models will be stored and loaded from
        models_directory = (os.path.dirname(os.path.realpath(__file__))).replace(
            'install/turtlebot3_drl/lib/python3.8/site-packages/turtlebot3_drl/ddpg_agent',
            'src/turtlebot3_drl/model')

        if self.load_session:
            self.session_dir = os.path.join(models_directory, self.load_session)
            sm.load_session(self, self.agent, self.session_dir, self.load_episode)
        else:
            self.session_dir = sm.new_session_dir(models_directory)

        self.results_file = open(os.path.join(self.session_dir, time.strftime("%Y%m%d-%H%M%S") + '.txt'), 'w+')

        # ===================================================================== #
        #                             Start Process                             #
        # ===================================================================== #

        self.step_com_client = self.create_client(DrlStep, 'step_com')
        self.goal_com_client = self.create_client(Goal, 'goal_com')
        self.process()


    def process(self):
        success_count = 0

        self.results_file.write(
            "episode, reward, success, duration, n_steps, success_count, memory length, avg_critic_loss, avg_actor_loss\n")

        episode = self.load_episode
        ddpg_plot = DDPGplot(self.session_dir, PLOT_INTERVAL, episode, self.rewards_data, self.avg_critic_loss_data, 
                                self.avg_actor_loss_data)


        while (True):
            past_action = [0.0, 0.0]
            state, _, _, _ = utilities.step(self, [], past_action)
            next_state = list()
            episode_done = False
            step = 0
            time.sleep(1.0)
            episode_start = time.time()
            reward_sum, loss_critic_sum, loss_actor_sum = 0.0, 0.0, 0.0

            while not episode_done:
                action = self.agent.get_action(state, step)
                next_state, reward, episode_done, success = utilities.step(self, action, past_action)
                past_action = copy.deepcopy(action)
                reward_sum += reward
                state = copy.deepcopy(next_state)
                step += 1

                if self.is_training == True:
                    self.replay_buffer.add_sample(state, action, reward, next_state, episode_done)
                    if self.replay_buffer.get_length() >= self.batch_size:
                        # TODO: how often train for every added sample?
                        cri_loss, act_loss, = self.agent.train(self.replay_buffer.sample(self.batch_size))
                        loss_critic_sum += cri_loss
                        loss_actor_sum += act_loss

                # time.sleep(0.01)  # While loop rate

            # Episode finished
            episode += 1
            episode_duration = time.time() - episode_start
            self.rewards_data.append(reward_sum)
            self.avg_critic_loss_data.append(loss_critic_sum / step)
            self.avg_actor_loss_data.append(loss_actor_sum / step)

            print(f"Episode: {episode} score: {reward_sum} success: {success} n_steps: {step} memory length: {self.replay_buffer.get_length()} episode duration: {episode_duration}")
            self.results_file.write(f"{episode}, {reward_sum}, {success}, {episode_duration}, {step}, {success_count}, {self.replay_buffer.get_length()}, {loss_critic_sum / step}, {loss_actor_sum / step}\n")
          
            ddpg_plot.update_plots(episode, self.rewards_data, self.avg_critic_loss_data, self.avg_actor_loss_data)
            if (self.is_training == True):
                if (episode % self.store_interval == 0) or (episode == 1):
                    sm.save_session(self, self.agent, self.session_dir, episode)

            if self.is_training != True:
                while(utilities.get_goal_status(self) == False):
                    print("Waiting for new goal...")
                    time.sleep(1.0)

def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    ddpg_agent = DDPGagent(args[0], args[1], args[2])
    rclpy.spin(ddpg_agent)

    ddpg_agent.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
