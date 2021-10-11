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

import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import pickle
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.client import device_lib
import json
import numpy
import os
import random
import sys
import time
from .ddpg import Critic, Actor
from .replaybuffer import ReplayBuffer

from turtlebot3_msgs.srv import Ddpg

import rclpy
from rclpy.node import Node


class DDPGAgent(Node):
    def __init__(self, stage):
        super().__init__('ddpg_agent')

        # ===================================================================== #
        #                       parameter initalization                         #
        # ===================================================================== #
        self.stage = int(stage)

        # State size and action size
        self.state_size = 24
        self.action_num = 2
        self.episode_size = 50000

        # DDPG hyperparameters
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_minimum = 0.05
        self.batch_size = 64
        self.target_update_interval = 8  # in episodes

        # DDPG hyperparameters
        self.tau = 0.01

        # Replay memory
        self.memory_size = 100000
        self.memory = ReplayBuffer(self.memory_size)

        # ===================================================================== #
        #                          GPU initalization                            #
        # ===================================================================== #
        print("GPU INITALIZATION")
        gpu_devices = tf.config.experimental.list_physical_devices(
            'GPU')
        print("GPU devices ({}): {}".format(len(gpu_devices),  gpu_devices))

        # ===================================================================== #
        #                        Models initialization                          #
        # ===================================================================== #

        self.actor = Actor(self.state_size)
        self.critic = Critic(self.state_size, self.action_num)
        self.target_actor = Actor(self.state_size)
        self.target_critic = Critic(self.state_size, self.action_num)

        self.actor.build_model()
        self.critic.build_model()
        self.target_actor.build_model()
        self.target_critic.build_model()

        # TODO: initalize same weights between model and target model?
        self.update_network_parameters(self.tau)

        # ===================================================================== #
        #                             Model loading                             #
        # ===================================================================== #

        models_dir = (os.path.dirname(os.path.realpath(__file__))).replace('install/turtlebot3_dqn/lib/python3.8/site-packages/turtlebot3_dqn/dqn_agent',
                                                                           'src/turtlebot3_machine_learning/turtlebot3_dqn/model')
        # models_dir = '/media/tomas/JURAJ\'S USB'

        # Load saved models if needed
        self.load_model = False  # change to false to not load model
        self.load_episode = 2800 if self.load_model else 0
        if self.load_model:
            self.model_dir = os.path.join(models_dir, self.load_model)
            # load weights
            self.model_file = os.path.join(self.model_dir,
                                           'stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.h5')
            print("continuing agent model from file: %s" % self.model_file)
            self.model.set_weights(load_model(self.model_file).get_weights())
            # load hyperparameters
            with open(os.path.join(self.model_dir,
                                   'stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.json')) as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')
            # load replay memory buffer
            with open(os.path.join(self.model_dir,
                                   'stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.pkl'), 'rb') as f:
                self.memory = pickle.load(f)
            print("memory length:", self.memory.get_length())
            print("continuing agent model from dir: %s" % self.model_dir)
        else:  # make new dir
            i = 0
            self.model_dir = os.path.join(models_dir, "ddpg_%s" % i)
            while(os.path.exists(self.model_dir)):
                i += 1
                self.model_dir = os.path.join(models_dir, "ddpg_%s" % i)
            print("making new model dir: %s" % self.model_dir)
            os.mkdir(self.model_dir)

            # Determine summary file name
            self.timestr = time.strftime("%Y%m%d-%H%M%S")
            summary_path = os.path.join(self.model_dir,
                                        self.timestr + '.txt')
            self.summary_file = open(summary_path, 'w+')

            # ===================================================================== #
            #                             Start Process                             #
            # ===================================================================== #

        self.ddpg_com_client = self.create_client(Ddpg, 'ddpg_com')
        self.process()

    # ===================================================================== #
    #                           Class functions                             #
    # ===================================================================== #

    def update_network_parameters(self, tau):
        # update target actor
        weights = []
        target_weights = self.target_actor.model.weights
        for i, new_weight in enumerate(self.actor.model.weights):
            weights.append(new_weight * tau + target_weights[i]*(1-tau))
        self.target_actor.model.set_weights(weights)

        # update target critic
        weights = []
        target_weights = self.target_critic.model.weights
        for i, new_weight in enumerate(self.critic.model.weights):
            weights.append(new_weight * tau + target_weights[i]*(1-tau))
        self.target_critic.model.set_weights(weights)

    def save_progress(self, episode):
        print("saving data for episode: ", episode)
        # Store weights state
        self.actor_model_file = os.path.join(
            self.model_dir, 'am_stage'+str(self.stage)+'_episode'+str(episode)+'.h5')
        self.actor.model.save_weights(self.actor_model_file)

        self.actor_target_file = os.path.join(
            self.model_dir, 'at_stage'+str(self.stage)+'_episode'+str(episode)+'.h5')
        self.target_actor.model.save_weights(self.actor_target_file)

        self.critic_model_file = os.path.join(
            self.model_dir, 'cm_stage'+str(self.stage)+'_episode'+str(episode)+'.h5')
        self.critic.model.save_weights(self.critic_model_file)

        self.critic_target_file = os.path.join(
            self.model_dir, 'ct_stage'+str(self.stage)+'_episode'+str(episode)+'.h5')
        self.target_critic.model.save_weights(self.critic_target_file)

        # Store parameters state
        param_keys = ['stage', 'epsilon', 'epsilon_decay', 'epsilon_minimum', 'batch_size', 'learning_rate',
                      'discount_factor', 'episode_size', 'action_num',  'state_size', 'target_update_interval', 'memory_size', 'tau']
        param_values = [self.stage, self.epsilon, self.epsilon_decay, self.epsilon_minimum, self.batch_size, self.learning_rate, self.
                        discount_factor, self.episode_size, self.action_num, self.state_size, self.target_update_interval, self.memory_size, self.tau]
        param_dictionary = dict(zip(param_keys, param_values))
        with open(os.path.join(
                self.model_dir, 'stage'+str(self.stage)+'_episode'+str(episode)+'.json'), 'w') as outfile:
            json.dump(param_dictionary, outfile)

        # Store replay buffer state
        with open(os.path.join(
                self.model_dir, 'stage'+str(self.stage)+'_episode'+str(episode)+'.pkl'), 'wb') as f:
            pickle.dump(self.memory, f, pickle.HIGHEST_PROTOCOL)

    # TODO: use this to speed up performance?
    # TODO: pause gazebo during graph construction by this decorator
    @tf.function
    def update_weights(self, current_states, actions, rewards, new_states, dones):
        # Train the critic model
        with tf.GradientTape() as tape:
            target_actions = self.target_actor.forward_pass(new_states)
            future_rewards = self.target_critic.forward_pass(new_states, target_actions)
            target = rewards + self.discount_factor * future_rewards * (1 - dones)
            actual_rewards = self.critic.forward_pass(current_states, actions)
            critic_loss = tensorflow.keras.losses.MSE(target, actual_rewards)

        # Update critic weights
        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.model.trainable_variables)
        self.critic.model.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.model.trainable_variables))

        # Train the actor model
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor.forward_pass(current_states)
            actor_loss = tf.math.reduce_mean(-self.critic.forward_pass(current_states,
                                                                       new_policy_actions))
            # Update actor weights
        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.model.trainable_variables)
        self.actor.model.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.model.trainable_variables))

        return critic_loss, actor_loss

    def train(self):
        if self.memory.get_length() < self.batch_size:  # batch_size:
            return 0, 0

        mini_batch = self.memory.get_sample(self.batch_size)
        current_states, actions, rewards, new_states, dones = zip(*mini_batch)
        c_l, a_l = self.update_weights(tf.convert_to_tensor(current_states, numpy.float32),
                                       tf.convert_to_tensor(actions, numpy.float32),
                                       tf.convert_to_tensor(rewards, numpy.float32),
                                       tf.convert_to_tensor(new_states, numpy.float32),
                                       tf.convert_to_tensor(dones, numpy.float32))
        return c_l, a_l

    def step(self, action):
        req = Ddpg.Request()
        req.action = action

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
            "episode, reward, duration, n_steps, epsilon, success_count, memory length\n")

        for episode in range(self.load_episode+1, self.episode_size):
            state, _, _ = self.step([])
            next_state = list()
            done = False
            step = 0
            score = 0
            time.sleep(1.0)
            episode_start = time.time()

            while not done:
                step_start = time.time()
                # Send action and receive next state and reward
                action = self.actor.get_action(state, self.epsilon)
                next_state, reward, done = self.step(action)
                score += reward

                if step > 1:
                    self.memory.append_sample(state, action, reward, next_state, done)
                    train_start = time.time()
                    critic_loss, actor_loss = self.train()  # TODO: alternate experience gathering and training?
                    # print("critic_loss: {}, actor_loss: {}, train time: {}".format(
                    #       critic_loss, actor_loss, time.time() - train_start))
                    if episode % self.target_update_interval == 0:
                        self.update_network_parameters(self.tau)

                    if done:
                        episode_duration = time.time() - episode_start
                        print("Episode: %d score: %d n_steps: %d memory length: %d epsilon: %d episode duration: %d",
                              episode, score, step, self.memory.get_length(), self.epsilon, episode_duration)
                        self.summary_file.write("{}, {}, {}, {}, {}, {}, {}\n".format(  # todo: remove format
                            episode, score, episode_duration, step, self.epsilon, success_count, self.memory.get_length()))

                # Prepare for next step
                state = next_state
                step += 1
                print("step time: ", time.time() - step_start)
                # time.sleep(0.01)  # While loop rate

            # Update result and save model every 25 episodes
            if (episode % 5 == 0) or (episode == 1):
                self.save_progress(episode)

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
