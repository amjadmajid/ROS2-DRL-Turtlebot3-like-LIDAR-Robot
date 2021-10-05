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
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
import keras.backend as K
import pickle
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
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

import rclpy
from rclpy.node import Node

from turtlebot3_msgs.srv import Dqn


class DQNAgent(Node):
    def __init__(self, stage):
        super().__init__('dqn_agent')

        # ===================================================================== #
        #                       parameter initalization                         #
        # ===================================================================== #
        self.stage = int(stage)

        # State size and action size
        self.state_size = 24
        self.action_size = 5
        self.episode_size = 50000

        # DQN hyperparameter
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_minimum = 0.05
        self.batch_size = 64
        self.target_update = 8

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
        #                              Actor Model                              #
        # ===================================================================== #

        self.actor_model = Actor(self.state_size)
        self.actor_model.build_model()
        self.target_actor_model = Actor(self.state_size)
        self.target_actor_model.build_model()

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #
        self.critic_model = Critic(self.state_size, self.action_size)
        self.critic_model.build_model()
        self.target_critic_model = Critic(self.state_size, self.action_size)
        self.target_critic_model.build_model()

        # where we will feed de/dC (from critic)
        self.actor_critic_grad = tf.placeholder(
            tf.float32, [None, self.state_size])

        self.update_target_model()
        self.update_target_model_start = 128

        # ===================================================================== #
        #                             Model loading                             #
        # ===================================================================== #

        print(os.path.dirname(os.path.realpath(__file__)))
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
            print("memory length:", len(self.memory))
            print("continuing agent model from dir: %s" % self.model_dir)
        else:  # make new dir
            i = 0
            self.model_dir = os.path.join(models_dir, "dqn_%s" % i)
            while(os.path.exists(self.model_dir)):
                i += 1
                self.model_dir = os.path.join(models_dir, "dqn_%s" % i)
            print("making new model dir: %s" % self.model_dir)
            os.mkdir(self.model_dir)

        # Determine summary file name
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        summary_path = os.path.join(self.model_dir,
                                    self.timestr + '.txt')
        self.summary_file = open(summary_path, 'w+')

        """************************************************************
        ** Initialise ROS clients
        ************************************************************"""
        # Initialise clients
        self.dqn_com_client = self.create_client(Dqn, 'dqn_com')

        """************************************************************
        ** Start process
        ************************************************************"""
        self.process()

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""

    def save_progress(self, episode):
        print("saving data for episode: ", episode)
        # Store weights state
        self.model_file = os.path.join(
            self.model_dir, 'stage'+str(self.stage)+'_episode'+str(episode)+'.h5')
        self.model.save(self.model_file)
        # Store parameters state
        param_keys = ['stage', 'epsilon', 'epsilon_decay', 'epsilon_minimum', 'batch_size', 'learning_rate',
                      'discount_factor', 'episode_size', 'action_size',  'state_size', 'update_target_model_start', 'memory_size']
        param_values = [self.stage, self.epsilon, self.epsilon_decay, self.epsilon_minimum, self.batch_size, self.learning_rate, self.
                        discount_factor, self.episode_size, self.action_size, self.state_size, self.update_target_model_start, self.memory_size]
        param_dictionary = dict(zip(param_keys, param_values))
        with open(os.path.join(
                self.model_dir, 'stage'+str(self.stage)+'_episode'+str(episode)+'.json'), 'w') as outfile:
            json.dump(param_dictionary, outfile)
        # Store replay buffer state
        with open(os.path.join(
                self.model_dir, 'stage'+str(self.stage)+'_episode'+str(episode)+'.pkl'), 'wb') as f:
            pickle.dump(self.memory, f, pickle.HIGHEST_PROTOCOL)

    # CHANGE  [[state1, action1,etc], [state2,action2,etc], [state3,action3,etc]]  INTO  [[state1, state2, state3],[action1, action2, action3],etc]
    def stack_samples(samples):
        array = np.array(samples)
        current_states = np.stack(array[:, 0]).reshape((array.shape[0], -1))
        actions = np.stack(array[:, 1]).reshape((array.shape[0], -1))
        rewards = np.stack(array[:, 2]).reshape((array.shape[0], -1))
        new_states = np.stack(array[:, 3]).reshape((array.shape[0], -1))
        dones = np.stack(array[:, 4]).reshape((array.shape[0], -1))
        return current_states, actions, rewards, new_states, dones

    def train(self):
        if len(self.memory) < self.batch_size:  # batch_size:
            return
        mini_batch = random.sample(self.memory, self.batch_size)
        print("samples is %s", mini_batch)
        print("samples shape is %s", mini_batch.shape)

        current_states, actions, rewards, new_states, dones = self.stack_samples(
            mini_batch)
        # forward pass to target actor
        target_actions = self.target_actor_model.forward_pass(new_states)
        # forward pass to target critic
        future_rewards = self.target_critic_model.forward_pass(new_states, target_actions)

        rewards = rewards + self.gamma * future_rewards * (1 - dones)

        # Train the critic model
        self.critic_model.fit([current_states, actions], rewards, verbose=0)

        # 5, train actor based on weights
        predicted_actions = self.actor_model.predict(current_states)
        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_state_input:  current_states,
            self.critic_action_input: predicted_actions
        })[0]

        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: current_states,
            self.actor_critic_grad: grads
        })
        # print("grads*weights is %s", grads)

    def process(self):
        success_count = 0

        self.summary_file.write(
            "episode, reward, duration, n_steps, epsilon, success_count, memory length\n")

        for episode in range(self.load_episode+1, self.episode_size):
            local_step = 0

            state = game_start_reset()
            next_state = list()
            done = False
            init = True
            score = 0

            step_reward = [0, 0]
            step_Q = [0, 0]
            step = 0

            # Reset DQN environment
            time.sleep(1.0)

            episode_start = time.time()

            while not done:
                local_step += 1

                # Action based on the current state
                action = self.actor_model.get_action(state, self.epsilon)
                print("chosen action: ", action)

                # Send action and receive next state and reward
                req = Dqn.Request()
                req.action = action
                req.init = init
                while not self.dqn_com_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info('service not available, waiting again...')
                future = self.dqn_com_client.call_async(req)

                while rclpy.ok():
                    rclpy.spin_once(self)
                    if future.done():
                        if future.result() is not None:
                            next_state = future.result().state
                            reward = future.result().reward
                            done = future.result().done
                            score += reward
                            init = False
                        else:
                            self.get_logger().error(
                                'Exception while calling service: {0}'.format(future.exception()))
                        break

                if local_step > 1:
                    self.append_sample(state, action, reward, next_state, done)

                    self.train()  # TODO: every 5 steps? like example

                    if done:
                        # TODO: should target model only start after a certain amount of episodes?
                        # and (episode > self.update_target_model_start):
                        if (episode % self.target_update == 0):
                            print("updating target network!")
                            self.update_target_model()

                        episode_duration = time.time() - episode_start

                        print(
                            "Episode:", episode,
                            "score:", score,
                            "n_steps:", local_step,
                            "memory length:", len(self.memory),
                            "epsilon:", self.epsilon,
                            "episode duration: ", episode_duration)
                        self.summary_file.write("{}, {}, {}, {}, {}, {}, {}\n".format(
                            episode, score, episode_duration, local_step, self.epsilon, success_count, len(self.memory)))

                state = next_state
                time.sleep(0.01)  # While loop rate

            # Update result and save model every 25 episodes
            if (episode % 200 == 0) or (episode == 1):
                self.save_progress(episode)

            # Epsilon
            if self.epsilon > self.epsilon_minimum:
                self.epsilon *= self.epsilon_decay

    # def update_target_model(self, model, target_model):
    #     target_model.set_weights(model.get_weights())

    def train_model(self, target_train_start=False):
        # train_start_time = time.time()
        mini_batch = random.sample(self.memory, self.batch_size)
        x_batch = numpy.empty((0, self.state_size), dtype=numpy.float64)
        y_batch = numpy.empty((0, self.action_size), dtype=numpy.float64)
        for i in range(self.batch_size):
            state = numpy.asarray(mini_batch[i][0])
            action = numpy.asarray(mini_batch[i][1])
            reward = numpy.asarray(mini_batch[i][2])
            next_state = numpy.asarray(mini_batch[i][3])
            done = numpy.asarray(mini_batch[i][4])

            q_value = (self.model(state.reshape(1, len(state)))).numpy()

            target_value = (self.target_model(
                next_state.reshape(1, len(next_state)))).numpy()
            if done:
                next_q_value = reward
            else:
                next_q_value = reward + self.discount_factor * \
                    numpy.amax(target_value)

            x_batch = numpy.append(
                x_batch, numpy.array([state.copy()]), axis=0)

            y_sample = q_value.copy()
            y_sample[0][action] = next_q_value
            y_batch = numpy.append(y_batch, numpy.array([y_sample[0]]), axis=0)

            if done:
                x_batch = numpy.append(
                    x_batch, numpy.array([next_state.copy()]), axis=0)
                y_batch = numpy.append(y_batch, numpy.array(
                    [[reward] * self.action_size]), axis=0)
        self.model.fit(x_batch, y_batch,
                       batch_size=self.batch_size, epochs=1, verbose=0)
        #print("total train time: {}".format(time.time() - train_start_time))


def main(args=sys.argv[1]):
    rclpy.init(args=args)
    dqn_agent = DQNAgent(args)
    rclpy.spin(dqn_agent)

    dqn_agent.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
