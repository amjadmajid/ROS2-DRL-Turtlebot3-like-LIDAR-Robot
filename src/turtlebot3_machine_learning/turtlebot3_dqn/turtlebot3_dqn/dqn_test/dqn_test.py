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

import collections
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import RMSprop
import json
import numpy
import os
import random
import sys
import time

import rclpy
from rclpy.node import Node

from turtlebot3_msgs.srv import Dqn


class DQNTest(Node):
    def __init__(self, stage):
        super().__init__('dqn_test')

        """************************************************************
        ** Initialise variables
        ************************************************************"""
        # Stage
        self.stage = int(stage)

        # State size and action size
        self.state_size = 4
        self.action_size = 5
        self.episode_size = 3000

        # DQN hyperparameter
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64

        # Replay memory
        self.memory = collections.deque(maxlen=1000000)

        # Build model
        self.model = self.build_model()

        # Load saved models
        self.load_model = True
        self.load_episode = 600
        self.model_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model_dir_path = self.model_dir_path.replace(
            'turtlebot3_dqn/dqn_test',
            'model')
        self.model_path = os.path.join(
            self.model_dir_path,
            'stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.h5')

        if self.load_model:
            self.model.set_weights(load_model(self.model_path).get_weights())
            with open(os.path.join(
                    self.model_dir_path,
                    'stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.json')) as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

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
    def process(self):
        global_step = 0

        for episode in range(self.load_episode+1, self.episode_size):
            global_step += 1
            local_step = 0

            state = list()
            next_state = list()
            done = False
            init = True
            score = 0

            # Reset DQN environment
            time.sleep(1.0)

            while not done:
                local_step += 1

                # Aciton based on the current state
                if local_step == 1:
                    action = 2  # Move forward
                else:
                    state = next_state
                    action = int(self.get_action(state))

                # Send action and receive next state and reward
                req = Dqn.Request()
                print(int(action))
                req.action = action
                req.init = init
                while not self.dqn_com_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info('service not available, waiting again...')

                future = self.dqn_com_client.call_async(req)

                while rclpy.ok():
                    rclpy.spin_once(self)
                    if future.done():
                        if future.result() is not None:
                            # Next state and reward
                            next_state = future.result().state
                            reward = future.result().reward
                            done = future.result().done
                            score += reward
                            init = False
                        else:
                            self.get_logger().error(
                                'Exception while calling service: {0}'.format(future.exception()))
                        break

                # While loop rate
                time.sleep(0.01)

    def build_model(self):
        model = Sequential()
        model.add(Dense(
            8,
            input_shape=(self.state_size,),
            activation='relu',
            kernel_initializer='lecun_uniform'))
        model.add(Dense(8, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()
        return model

    def get_action(self, state):
        if numpy.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = numpy.asarray(state)
            q_value = self.model.predict(state.reshape(1, len(state)))
            print(numpy.argmax(q_value[0]))
            return numpy.argmax(q_value[0])

def main(args=sys.argv[1]):
    rclpy.init(args=args)
    dqn_test = DQNTest(args)
    rclpy.spin(dqn_test)

    dqn_test.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
