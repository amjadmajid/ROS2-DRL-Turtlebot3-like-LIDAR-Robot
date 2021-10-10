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

import math
from typing import List
import numpy

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Ddpg


class DDPGEnvironment(Node):
    def __init__(self):
        super().__init__('ddpg_environment')

        """************************************************************
        ** Initialise variables
        ************************************************************"""
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.last_pose_x = 0.0
        self.last_pose_y = 0.0
        self.last_pose_theta = 0.0

        self.action_num = 2
        self.done = False
        self.collision = False
        self.succeed = False

        self.time_penalty = -0.1

        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.init_goal_distance = 1.0
        self.scan_ranges = []
        self.min_obstacle_distance = 10.0

        self.local_step = 0

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Initialise publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)

        # Initialise subscribers
        self.goal_pose_sub = self.create_subscription(
            Pose,
            'goal_pose',
            self.goal_pose_callback,
            qos)
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            qos)
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile=qos_profile_sensor_data)

        # Initialise client
        self.task_succeed_client = self.create_client(Empty, 'task_succeed')
        self.task_fail_client = self.create_client(Empty, 'task_fail')

        # Initialise servers
        self.ddpg_com_server = self.create_service(
            Ddpg, 'ddpg_com', self.ddpg_com_callback)

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""

    def goal_pose_callback(self, msg):
        self.goal_pose_x = msg.position.x
        self.goal_pose_y = msg.position.y

    def odom_callback(self, msg):
        self.last_pose_x = msg.pose.pose.position.x
        self.last_pose_y = msg.pose.pose.position.y
        _, _, self.last_pose_theta = self.euler_from_quaternion(
            msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x-self.last_pose_x)**2
            + (self.goal_pose_y-self.last_pose_y)**2)

        path_theta = math.atan2(
            self.goal_pose_y-self.last_pose_y,
            self.goal_pose_x-self.last_pose_x)

        goal_angle = path_theta - self.last_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    def scan_callback(self, msg):
        for i in range(len(msg.ranges)):
            if msg.ranges[i] > 3.5:  # max range is specified in model.sdf
                msg.ranges[i] = 3.5
        self.scan_ranges = msg.ranges
        self.min_obstacle_distance = min(self.scan_ranges)

    def get_state(self, action_linear, action_angular):
        state = self.scan_ranges.tolist()
        state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))
        state.append(float(action_linear))
        state.append(float(action_angular))
        self.local_step += 1

        # Succeed
        if self.goal_distance < 0.20:  # unit: m
            print("Goal! :)")
            self.succeed = True
            self.done = True
            self.cmd_vel_pub.publish(Twist())  # robot stop
            self.local_step = 0
            req = Empty.Request()
            while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.task_succeed_client.call_async(req)

        # Fail
        if self.min_obstacle_distance < 0.13:  # unit: m
            print("Collision! :(")
            self.collision = True
            self.done = True
            self.cmd_vel_pub.publish(Twist())  # robot stop
            self.local_step = 0
            req = Empty.Request()
            while not self.task_fail_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.task_fail_client.call_async(req)

        if self.local_step == 1000:
            print("Time out! :(")
            self.done = True
            self.local_step = 0
            req = Empty.Request()
            while not self.task_fail_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.task_fail_client.call_async(req)

        return state

    def get_reward(self, action_linear, action_angular):
        # yaw_reward will be between -1 and 1
        yaw_reward = 1 - 2*math.sqrt(math.fabs(self.goal_angle / math.pi))

        distance_reward = (2 * self.init_goal_distance) / \
            (self.init_goal_distance + self.goal_distance) - 1

        # Reward for avoiding obstacles
        if self.min_obstacle_distance < 0.25:
            obstacle_reward = -2
        else:
            obstacle_reward = 0

        if action_linear < 0.2:
            linear_reward = -2
        else:
            linear_reward = 0

        reward = yaw_reward + distance_reward + obstacle_reward + self.time_penalty + linear_reward

        # + for succeed, - for fail
        if self.succeed:
            reward += 50  # 5
        elif self.collision:
            reward -= 100  # -10
        return reward

    def ddpg_com_callback(self, request, response):
        if len(request.action) == 0:
            self.init_goal_distance = math.sqrt(
                (self.goal_pose_x-self.last_pose_x)**2
                + (self.goal_pose_y-self.last_pose_y)**2)
            response.state = self.get_state(0, 0)
            response.reward = 0
            response.done = False
            return response

        action = request.action
        action_linear = action[0][0]
        action_angular = action[0][1]

        twist = Twist()
        twist.linear.x = action_linear
        twist.angular.z = action_angular
        self.cmd_vel_pub.publish(twist)

        response.state = self.get_state(action_linear, action_angular)
        response.reward = self.get_reward(action_linear, action_angular)
        # print("step: {}, R: {:.3f}, A: {} GD: {:.3f}, GA: {:.3f}, MIND: {:.3f}, MINA: {:.3f}".format(
        print("step: {}, R: {:.3f}, A: {}".format(self.local_step, response.reward, action))
        response.done = self.done

        if self.done is True:
            self.done = False
            self.succeed = False
            self.collision = False

        return response

    """*******************************************************************************
    ** Below should be replaced when porting for ROS 2 Python tf_conversions is done.
    *******************************************************************************"""

    def euler_from_quaternion(self, quat):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quat = [x, y, z, w]
        """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w*y - z*x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    ddpg_environment = DDPGEnvironment()
    rclpy.spin(ddpg_environment)

    ddpg_environment.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
