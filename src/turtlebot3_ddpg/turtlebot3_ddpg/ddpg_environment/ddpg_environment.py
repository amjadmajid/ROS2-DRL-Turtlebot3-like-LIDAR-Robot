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
import numpy
from numpy.core.numeric import Infinity

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from turtlebot3_msgs.srv import Ddpg
from std_srvs.srv import Empty
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

INDEX_LIN = 0
INDEX_ANG = 1


class DDPGEnvironment(Node):
    def __init__(self):
        super().__init__('ddpg_environment')

        """************************************************************
        ** Initialise variables
        ************************************************************"""

        # Change these parameters if necessary
        self.action_size = 2    # number of action types (e.g. linear velocity, angular velocity)
        self.step_limit = 7500  # maximum number of steps before episode timeout occurs
        self.time_penalty = -1  # negative reward for every step taken

        # No need to change below
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.last_pose_x = 0.0
        self.last_pose_y = 0.0
        self.last_pose_theta = 0.0

        self.done = False
        self.collision = False
        self.succeed = False

        self.goal_angle = 0.0
        self.goal_distance = Infinity
        self.init_goal_distance = Infinity
        self.scan_ranges = []
        self.min_obstacle_distance = 3.5

        self.local_step = 0
        self.received = False

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Initialise publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)

        # Initialise subscribers
        self.goal_pose_sub = self.create_subscription(Pose, 'goal_pose', self.goal_pose_callback, qos)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, qos_profile=qos_profile_sensor_data)

        # Initialise client
        self.task_succeed_client = self.create_client(Empty, 'task_succeed')
        self.task_fail_client = self.create_client(Empty, 'task_fail')

        # Initialise servers
        self.ddpg_com_server = self.create_service(Ddpg, 'ddpg_com', self.ddpg_com_callback)

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

    def stop_reset_robot(self, success):
        self.cmd_vel_pub.publish(Twist())  # robot stop
        self.local_step = 0
        req = Empty.Request()
        if success:
            while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.task_succeed_client.call_async(req)
        else:
            while not self.task_fail_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.task_fail_client.call_async(req)

    def scan_callback(self, msg):
        for i in range(len(msg.ranges)):
            if msg.ranges[i] > 3.5:  # max range is specified in model.sdf
                msg.ranges[i] = 3.5
        self.scan_ranges = msg.ranges
        self.min_obstacle_distance = min(self.scan_ranges)
        self.received = True

    def get_state(self, previous_action_linear, previous_action_angular):
        state = self.scan_ranges.tolist()
        state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))
        state.append(float(previous_action_linear))
        state.append(float(previous_action_angular))
        self.local_step += 1

        # Succeed
        if self.goal_distance < 0.20 and self.local_step > 5:  # unit: m
            print("Goal! :)")
            self.succeed = True
            self.done = True
            self.stop_reset_robot(True)

        # Fail
        if self.min_obstacle_distance < 0.130 and self.local_step > 5:  # unit: m
            print("Collision! :( step: %d, %d", self.local_step, self.min_obstacle_distance)
            self.collision = True
            self.done = True
            self.stop_reset_robot(False)

        # Timeout
        if self.local_step == self.step_limit:
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
        # yaw_reward = 3 - (3 * 2 * math.sqrt(math.fabs(self.goal_angle / math.pi)))

        # Between -3.14 and 0
        yaw_reward = (math.pi - abs(self.goal_angle)) - math.pi

        # Between -4 and 0
        angular_penalty = -1 * (action_angular**2)

        distance_reward = (2 * self.init_goal_distance) / (self.init_goal_distance + self.goal_distance) - 1
        #distance_reward = 0

        # Reward for avoiding obstacles
        if self.min_obstacle_distance < 0.22:
            obstacle_reward = -40
        else:
            obstacle_reward = 0

        # Between -2 * (2.2^2) and 0
        linear_penality = -1 * (((0.22 - action_linear) * 10) ** 2)

        reward = yaw_reward + distance_reward + obstacle_reward + linear_penality + angular_penalty + self.time_penalty
        print("R_angle: {:.3f}, R_obst: {:.3f}, R_speed: {:.3f}, R_turning: {:.3f}".format(
            yaw_reward, obstacle_reward, linear_penality, angular_penalty))

        if self.succeed:
            reward += 8000
        elif self.collision:
            reward -= 7000
        return float(reward)

    def ddpg_com_callback(self, request, response):
        if len(request.action) == 0:
            self.init_goal_distance = math.sqrt(
                (self.goal_pose_x-self.last_pose_x)**2
                + (self.goal_pose_y-self.last_pose_y)**2)
            response.state = self.get_state(0, 0)
            response.reward = 0.0
            response.done = False
            return response

        action = request.action
        action_linear = action[INDEX_LIN]
        action_angular = action[INDEX_ANG]

        twist = Twist()
        twist.linear.x = action_linear
        twist.angular.z = action_angular
        self.cmd_vel_pub.publish(twist)

        # TODO: should there be some kind of delay here to balance laser update rate and vel publish
        # self.received = False
        # while rclpy.ok():
        #     while self.received == False:
        #         rclpy.spin_(self)
        #         print("waiting for laser")

        previous_action_linear = request.previous_action[INDEX_LIN]
        previous_action_angular = request.previous_action[INDEX_ANG]
        response.state = self.get_state(previous_action_linear, previous_action_angular)
        response.reward = self.get_reward(action_linear, action_angular)
        print("step: {}, GD: {:.3f}, GA: {:.3f}° A0: {:.3f}, A1: {:.3f}, R: {:.3f}".format(
            self.local_step, self.goal_distance, math.degrees(self.goal_angle), action[0], action[1], response.reward))
        response.done = self.done
        response.success = self.succeed

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
        if sinp < -1:
            sinp = -1
        if sinp > 1:
            sinp = 1
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
