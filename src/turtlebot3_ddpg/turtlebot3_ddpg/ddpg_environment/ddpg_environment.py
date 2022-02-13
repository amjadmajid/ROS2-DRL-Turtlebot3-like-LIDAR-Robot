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
import sys
from numpy.core.numeric import Infinity

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from turtlebot3_msgs.srv import Ddpg
from turtlebot3_msgs.srv import Goal
from std_srvs.srv import Empty
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data

INDEX_LIN = 0
INDEX_ANG = 1

NUM_SCAN_SAMPLES = 10
HARD_MAX_LIDAR_VALUE = 16  # in meters
MAX_LIDAR_VALUE = 10


class DDPGEnvironment(Node):
    def __init__(self, real_robot, is_training):
        super().__init__('ddpg_environment')

        self.episode = 0

        """************************************************************
        ** Initialise variables
        ************************************************************"""
        self.real_robot = real_robot
        self.is_training = is_training

        self.scan_topic = 'scan'
        if self.real_robot == True:
            self.vel_topic = 'jetbot/cmd_vel'
            self.goal_topic = 'jetbot_goal'
            self.odom_topic = 'odom_rf2o'
        else:
            self.vel_topic = 'cmd_vel'
            self.goal_topic = 'goal_pose'
            self.odom_topic = 'odom'

        # Change these parameters if necessary
        self.action_size = 2    # number of action types (e.g. linear velocity, angular velocity)
        if self.is_training == True: #TODO fix this statement
            self.step_limit = 500
        else:
            self.step_limit = 500 # maximum number of steps before episode timeout occurs
        self.time_penalty = -1  # negative reward for every step taken

        # No need to change below
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.last_pose_x = 0.0
        self.last_pose_y = 0.0
        self.last_pose_theta = 0.0

        self.previous_distance = 0

        self.done = False
        self.collision = False
        self.succeed = False

        self.new_goal = False
        self.goal_angle = 0.0
        self.goal_distance = Infinity
        self.init_goal_distance = Infinity
        self.scan_ranges = []
        self.previous_scan = [HARD_MAX_LIDAR_VALUE] * NUM_SCAN_SAMPLES
        self.min_obstacle_distance = HARD_MAX_LIDAR_VALUE

        self.local_step = 1
        self.received = False

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Initialise publishers
        self.cmd_vel_pub = self.create_publisher(Twist, self.vel_topic, qos)

        # Initialise subscribers
        self.goal_pose_sub = self.create_subscription(Pose, self.goal_topic, self.goal_pose_callback, qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos)
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, qos_profile=qos_profile_sensor_data)

        # Initialise client
        if self.real_robot != True:
            self.task_succeed_client = self.create_client(Empty, 'task_succeed')
            self.task_fail_client = self.create_client(Empty, 'task_fail')

        # Initialise servers
        self.ddpg_com_server = self.create_service(Ddpg, 'ddpg_com', self.ddpg_com_callback)
        self.goal_com_server = self.create_service(Goal, 'goal_com', self.goal_com_callback)

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""

    def goal_pose_callback(self, msg):
        self.goal_pose_x = msg.position.x
        self.goal_pose_y = msg.position.y
        self.new_goal = True

    def goal_com_callback(self, request, response):
        response.new_goal = self.new_goal
        return response

    def odom_callback(self, msg):
        self.last_pose_x = msg.pose.pose.position.x
        self.last_pose_y = msg.pose.pose.position.y
        _, _, self.last_pose_theta = self.euler_from_quaternion(
            msg.pose.pose.orientation)

        diff_y = self.goal_pose_y - self.last_pose_y
        diff_x = self.goal_pose_x - self.last_pose_x

        goal_distance = math.sqrt(diff_x**2 + diff_y**2)

        # Note: it appears the goal_angle got inverted during training.
        # This means the NN expects inverted values for goal_angle input
        # Thus *-1 to invert for the NN and non-inverted for debug output
        if self.real_robot == True:
            path_theta = math.atan2(-1 * diff_y, -1 * diff_x)
        else:
            path_theta = math.atan2(diff_y, diff_x)

        if path_theta < 0:
            path_theta += 2 * math.pi

        if self.last_pose_theta < 0:
            self.last_pose_theta += 2 * math.pi

        goal_angle = path_theta - self.last_pose_theta

        while goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        while goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        path_theta_print = math.atan2(diff_y, diff_x)
        goal_angle_print = path_theta_print - self.last_pose_theta
        # print(f"atan2: {math.degrees(path_theta_print):.3f}, theta: {math.degrees(self.last_pose_theta):.3f}, \
        # goal_angle: {math.degrees(goal_angle_print):.3f} Goal[X:{self.goal_pose_x}, Y:{self.goal_pose_y}]")

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle
        self.received = True

    def stop_reset_robot(self, success):
        self.episode += 1
        if self.episode > 200:
            self.step_limit = 800
        self.done = True
        self.cmd_vel_pub.publish(Twist())  # robot stop
        self.local_step = 1
        if self.real_robot == True:
            self.new_goal = False
        else:
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
        selected_scans = []
        if self.real_robot == True:
            for i in range(NUM_SCAN_SAMPLES):
                selected_scans[i] = msg.ranges[i * 20]
        else:
            selected_scans = msg.ranges

        for i in range(len(selected_scans)):
            if selected_scans[i] > HARD_MAX_LIDAR_VALUE:  # max value for rplidar A2
                selected_scans[i] = float(self.previous_scan[i])
            elif selected_scans[i] > MAX_LIDAR_VALUE:
                selected_scans[i] = 1
            else:
                # noramlize laser values
                selected_scans[i] = float(selected_scans[i]) / MAX_LIDAR_VALUE
        self.scan_ranges = selected_scans
        self.min_obstacle_distance = min(self.scan_ranges)
        self.previous_scan = selected_scans

    def get_state(self, previous_action_linear, previous_action_angular):
        state = self.scan_ranges
        state = state[:NUM_SCAN_SAMPLES]  # Truncate if too many laser readings are returned
        state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))
        state.append(float(previous_action_linear))
        state.append(float(previous_action_angular))
        self.local_step += 1

        # Succeed
        if self.goal_distance < 0.15 and self.local_step > 5:  # unit: m
            print("Goal! :)")
            self.succeed = True
            self.stop_reset_robot(True)

        # Fail
        if self.min_obstacle_distance < (0.15 / MAX_LIDAR_VALUE) and self.local_step > 5:  # unit: m
            print("Collision! :(")
            self.collision = True
            self.stop_reset_robot(False)

        # Timeout
        if self.local_step == self.step_limit:
            print("Time out! :(")
            self.collision = False
            self.stop_reset_robot(False)
        return state

    def get_reward(self, action_linear, action_angular):
        # yaw_reward will be between -1 and 1
        # yaw_reward = 3 - (3 * 2 * math.sqrt(math.fabs(self.goal_angle / math.pi)))

        # Between -1 and 0
        # yaw_reward = -1 * abs(self.goal_angle) / (math.pi)
        yaw_reward = 0

        # Between -4 and 0
        # angular_penalty = -0.5 * (action_angular**2)
        angular_penalty = 0

        # distance_reward = (2 * self.init_goal_distance) / (self.init_goal_distance + self.goal_distance) - 1
        # distance_reward = (self.previous_distance - self.goal_distance) * ((self.step_limit/100) / self.local_step) * 10
        distance_reward = (self.previous_distance - self.goal_distance) * 500
        # distance_reward = 0

        self.previous_distance = self.goal_distance

        # Reward for avoiding obstacles
        # if self.min_obstacle_distance < 0.25:
        #     obstacle_reward = -10
        # else:
        obstacle_reward = 0

        # Between -1 * (2.2^2) * 0.2 and 0
        # linear_penality = -1 * (((0.22 - action_linear) * 10) ** 2) * 0.2
        linear_penality = 0

        reward = yaw_reward + distance_reward + obstacle_reward + linear_penality + angular_penalty + self.time_penalty
        print("{:0>4} - Rdist: {:.3f}, Rangle: {:.3f}, Rspeed: {:.3f}".format(
            self.local_step, distance_reward, yaw_reward, linear_penality), end=' ')
        reward = reward 

        if self.succeed:
            reward += 400
        elif self.collision:
            reward -= 100
        return float(reward)

    def ddpg_com_callback(self, request, response):
        if len(request.action) == 0:
            self.init_goal_distance = math.sqrt(
                (self.goal_pose_x-self.last_pose_x)**2
                + (self.goal_pose_y-self.last_pose_y)**2)
            self.previous_distance = self.init_goal_distance
            response.state = self.get_state(0, 0)
            response.reward = 0.0
            response.done = False
            return response

        action = request.action
        action_linear = action[INDEX_LIN]
        action_angular = action[INDEX_ANG]

        twist = Twist()
        if self.real_robot:
            twist.linear.x = action_linear * 0.3 + 0.06
            twist.angular.z = action_angular * 0.3
        else:
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
        print(
            f"Rtot: {response.reward:.3f}, GD: {self.goal_distance:.3f}, GA: {math.degrees(self.goal_angle):.3f}° MinD: {self.min_obstacle_distance:.3f}, Alin: {action[0]:.3f}, Aturn: {action[1]:.3f}")
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


def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    ddpg_environment = DDPGEnvironment(args[0], args[1])
    rclpy.spin(ddpg_environment)

    ddpg_environment.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
