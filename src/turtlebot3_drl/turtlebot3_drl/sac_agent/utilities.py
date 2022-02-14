import matplotlib.pyplot as plt
import numpy
import os

from turtlebot3_msgs.srv import DrlStep
from turtlebot3_msgs.srv import Goal
import rclpy
import torch

class SACplot():
    def __init__(self, session_dir, plot_interval, episode, rewards, critic_loss, actor_loss, alpha_loss):
        plt.figure(figsize=(14,10))
        plt.axis([-50, 50, 0, 10000])
        plt.ion()
        plt.show()

        self.session_dir = session_dir
        self.plot_interval = plot_interval
        self.update_plots(episode, rewards, critic_loss, actor_loss, alpha_loss)

    def update_plots(self, episode, rewards, critic_loss, actor_loss, alpha_loss):
        
        xaxis = numpy.array(range(episode))
        x = xaxis
        y = numpy.array(critic_loss)
        plt.subplot(2, 2, 1)
        plt.gca().set_title('avg critic loss over episode')
        plt.plot(x, y)

        x = xaxis
        y = numpy.array(actor_loss)
        plt.subplot(2, 2, 2)
        plt.gca().set_title('avg actor loss over episode')
        plt.plot(x, y)

        x = xaxis
        y = numpy.array(alpha_loss)
        plt.subplot(2, 2, 3)
        plt.gca().set_title('avg alpha loss over episode')
        plt.plot(x, y)

        count = int(episode / self.plot_interval)
        if count > 0:
            x = numpy.array(range(self.plot_interval, episode+1, self.plot_interval))
            averages = list()
            for i in range(count):
                avg_sum = 0
                for j in range(self.plot_interval):
                    avg_sum += rewards[i * self.plot_interval + j]
                averages.append(avg_sum / self.plot_interval)
            y = numpy.array(averages)
            plt.subplot(2, 2, 4)
            plt.gca().set_title('avg reward over 10 episodes')
            plt.plot(x, y)

        plt.draw()
        plt.pause(0.001)
        plt.show()
        plt.savefig(os.path.join(self.session_dir, "_figure.png"))


def check_gpu():
    print("gpu torch available: ", torch.cuda.is_available())
    if (torch.cuda.is_available()):
        print("device name: ", torch.cuda.get_device_name(0))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def step(agent_self, action, previous_action):
    req = DrlStep.Request()
    req.action = action
    req.previous_action = previous_action

    while not agent_self.step_com_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('env step service not available, waiting again...')
    future = agent_self.step_com_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                return res.state, res.reward, res.done, res.success
            else:
                agent_self.get_logger().error(
                    'Exception while calling service: {0}'.format(future.exception()))
                print("ERROR getting step service response!")

def get_goal_status(agent_self):
    req = Goal.Request()
    while not agent_self.goal_com_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('new goal service not available, waiting again...')
    future = agent_self.goal_com_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                return res.new_goal
            else:
                agent_self.get_logger().error(
                    'Exception while calling service: {0}'.format(future.exception()))
                print("ERROR getting new_goal service response!")