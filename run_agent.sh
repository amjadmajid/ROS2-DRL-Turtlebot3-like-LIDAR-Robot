#!/bin/bash

colcon build

## change these to whatever you actually need
command1="source install/setup.bash; sleep 2; ros2 run turtlebot3_dqn dqn_agent 3; bash"
command2="source install/setup.bash; sleep 2; ros2 run turtlebot3_dqn dqn_gazebo 3; bash"
command3="source install/setup.bash; sleep 2; ros2 run turtlebot3_dqn dqn_environment; bash"
command4="source install/setup.bash; ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage3.launch.py; bash"

## Modify terminator's config
sed -i.bak "s#COMMAND1#$command1#; s#COMMAND2#$command2#; s#COMMAND3#$command3#; s#COMMAND4#$command4#;" ~/.config/terminator/config

## Launch a terminator instance using the new layout
terminator -l turtle

## Return the original config file
mv ~/.config/terminator/config.bak ~/.config/terminator/config
