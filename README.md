# This repository contains the code for CPR2L.


# **INSTALLATION INSTRUCTIONS**

## **Dependencies**

*   Ubuntu 20.04 LTS (Focal Fossa) [download](https://releases.ubuntu.com/20.04)
*   ROS2 Foxy Fitzroy [download](https://github.com/ros2/ros2/releases?after=release-dashing-20200722)
*   Gazebo
*   PyTorch


## **Installing ROS2**
Install ROS2 foxy according to the following guide: [link](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html). <br>
To prevent having to manually source the setup script everytime, add the following line at the end of your ~/.bashrc file:

```
source /opt/ros/foxy/setup.bash
```
Note: for some installations it might also be required to add the following line in bashrc in addition to the previous source script:
```
source ~/ros2_foxy/ros2-linux/setup.bash
```

You might receive a warning about Connext not being supported but you can ignore this. <br>
Make sure to run the *talker* and *listener* example as describe in the link.
<br>
Alternative installation can be found guide [here](https://automaticaddison.com/how-to-install-ros-2-foxy-fitzroy-on-ubuntu-linux/).


## **Installing Gazebo**

For this project we will be using Gazebo **11.0.** To install Gazebo 11.0, navigate to the following [page](http://gazebosim.org/tutorials?tut=install_ubuntu), select Version 11.0 in the top-right corner and follow the regular installation instructions.

Next, we need to install a package which allows ROS2 to interface with Gazebo. 
To install this package we only need to execute the following command in a terminal:
```
sudo apt install ros-foxy-gazebo-ros-pkgs
```
After successfull installation we are now going to test our ROS2 + Gazebo setup by making a demo model move in the simulator. First install two additional packages for demo purposes (they might already be installed):
```
sudo apt install ros-foxy-ros-core ros-foxy-geometry2
```
Now, let's load the demo model in gazebo:
```
gazebo --verbose /opt/ros/foxy/share/gazebo_plugins/worlds/gazebo_ros_diff_drive_demo.world
```
This should launch the Gazebo GUI with a simple vehicle model. Open a second terminal and provide the following command to make the vehicle move:
```
ros2 topic pub /demo/cmd_demo geometry_msgs/Twist '{linear: {x: 1.0}}' -1
```
The vehicle should start moving forward. It is configured to listen to *geometry_msgs/Twist* messages which can be used to signal a velocity.

## **Installing Python3, Pytorch** 

If you are using Ubuntu 20.04 as specified in this guide, then Python should already be preinstalled. The last tested vesion for this project was Python 3.8.10

Install pip3 (python package manager for python 3) as follows:
```
sudo apt install python3-pip
```

To install our version of PyTorch, run:
```
pip3 install torch==1.4.0
```


## **Downloading the code base and building**
<!-- Now it's time to create a workspace which will serve as the basis for our project. To do this, follow the tutorial [here](https://automaticaddison.com/how-to-create-a-workspace-ros-2-foxy-fitzroy/) -->

Now it's time to download the repo and run the project:  https://github.com/amjadmajid/longNav

Since ROS2 does not yet support metapackages, we will have to download the whole workspace from Git. 

Open a terminal in the desired location for the new workspace. Clone the repository using:
```
git clone git@github.com:amjadmajid/longNav.git
```

Checkout the DDPG pytorch branch using
```
git checkout feature/ddpg-pytorch
```

Next, install the correct rosdep tool
```
sudo apt install python3-rosdep2
```

Then initalize rosdep by running
```
rosdep update
```

Now we can use rosdep to install all ROS packaged needed by our repository
```
rosdep install -i --from-path src --rosdistro foxy -y
```

Now that we have all of the packages in place it is time to build the repository. First update your package list
```
sudo apt update
```

Then install the build tool **colcon** which we will use to build our ROS2 package
```
sudo apt install python3-colcon-common-extensions
```

Next, it's time to actually build the ros package!
```
colcon build
```

The last thing to do before we can run is add some exports and sources to our ~/.bashrc file which runs every time we open a new terminal. Add all of the following lines at the end of your ~/.bashrc file and fill in the directory to your workspace as WORKSPACE_DIR:
```
# ROS2 domain id for network communication
export ROS_DOMAIN_ID=3

# Path to our workspace
WORKSPACE_DIR = ~/code/thesis/longNav

export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$WORKSPACE_DIR/src/turtlebot3_simulations/turtlebot3_gazebo/models

# Select which turtlebot model we will be using
export TURTLEBOT3_MODEL=burger

# Plugin for moving obstacles in stage 4
export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:$WORKSPACE_DIR/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_dqn_world/obstacle_plugin/lib
```

(check the steps in [this guide](https://automaticaddison.com/how-to-create-a-workspace-ros-2-foxy-fitzroy/) for additional instructions if something doesn't work)

Note: Make sure to first run the ```install/local_setup.bash``` after building a ros package before running the node.

## **Running and training the DDPG agent!**

Now that we have finally completed the setup, all that's left to do is run and train the agent. 

Open up four different terminals however you like. In the first terminal run
```
ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage4.launch.py
```
You should see the gazebo GUI come up with the robot model loaded and two moving obstacles (this might take a while to load).

In a second terminal run
```
ros2 run turtlebot3_dqn dqn_gazebo 4 
```
In a third terminal run
```
ros2 run turtlebot3_ddpg ddpg_environment 
```
And lastly, in the fourth terminal run
```
ros2 run turtlebot3_ddpg ddpg_agent 4
```

Your robot should now be moving and logging output is being printed to the terminals!

### switch environments

You can can switch to the 3 other environments by changing the 4 in all of the commands to 1, 2 or 3.

### ddpg_agent: change parameters, reload trained agents

Furthermore, ddpg_agent.py contains most of the interesting code and parameters that you might wish to change. This is also where you can load different models and choose to enable training or not, which is all documented in the code itself.

### ddpg_environment: tweak reward design

ddpg_environment.py contains other important aspects that you will want to play with such as the reward design.
