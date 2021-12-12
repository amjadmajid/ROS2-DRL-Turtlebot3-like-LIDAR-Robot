
- clone repo
- git submodule update --init
- set $ROS_DOMAIN_ID to same number as on jetbot in bashrc
- ros2 topic pub --once /goal_pose geometry_msgs/msg/Pose "{position: {x: 1.0, y: 2.0}}"

