#!/bin/bash
rostopic pub -1 -s /joint_states sensor_msgs/JointState '{header: auto, name: [joint1, joint2, joint3, joint4, joint5, joint6], position: [0, 0, 0, 0, 0, 0], effort: [], velocity: [] }'
