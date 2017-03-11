Phasespace ROS Package
======================
This package contains tools for interfacing Phasespace mocap systems with ROS. It allows users to:
- Publish raw mocap marker coordinates as a ROS PointCloud
- Compute the rigid transformations between robots in a scene and the Phasespace coordinate system
- Define and track rigid bodies using ROS's standard _tf_ tools


Publish Live Mocap Data on a ROS Topic
--------------------------------------
To publish raw data on the `/mocap_point_cloud` topic, run:
```
roslaunch phasespace mocap.launch
```

or, to also start the RViz GUI to view the mocap data
```
roslaunch phasespace mocap_gui.launch
```


Align Mocap and Robot Coordinate Frames
---------------------------------------
In order to represent both mocap and robot pose data in one unified tf transform tree (useful if you'd like to view robot+mocap simultaneously in RViz or program a robot to interact with objects tracked using mocap markers), we need to tell tf the /robot->/mocap coordinate transform. The `publish_robot_transform.py` script will compute and publish this transform for you.

To compute and publish the robot->mocap transform, attach a single mocap marker (configured as marker 0) to the end of one of your robot's limbs. Start the mocap data publisher, as described above. Then run:
```
rosrun phasespace robot_frame_publisher.py <data_file> --calibrate <robot_base_frame> <marker_parent_frame>
```
where `<data_file>` is the name of the file in which to save the computed coordinate transform, `<robot_base_frame>` is the base tf frame for your robot, and `<marker_parent_frame>` is the tf frame to which your single mocap marker is rigidly attached (e.g. if you've attached the marker to Baxter's left gripper, you'd set this to `/left_hand`).

The script will collect calibration data for 15 seconds after you run this command. Move the robot's arm with the mocap marker attached through as wide a range of motion as possible until the script displays the computed transformation matrix. After this, the transformation will be automatically published. You can check this by verifying that the mocap point cloud is correctly aligned with the robot model in RViz.

If you've previously computed the robot->mocap transform for your robot and the robot hasn't moved, you can re-publish the existing transform with:
```
rosrun robot_frame_publisher.py <data_file>
```
where `<data_file>` is the file name you used when you originally calibrated this robot.