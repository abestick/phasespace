#!/usr/bin/env python
from __future__ import print_function
import rospy
import time
import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
import load_mocap
import sys
import argparse

def main():
    
    #Read ROS parameter arguments, if present
    #phasespace_ip_addr="..."
    #framerate=50
    # ros_params = rospy.get_param('~')

    # #Read command line options
    # parser = argparse.ArgumentParser()
    # parser.add_argument('server_ip')
    # parser.add_argument('-f', '--framerate', default=1, type=float, help='Desired framerate')
    # args = parser.parse_args()
    
    #Initialize the ROS node
    pub = rospy.Publisher('mocap_point_cloud', sensor_msgs.PointCloud)
    rospy.init_node('mocap_streamer')

    #Load the mocap stream
    with load_mocap.PhasespaceStream('192.168.1.120', num_points=32, 
                                     framerate=50, buffer_length=2) as mocap:

        #Play the points from the mocap stream
        #Loop until the node is killed with Ctrl+C
        frame_num = 0
        while 1:
            frame = mocap.read(block=True)[0].squeeze()
            if rospy.is_shutdown():
                break
            
            print('STREAMING: Frame ' + str(frame_num), end='\r')
            sys.stdout.flush()
            
            #Construct and publish the message
            message = sensor_msgs.PointCloud()
            message.header = std_msgs.Header()
            message.header.frame_id = 'mocap'
            #message.header.time = rospy.get_rostime()
            message.points = []
            for i in range(frame.shape[0]):
                point = geometry_msgs.Point32()
                point.x = frame[i,0]/1000
                point.y = frame[i,1]/1000
                point.z = frame[i,2]/1000
                message.points.append(point)
            pub.publish(message)
            frame_num += 1
            
    print('\rSTOPPED: Frame ' + str(frame_num))

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
