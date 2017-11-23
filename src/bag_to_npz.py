#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud


rospy.init_node('mocap_relay')
pub = rospy.Publisher('mocap_relay', PointCloud, queue_size=100)


def relay(msg):
    """

    :param PointCloud msg:
    :return:
    """
    idx = rospy.get_param('~idx', 0)
    msg.points = [msg.points[idx]]
    pub.publish(msg)


sub = rospy.Subscriber(rospy.get_param('~topic'), PointCloud, relay)
rospy.spin()
