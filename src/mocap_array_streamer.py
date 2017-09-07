#!/usr/bin/env python
import argparse
import numpy as np
import phasespace.load_mocap as load_mocap
import rospy
from baxter_force_control.steppables import MocapMeasurement, PointCloudPublisher
from baxter_force_control.system import ForwardBlockNode, ForwardSystem, ForwardRoot


FRAMERATE = 50


def main():
    rospy.init_node('mocap_streamer')

    parser = argparse.ArgumentParser()
    parser.add_argument('mocap_npz')
    args = parser.parse_args(rospy.myargv()[1:])


    mocap_data = np.load(args.mocap_npz)['full_sequence']
    stream(mocap_data)


def stream(mocap_data, bag=None):
    # Put into a MocapArray
    mocap_array = load_mocap.ArrayMocapSource(mocap_data, FRAMERATE)

    print('Number of data points: %d' % len(mocap_array))

    # Define the measurement block
    mocap_measurement = MocapMeasurement(mocap_array, 'mocap_measurement')


    ##############################
    ## Build up the system nodes##
    ##############################
    measurement_node = ForwardBlockNode(mocap_measurement, 'Mocap Measurement', 'raw_mocap')

    # Define the root (all source nodes)
    root = ForwardRoot([measurement_node])

    # Create the system
    system = ForwardSystem(root)

    measurement_node.add_raw_output(PointCloudPublisher('mocap_point_cloud', 'world', bag, system.get_time),
                                    'PointCloud Publisher', None, 'states')

    return system.run_timed(FRAMERATE, record=False, print_steps=50)


if __name__ == '__main__':
    main()
