#!/usr/bin/env python
import argparse
import rospy
import tf
import tf.transformations as convert
import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs
import std_msgs.msg as std_msgs
import json
import numpy as np
import numpy.linalg as la
import scipy.linalg as spla
import scipy.optimize as opt
import scipy as sp
import sys
import pprint

# Sample command to calibrate a rigid with markers 0-4
# rosrun phasespace rigid_tracker.py data.txt --calibrate 0 1 2 3 4 --frame /hand

# Sample command to track the same rigid and publish the transform as /hand
# rosrun phasespace rigid_tracker.py data.txt --frame /hand

DEFAULT_FILE = 'rigid_bodies.txt'

parser = argparse.ArgumentParser()
parser.add_argument('data_file', help='The file containing the rigid body data')
parser.add_argument('--calibrate', nargs='+', type=int)
parser.add_argument('--frame', default='\human')
args = parser.parse_args()

class MocapFramePublisher():
    def __init__(self, base_markers, desired, frame, skip_frames=3):
        self._desired = desired
        self._base_markers = base_markers
        self._frame_num = 0
        self._skip_frames = skip_frames
        self._last_rot = np.array([0.0,0,0])
        self._br = tf.TransformBroadcaster()
        self._sub = rospy.Subscriber('/mocap_point_cloud', 
                                     sensor_msgs.PointCloud, 
                                     self._new_frame_callback)
        self._frame = frame

    def _new_frame_callback(self, message):
        if self._frame_num % self._skip_frames == 0:
            data = point_cloud_to_array(message)[self._base_markers,:,0]            

            #Remove markers which are not visible
            visible = np.where(np.logical_not(np.isnan(data[:,0])))[0]
            data = data[visible, :]
            desired = self._desired[visible,:]

            if len(visible) >= 3:
                #Compute the current transform
                homog, rot = find_homog_trans(data, desired, rot_0=self._last_rot)
                self._last_rot = rot
                homog = la.inv(homog)
                print(homog)

                #Compute the translation and rotation components
                translation = convert.translation_from_matrix(homog)
                rotation = convert.quaternion_from_matrix(homog)

                #Publish the transform
                self.publish_transform(translation, rotation)

        self._frame_num += 1

    def publish_transform(self, translation, rotation):
        self._br.sendTransform(translation,
                         rotation,
                         rospy.Time.now(),
                         self._frame,
                         '/mocap')


class MocapFrameCalibrator():
    def __init__(self, base_markers, skip_frames=1):
        self._frame_num = 0
        self._base_markers = base_markers
        self._skip_frames = skip_frames
        self._br = tf.TransformBroadcaster()
        self._sub = rospy.Subscriber('/mocap_point_cloud', 
                                     sensor_msgs.PointCloud, 
                                     self._new_frame_callback)

    def _new_frame_callback(self, message):
        if self._frame_num % self._skip_frames == 0:
            data = point_cloud_to_array(message)[self._base_markers,:,0]            

            #Remove markers which are not visible
            visible = np.where(np.logical_not(np.isnan(data[:,0])))[0]
            data = data[visible, :]

            #Compute the centroid of the observed points
            centroid = np.mean(data, axis=0)

            #Compute the translation and rotation components
            translation = centroid.squeeze()
            rotation = convert.quaternion_from_matrix(convert.identity_matrix())

            #Publish the transform
            self.publish_transform(translation, rotation)

            #Compute the transformation matrix
            homog = convert.identity_matrix()
            homog[0:3,3] = translation

            #Transform points into rigid body frame
            homog_data = np.vstack((data.T, np.ones((1, data.shape[0]))))
            desired = la.inv(homog).dot(homog_data)
            print(desired)

            #Save the marker arrangement if all markers are visible
            if desired.shape[1] == len(self._base_markers):
                try:
                    save_rigid_body_config(args.data_file, 'test_body', self._base_markers, desired[0:3,:])
                except IOError:
                    print('Error: Unable to save rigid body config')
                rospy.signal_shutdown('Captured all markers in a single frame') 

        self._frame_num += 1

    def publish_transform(self, translation, rotation):
        self._br.sendTransform(translation,
                         rotation,
                         rospy.Time.now(),
                         '/human',
                         '/mocap')

def point_cloud_to_array(message):
    num_points = len(message.points)
    data = np.empty((num_points, 3, 1))
    for i, point in enumerate(message.points):
        data[i,:,0] = [point.x, point.y, point.z]
    return data

def find_homog_trans(points_a, points_b, err_threshold=0, rot_0=None, alg='svd'):
    """Finds a homogeneous transformation matrix that, when applied to 
    the points in points_a, minimizes the squared Euclidean distance 
    between the transformed points and the corresponding points in 
    points_b. Both points_a and points_b are (n, 3) arrays.
    """
    #OLD ALGORITHM ----------------------
    if alg == 'opt':
        #Align the centroids of the two point clouds
        cent_a = sp.average(points_a, axis=0)
        cent_b = sp.average(points_b, axis=0)
        points_a = points_a - cent_a
        points_b = points_b - cent_b
        
        #Define the error as a function of a rotation vector in R^3
        rot_cost = lambda rot: (sp.dot(vec_to_rot(rot), points_a.T).T
                        - points_b).flatten()**2
        
        #Run the optimization
        if rot_0 == None:
            rot_0 = sp.zeros(3)
        rot = opt.leastsq(rot_cost, rot_0)[0]
        
        #Compute the final homogeneous transformation matrix
        homog_1 = sp.eye(4)
        homog_1[0:3, 3] = -cent_a
        homog_2 = sp.eye(4)
        homog_2[0:3,0:3] = vec_to_rot(rot)
        homog_3 = sp.eye(4)
        homog_3[0:3,3] = cent_b
        homog = sp.dot(homog_3, sp.dot(homog_2, homog_1))
        return homog, rot
    #NEW ALGORITHM -----------------------
    elif alg == 'svd':
        homog = convert.superimposition_matrix(points_a.T, points_b.T)
        return homog, None


def vec_to_rot(x):
    #Make a 3x3 skew-symmetric matrix
    skew = sp.zeros((3,3))
    skew[1,0] = x[2]
    skew[0,1] = -x[2]
    skew[2,0] = -x[1]
    skew[0,2] = x[1]
    skew[2,1] = x[0]
    skew[1,2] = -x[0]
    
    #Compute the rotation matrix
    rot = spla.expm(skew)
    return rot

def save_rigid_body_config(filepath, body_name, marker_indices, desired):
    try:
        with open(filepath) as file_handle:
            data = json.load(file_handle)
    except IOError:
        # File doesn't exist yet
        data = {}
    object_dict = {'type': 'rigid_body',
                   'marker_indices': marker_indices,
                   'desired': desired.tolist()}
    data[body_name] = object_dict
    with open(filepath, 'w') as file_handle:
        json.dump(data, file_handle)

def load_rigid_body_config(filepath, body_name):
    with open(filepath) as file_handle:
        data = json.load(file_handle)
    desired = np.asarray(data[body_name]['desired']).T
    return data[body_name]['marker_indices'], desired

def main():
    print(args.calibrate)
    print(args.frame)
    rospy.init_node('human_frame_publisher', anonymous=True)
    if args.calibrate:
        frame_calibrator = MocapFrameCalibrator(args.calibrate)
    else:
        base_indices, desired = load_rigid_body_config(args.data_file, 'test_body')
        frame_publisher = MocapFramePublisher(base_indices, desired, args.frame)
    rospy.spin()

if __name__ == '__main__':
    main()
