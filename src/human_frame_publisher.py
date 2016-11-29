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

#NOT YET WORKING
BASE_MARKERS = ['back', 'chest']

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help='The directory containing the calibration data')
args = parser.parse_args()

PREFIX = args.data_dir + '/'
ASSIGNMENTS = 'assignments.json'
NPZ = 'right_arm_skel_fit.npz'

class MocapFramePublisher():
    def __init__(self, base_markers, desired, skip_frames=3):
        self._desired = desired
        self._base_markers = base_markers
        self._frame_num = 0
        self._skip_frames = skip_frames
        self._last_rot = np.array([0.0,0,0])
        self._br = tf.TransformBroadcaster()
        self._sub = rospy.Subscriber('/mocap_point_cloud', 
                                     sensor_msgs.PointCloud, 
                                     self._new_frame_callback)

    def _new_frame_callback(self, message):
        print('call')
        if self._frame_num % self._skip_frames == 0:
            data = point_cloud_to_array(message)[self._base_markers,:,0]            

            #Remove markers which are not visible
            visible = np.where(np.logical_not(np.isnan(data[:,0])))[0]
            data = data[visible, :]
            desired = self._desired[visible,:]

            if len(visible) >= 3:
                #Compute the current transform
                homog, rot = load_mocap.find_homog_trans(data, desired, rot_0=self._last_rot)
                self._last_rot = rot
                homog = la.inv(homog)
                print(homog)
                print(rot)

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
                         '/human',
                         '/mocap')


class MocapFrameCalibrator():
    def __init__(self, base_markers, skip_frames=3):
        self._base_markers = base_markers
        self._skip_frames = skip_frames
        self._br = tf.TransformBroadcaster()
        self._sub = rospy.Subscriber('/mocap_point_cloud', 
                                     sensor_msgs.PointCloud, 
                                     self._new_frame_callback)

    def _new_frame_callback(self, message):
        print('call')
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

def find_homog_trans(points_a, points_b, err_threshold=0, rot_0=None):
    """Finds a homogeneous transformation matrix that, when applied to 
    the points in points_a, minimizes the squared Euclidean distance 
    between the transformed points and the corresponding points in 
    points_b. Both points_a and points_b are (n, 3) arrays.
    """
    #OLD ALGORITHM ----------------------
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
    #---------------------------------------


    # #Define the error function
    # def error(state):
    #     rot = state[0:3]
    #     trans = state[3:6]

    #     #Construct a homography matrix
    #     homog = np.eye(4)
    #     homog[0:3,3] = trans
    #     homog[0:3,0:3] = vec_to_rot(rot)

    #     #Transform points_a
    #     points_a_h = np.hstack((points_a, np.ones((points_a.shape[0],1))))
    #     trans_points_a = homog.dot(points_a_h.T).T[:,0:3]

    #     #Compute the error
    #     err = la.norm(points_a - trans_points_a, axis=1)**2
    #     return err
    
    # #Run the optimization
    # if rot_0 == None:
    #     rot_0 = sp.zeros(6)
    # rot = opt.leastsq(error, rot_0, ftol=1e-20)[0]
    
    # #Compute the final homogeneous transformation matrix
    # homog = np.eye(4)
    # homog[0:3,3] = rot[3:6]
    # homog[0:3,0:3] = vec_to_rot(rot[0:3])
    # return homog, rot

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

def main():
    rospy.init_node('human_frame_publisher')

    #Publish frame --------------------------------
    #Read the desired coordinates from a file
    # assignments = None
    # with open(PREFIX + ASSIGNMENTS, 'r') as ass_file:
    #     assignments = json.load(ass_file)['assignments']

    # base_indices = []
    # for group in BASE_MARKERS:
    #     base_indices.extend(assignments[group])

    # desired = np.load(PREFIX+NPZ)['base_config']

    # frame_publisher = MocapFramePublisher(base_indices, desired)
    # rospy.spin()

    #Calibrate frame -----------------------------------
    frame_calibrator = MocapFrameCalibrator([0,1,2,3])

    rospy.spin()

if __name__ == '__main__':
    main()
