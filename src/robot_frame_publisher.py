import rospy
import tf
import tf.transformations
import numpy as np
import numpy.linalg as npla
import sensor_msgs.msg
import scipy.optimize
import argparse
import json

def point_cloud_to_array(message):
    num_points = len(message.points)
    data = np.empty((num_points, 3, 1))
    for i, point in enumerate(message.points):
        data[i,:,0] = [point.x, point.y, point.z]
    return data

class StaticTransformPublisher:
    def __init__(self, parent_frame, child_frame, homog, rate=50.0):
        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self.trans = homog[0:3,3]
        self.rot = quaternion_from_matrix(homog)
        self.rate = rate

    def run():
        publisher = tf.TransformBroadcaster()
        timer = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            publisher.sendTransform(self.trans, self.rot, rospy.Time.now(),
                self.child_frame, self.parent_frame)
            timer.sleep()


class RobotTransformCalibrator:
    def __init__(self, base_frame, parent_frame, marker=0, mocap_topic='/mocap_point_cloud'):
        self.base_frame = base_frame
        self.parent_frame = parent_frame
        self.tf_listener = tf.TransformListener()
        self.marker = np.array(marker)
        self.mocap_topic = mocap_topic
        self.mocap_samples = []
        self.tf_samples = []
        self.last_sample = rospy.get_time()

    def collect_samples(time=15.0):
        mocap_listener = rospy.Subscriber(self.mocap_topic, sensor_msgs.msg.PointCloud,
                self.new_frame_callback)
        rospy.sleep(time)
        mocap_listener.unregister()

    def new_frame_callback(msg):
        RATE = 5.0
        if rospy.get_time() >= self.last_sample + (1.0 / RATE):
            self.last_sample = rospy.get_time()
            try:
                (trans,rot) = self.tf_listener.lookupTransform(self.base_frame, self.parent_frame,
                        rospy.Time(0))
                homog = tf.transformations.quaternion_matrix(rot)
                homog[0:3,3] = np.array(trans)
                tf_samples.append(homog)
                point_cloud = point_cloud_to_array(msg)
                mocap_samples.append(np.concatenate((point_cloud[marker,:,:], np.array([1]))))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
        
    def calibrate():
        # Concatenate all samples into arrays
        p_M = np.concatenate(mocap_samples, axis=1)
        g_BH = np.concatenate(tf_samples, axis=2)

        #Optimize the loss function
        loss_func = vectorize_loss_function(get_loss_function(p_M, g_BH))
        x0 = np.ones((10,))
        vectorized_loss_func = lambda params: loss_func(*unvectorize_params(params))
        result = scipy.optimize.minimize(vectorized_loss_func, x0)
        return result



def get_loss_function(p_M, g_BH):
    """Returns the loss function to optimize
    p_M (4, T) - marker positions vs. t
    g_BH (4, 4, T) - base to hand transformation matrix vs. t
    """
    if p_M.shape[1] != g_BH.shape[2]:
        raise ValueError('Number of samples in p_M and g_BH must match')

    def loss(g_BM, p_H):
        sum_squared_error = 0
        for t in range(p_M.shape[1]):
            left = g_BM.dot(p_M[:,t])
            right = g_BH[:,:,t].dot(p_H)
            sum_squared_error += np.sum((left - right) ** 2)
        return sum_squared_error
    return loss

def vectorize_params(g_BM, p_H):
    """Vectorizes the parameters of the loss function.
    params (10,) - [g_BM rotation (x,y,z,w), g_BM translation (x,y,z), p_H (x,y,z)]
    """
    params = np.zeros((10,))
    params[0:4] = tf.transformations.quaternion_from_matrix(g_BM)
    params[4:7] = g_BM[0:3,3]
    params[7:10] = p_H[0:3]
    return params

def unvectorize_params(params):
    rotation = params[0:4] / npla.norm(params[0:4])
    g_BM = tf.transformations.quaternion_matrix(rotation)
    g_BM[0:3,3] = params[4:7]
    p_H = np.hstack((params[7:10], np.array([1])))
    return g_BM, p_H

def save_robot_calibration(filepath, robot_name, robot_base_frame, homog):
    try:
        with open(filepath) as file_handle:
            data = json.load(file_handle)
    except IOError:
        # File doesn't exist yet
        data = {}
    object_dict = {'robot_base_frame': robot_base_frame,
                   'homog': homog.tolist()}
    with open(filepath, 'w') as file_handle:
        json.dump(object_dict, file_handle)

def load_robot_calibration(filepath):
    with open(filepath) as file_handle:
        data = json.load(file_handle)
    robot_base_frame = data['robot_base_frame']
    homog = np.asarray(data['homog'])
    return robot_base_frame, homog

def main():
    # Specify: Robot base frame, marker parent frame, data file
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='The file containing the robot calibration data')
    parser.add_argument('--calibrate', nargs=2)
    parser.add_argument('--frame', default='\mocap')
    args = parser.parse_args()

    if args.calibrate is not None:
        calib = RobotTransformCalibrator(args.calibrate[0], args.calibrate[1])
        calib.collect_samples()
        result = calib.calibrate()

    # Load the calibration data from a file and publish
    robot_base_frame, homog = load_robot_calibration(args.data_file)
    transform_pub = StaticTransformPublisher(self, robot_base_frame, args.frame, homog, rate=50.0)
    transform_pub.run()


    p_M_1 = np.array([1,2,3,1])
    p_M_2 = np.array([4,2,6,1])
    p_M_3 = np.array([0,2,5,1])
    p_M_4 = np.array([3,-1,5,1])
    p_M = np.vstack((p_M_1, p_M_2, p_M_3, p_M_4)).T
    g_BH_1 = tf.transformations.random_rotation_matrix()
    g_BH_1[0:3,3] = np.array([4,5,6])
    g_BH_2 = tf.transformations.random_rotation_matrix()
    g_BH_2[0:3,3] = np.array([4,0,1])
    g_BH_3 = tf.transformations.random_rotation_matrix()
    g_BH_3[0:3,3] = np.array([3,-2,1])
    g_BH_4 = tf.transformations.random_rotation_matrix()
    g_BH_4[0:3,3] = np.array([4,2,7])
    g_BH = np.dstack((g_BH_1, g_BH_2, g_BH_3, g_BH_4))
    loss_func = get_loss_function(p_M, g_BH)
    g_BM = tf.transformations.random_rotation_matrix()
    p_H = np.array([1,2,3,1])
    loss_func(g_BM, p_H)

    loss_func(*unvectorize_params(np.array([1,2,3,4,5,6,7,8,9,10])))

    1/0

if __name__ == '__main__':
    main()