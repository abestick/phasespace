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
        self.rot = tf.transformations.quaternion_from_matrix(homog)
        self.rate = rate

    def run(self):
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

    def collect_samples(self, time=15.0):
        mocap_listener = rospy.Subscriber(self.mocap_topic, sensor_msgs.msg.PointCloud,
                self.new_frame_callback)
        rospy.sleep(time)
        mocap_listener.unregister()

    def new_frame_callback(self, msg):
        RATE = 5.0
        if rospy.get_time() >= self.last_sample + (1.0 / RATE):
            self.last_sample = rospy.get_time()
            try:
                (trans,rot) = self.tf_listener.lookupTransform(self.base_frame, self.parent_frame,
                        rospy.Time(0))
                point_cloud = point_cloud_to_array(msg)
                if not np.isnan(point_cloud[self.marker,0,0]):
                    self.mocap_samples.append(np.concatenate((point_cloud[self.marker,:,0], np.array([1])))[:,None])
                    homog = tf.transformations.quaternion_matrix(rot)
                    homog[0:3,3] = np.array(trans)
                    self.tf_samples.append(homog[:,:,None])
                    print('Collected sample at t=' + str(rospy.get_time()))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass
        
    def calibrate(self):
        # Concatenate all samples into arrays
        p_M = np.concatenate(self.mocap_samples, axis=1)
        g_BH = np.concatenate(self.tf_samples, axis=2)

        #Optimize the loss function
        loss_func = get_loss_function(p_M, g_BH)
        x0 = np.ones((10,))
        vectorized_loss_func = lambda params: loss_func(*unvectorize_params(params))
        result = scipy.optimize.minimize(vectorized_loss_func, x0)
        return unvectorize_params(result.x)[0]



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

def save_robot_calibration(filepath, robot_base_frame, homog):
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
    parser.add_argument('--frame', default='/mocap')
    args = parser.parse_args()
    rospy.init_node('robot_frame_publisher')

    if args.calibrate is not None:
        calib = RobotTransformCalibrator(args.calibrate[0], args.calibrate[1])
        calib.collect_samples()
        result = calib.calibrate()
        print(result)
        save_robot_calibration(args.data_file, '/base', result)

    # Load the calibration data from a file and publish
    robot_base_frame, homog = load_robot_calibration(args.data_file)
    transform_pub = StaticTransformPublisher(robot_base_frame, args.frame, homog, rate=50.0)
    transform_pub.run()

if __name__ == '__main__':
    main()