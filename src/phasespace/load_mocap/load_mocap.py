#!/usr/bin/env python
"""Loads data from C3D format files. This module provides a simple 
wrapper for the C3D-reading functionality of the Biomechanical Toolkit
(https://code.google.com/p/b-tk/). At present, it just provides methods 
to load a file, get basic metadata, and read individual samples or 
groups of samples.

In the future, the MocapFile class could be modified to support
mocap file formats other than C3D, such as BVH, CSV, etc.

Author: Aaron Bestick
"""
from __future__ import print_function
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy as sp
import scipy.linalg as spla
import numpy.linalg as la
from collections import deque
import matplotlib.pyplot as plt
import scipy.optimize as opt
import Queue
from threading import Thread, RLock
import time
from copy import deepcopy
import tf.transformations as transformations

# OWL is only needed by PhasespaceStream
try:
    import OWL
except(ImportError):
    pass

class MocapSource(object):
    """Base class for online and offline mocap sources.

    After initializing a MocapSource, multiple clients can access the data from the source through
    MocapStreams, like:
    >>> source = MocapSource()
    >>> stream = source.get_stream()
    >>> frame, timestamp = stream.read()
    >>> for (frame, timestamp) in stream:
    >>>     ...
    """
    __metaclass__=ABCMeta

    def __init__(self):
        self._buffers = []

    @abstractmethod
    def get_num_points(self):
        """Get the number of points in each frame from this source (i.e. frames.shape[0])
        """
        pass

    @abstractmethod
    def get_length(self):
        """Get the length (total # of frames) in this MocapSource.

        Sources for which this length is known a priori (e.g. offline sources) should return the
        actual length of the offline data source. Online sources with unknown length should just
        return length=0 (necessary because the len() builtin requires a value >=0).
        """
        pass

    @abstractmethod
    def get_stream(self):
        """Returns a new MocapStream for this source.

        This should be the typical way to get data from a mocap source. The method is left abstract
        here because offline and online MocapSources will want different buffer lengths for their
        streams. Online sources should have a short buffer length to prevent out of memory errors if
        a stream object is not properly closed after use, while offline sources will want a buffer
        length equal to the length of the entire offline dataset.
        """
        pass

    def get_framerate(self):
        """Get the framerate of this mocap source in Hz.

        Sources for which the framerate is known and constant should override this method with one
        that returns the correct framerate. Other sources should use the default implementation, 
        which returns None and allows MocapStreams connected to this source to compute the framerate
        directly from the frames' timestamps.
        """
        return None

    def register_buffer(self, buffer):
        """
        Takes in the reference to a queue-like object and populates it with mocap data processed.
        
        Args:
        buffer: MocapStream (or any object with a .put() method) - the buffer to register
        """
        self._buffers.append(buffer)

    def unregister_buffer(self, buffer):
        """Unregister a previously registered buffer

        Args:
        buffer: reference to the buffer to unregister
        """
        self._buffers.remove(buffer)

    def __len__(self):
        return self.get_length()


class OnlineMocapSource(MocapSource):
    """Base class for real-time streaming MocapSource instances.

    These sources receive their data from a source such as a mocap system or a ROS topic,
    so they don't have a predefined length (len(source)=0). By default, OnlineMocapSources don't
    do any buffering of their own, so buffers registered with them will receive only the frames
    which arrive after the buffer is registered.
    """
    __metaclass__=ABCMeta
    DEFAULT_BUFFER_LEN = 10

    def __init__(self):
        super(OnlineMocapSource, self).__init__()

    def _output_frame(self, frame, timestamp):
        """Writes a new frame to each of the currently registered buffers.

        Args:
        frame: (num_markers, 3, 1) ndarray - the frame data
        timestamps: (1,) ndarray - the corresponding timestamp
        """
        for buffer in self._buffers:
            buffer.put((frame, timestamp))

    def get_length(self):
        """Return a length of 0 for online MocapSources (since the real length is undetermined)
        """
        return 0

    def get_stream(self):
        """Gets a stream for this MocapSource.

        The stream will be created with a short max length to prevent memory leaks if it
        receives data for an extended time without being read from or closed.
        """
        return MocapStream(self, max_buffer_len=self.DEFAULT_BUFFER_LEN)

class OfflineMocapSource(MocapSource):
    __metaclass__=ABCMeta

    def __init__(self, frames, timestamps):
        """Initialize an OfflineMocapSource with the frames and timestamps ndarrays from the file,
        array, etc.

        These sources are backed by a static data source like an array or a file. Pointers to this
        backing data source are passed to the constructor. When a new buffer is registered with
        this source, all the frames from the mocap source will be immediately written to it,
        followed by a None value to indicate the end of the data source. Unlike OnlineMocapSources,
        every buffer will receive exactly the same frames from this source, regardless of when it
        is registered.

        Args:
        frames: (num_markers, 3, num_frames) ndarray - the backing array of all mocap frames
        timestamps: (num_frames,) - array of corresponding timestamps for each frame
        """
        super(OfflineMocapSource, self).__init__()
        if frames.ndim != 3 or frames.shape[1] != 3:
            raise ValueError('Frames array must be (num_markers, 3, num_frames)')
        if timestamps.ndim != 1 or timestamps.shape[0] != frames.shape[2]:
            raise ValueError('Timestamps array must be (num_frames,)')
        self._frames = frames
        self._timestamps = timestamps

    def register_buffer(self, buffer):
        """Register a buffer a new buffer.

        All frames from this MocapSource's backing arrays will be immediately written to the new
        buffer, followed by a None value to indicate the end of the data. To avoid missing frames,
        ensure the buffer has a max_length>=len(source)+1.

        Args:
        buffer: MocapStream (or any object with a .put() method) - the buffer to register
        """
        super(OfflineMocapSource, self).register_buffer(buffer)
        for i in range(self._frames.shape[2]):
            buffer.put((self._frames[:,:,i:i+1], self._timestamps[i:i+1]))
        buffer.put(None)

    def _get_frames(self):
        """Returns a (num_points, 3, length) array of the mocap points.
        """
        return self._frames
    
    def _get_timestamps(self):
        """Returns a 1-D array of the timestamp (in seconds) for each frame
        """
        return self._timestamps

    def get_framerate(self):
        """Returns the average framerate of the mocap file in Hz"""
        duration = self._timestamps[-1] - self._timestamps[0]
        framerate = (len(self) - 1) / duration
        return framerate

    def get_num_points(self):
        """Returns the total number of points tracked in the mocap file
        """
        return self._get_frames().shape[0]
    
    def get_length(self):
        """Returns the total number of frames in the mocap file
        """
        return self._get_frames().shape[2]

    def plot_frame(self, frame_num=None, mark_num=None, xyz_rotation=(0,0,0), azimuth=60,
            elevation=30):
        """Plots the location of each marker in the specified frame
        """
        #Get the frame
        if frame_num is not None:
            frame = self._get_frames()[:,:,frame_num]
        else:
            frame = self.read()[0][:,:,0]

        # Apply any desired rotation
        rot_matrix = transformations.euler_matrix(*xyz_rotation)[0:3,0:3]
        frame = rot_matrix.dot(frame.T).T
        xs = frame[:,0]
        ys = frame[:,1]
        zs = frame[:,2]
        
        #Make the plot
        figure = plt.figure(figsize=(11,10))
        axes = figure.add_subplot(111, projection='3d')
        markers = ['r']*self.get_num_points()
        if mark_num is not None:
            markers[mark_num] = 'b'
        axes.scatter(xs, ys, zs, c=markers, marker='o', s=60)
        axes.auto_scale_xyz([-0.35,0.35], [-0.35,0.35], [-0.35,0.35])
        axes.set_xlabel('X (m)')
        axes.set_ylabel('Y (m)')
        axes.set_zlabel('Z (m)')
        # axes.view_init(elev=elevation, azim=azimuth)

    def get_stream(self):
        """Gets a stream for this MocapSource.

        Since all the frames from this source will be immediately written to the stream's buffer,
        the buffer is created with max_length=len(self)+1 so it can hold all the data.
        """
        return MocapStream(self, max_buffer_len=len(self)+1)


class MocapStream(object):
    def __init__(self, mocap_source, max_buffer_len=0):
        """The default stream/buffer object to use for reading data from a MocapSource.

        A MocapStream instance buffers data received from a MocapSource until it is read. Data can
        be read using the read() or read_dict() methods, or the MocapStream instance itself can be
        iterated over in a loop. 

        Args:
        mocap_source: the MocapSource instance to attach to (the new MocapStream instance will
            be automatically registered with this source)
        max_buffer_len: int - the maximum length of the new frame buffer. If new frames arrive and
            the buffer is full, the oldest frames will be discarded to accomodate the new data. 
        """
        self._source = mocap_source
        self._buffer = Queue.Queue(maxsize=max_buffer_len)
        self._is_eof = False

        # Counters for get_framerate()
        self._frame_count = 0
        self._start_time = None

        # Data about the coordinate transformation to apply
        self._coordinates_mode = None
        self._desired_coords = None
        self._desired_idxs = None
        self._last_transform = np.identity(4)

        # Register this stream (do this last so it's ready to receive frames from an offline source)
        self._source.register_buffer(self)

    def put(self, frame):
        """Adds new a new frame to the stream's buffer. MocapSources with which this stream is 
        registered will call this method whenever a new frame arrives.

        If the buffer already contains max_buffer_len frames, the oldest frame will be discarded and
        replaced. Be sure to set max_buffer_len appropriately, particularly for offline sources 
        which put all their frames into the buffer immediately after it's registered with them.

        Args:
        frame: tuple containing:
            (num_points, 3, 1) ndarray - the frame to add to the queue
            (1,) ndarray - the timestamp of the frame
        """
        if frame is not None:
            if frame[0].ndim != 3 or frame[0].shape[1] != 3 or frame[0].shape[2] != 1:
                raise ValueError("Frame array must be (num_points, 3, 1)")
            if frame[1].ndim != 1 or frame[1].shape[0] != 1:
                raise ValueError("Timestamp array must be (1,)")

        # Add the frame to the buffer, removing and old frame first if the buffer is full
        try:
            self._buffer.put_nowait(frame)
        except Queue.Full:
            self._buffer.get_nowait()
            self._buffer.put_nowait(frame)

        # Update counters framerate counters
        self._frame_count += 1
        if self._start_time is None:
            self._start_time = time.time()

    def _get_frames(self, length, block):
        """Gets the specified number of frames and timestamps from the buffer and returns them as a
        tuple of ndarrays. Raises an EOFError when the end of the stream is reached.
        """
        if self._is_eof:
            raise EOFError()
        frames = []
        timestamps = []
        for i in range(length):
            try:
                next_sample = self._buffer.get(block=block)
                if next_sample is None:
                    self._is_eof = True
                    if i is 0:
                        raise EOFError()
                    else:
                        break
            except Queue.Empty:
                break
            frames.append(next_sample[0])
            timestamps.append(next_sample[1])
        return np.dstack(frames), np.hstack(timestamps)

    def read(self, length=1, block=True):
        """Reads the specified number of frames from the MocapSource. Raises an EOFError
        when the end of the stream is reached.
        """
        frames, timestamps = self._get_frames(length, block)
        frames, timestamps = self._process_frames(frames, timestamps)
        return frames, timestamps

    def _process_frames(self, frames, timestamps):
        """Apply preprocessing steps to frames and timestamps before returning them.
        """
        # Compute and apply a coordinate transformation, if any
        if self._coordinates_mode is not None:
            if self._coordinates_mode == 'time_varying':
                # Iterate over each frame
                trans_points = frames.copy()
                for i in range(trans_points.shape[2]):
                    # Find which of the specified markers are visible in this frame
                    # visible_inds = np.where(~np.isnan(trans_points[self._desired_idxs[coordinate_frame],0,i]))[0]
                    visible_inds = ~np.isnan(trans_points[self._desired_idxs, :, i]).any(axis=1)

                    # Compute the transformation
                    orig_points = trans_points[self._desired_idxs[visible_inds], :, i]
                    desired_points = self._desired_coords[visible_inds]
                    try:
                        homog = find_homog_trans(orig_points, desired_points, rot_0=None)[0]
                        self._last_transform = homog
                    except ValueError:
                        # Not enough points visible for tf.transformations to compute the transform
                        homog = self._last_transform

                    #Apply the transformation to the frame
                    homog_coords = np.vstack((trans_points[:, :, i].T, np.ones((1, trans_points.shape[0]))))
                    homog_coords = np.dot(homog, homog_coords)
                    trans_points[:, :, i] = homog_coords.T[:, 0:3]
            else:
                raise TypeError('The specified coordinate transform mode is invalid')

            #Save the transformed points
            frames = trans_points

        # Return the original frames and timestamps with any transformations applied
        return frames, timestamps

    def read_dict(self, name_dict, block=True):
        """Returns a dict mapping marker names to numpy arrays of marker coordinates.
        Raises an EOFError when the end of the stream is reached.

        Args:
        name_dict: dict - maps marker names to the corresponding marker indices
        """
        data = self.read(length=1, block=block)
        data_array = data[0]
        timestamp = data[1][0]
        if data_array is None:
            return None
        output_dict = {}
        for marker_name in name_dict:
            data_point = data_array[name_dict[marker_name],:,0]
            output_dict[marker_name] = data_point
        return output_dict, timestamp

    def get_num_points(self):
        return self._source.get_num_points()

    def get_length(self):
        return len(self._source)

    def get_framerate(self):
        """Get this stream's framerate.

        If the underlying MocapSource specifies a framerate, just returns that. Otherwise, computes
        the framerate directly given the frames seen so far.
        """
        if self._source.get_framerate() is not None:
            return self._source.get_framerate()
        else:
            return self._frame_count / (time.time() - self._start_time)

    def set_coordinates(self, markers, new_coords, mode='time-varying'):
        """
        Designates a new coordinate frame defined by a subset of markers and their desired positions
        in the new frame. 

        Args:
        markers: list - the markers which define this frame
        new_coords: (len(markers), 3) ndarray - the desired coordinates of the designated markers in
            the new frame
        mode: string - transform mode (currently only 'time-varying' is supported)
        """
        self._coordinates_mode = mode
        self._desired_coords = new_coords.squeeze()
        self._desired_idxs = np.array(markers).squeeze()
        self._last_transform = np.identity(4)

    def get_last_coordinates(self):
        return self._last_transform

    def close(self):
        self._source.unregister_buffer(self)

    def __iter__(self):
        return MocapStreamIterator(self)

    def __len__(self):
        return self.get_length()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class PhasespaceMocapSource(OnlineMocapSource):
    def __init__(self, ip_address, num_points, framerate=None):
        super(PhasespaceMocapSource, self).__init__()
        self._num_points = num_points
        self._shutdown_flag = False

        # Run the read loop at 2000Hz regardless of actual framerate to control
        # jitter
        self._timer = RateTimer(2000)

        # Connect to the server
        OWL.owlInit(ip_address, 0)
        OWL.owlSetInteger(OWL.OWL_FRAME_BUFFER_SIZE, 0)

        tracker = 0
        OWL.owlTrackeri(tracker, OWL.OWL_CREATE, OWL.OWL_POINT_TRACKER)

        # add the points we want to track to the tracker
        for i in range(num_points):
            OWL.owlMarkeri(OWL.MARKER(tracker, i), OWL.OWL_SET_LED, i)

        # Activate tracker
        OWL.owlTracker(0, OWL.OWL_ENABLE)

        # Set frequency
        if framerate is None:
            OWL.owlSetFloat(OWL.OWL_FREQUENCY, OWL.OWL_MAX_FREQUENCY)
            # self._timer = RateTimer(1500)
        else:
            OWL.owlSetFloat(OWL.OWL_FREQUENCY, framerate)
            # self._timer = RateTimer(framerate*3)

        # Start streaming
        OWL.owlSetInteger(OWL.OWL_STREAMING, OWL.OWL_ENABLE)

        # Check for errors
        if OWL.owlGetError() != OWL.OWL_NO_ERROR:
            raise RuntimeError('An error occurred while connecting to the mocap server')

        # Start the reader thread
        self._reader = Thread(target=self._reader_thread)
        self._reader.daemon = True
        self._start_time = time.time()
        self._reader.start()

    def _reader_thread(self):
        while(1):
            self._timer.wait()
            markers = OWL.owlGetMarkers()
            if markers.size() > 0:
                #If there's data, add a frame to the buffer
                new_frame = np.empty((self._num_points, 3, 1))
                new_frame.fill(np.nan)

                #Add the markers
                for i in range(markers.size()):
                    m = markers[i]
                    if m.cond > 0:
                        new_frame[m.id,0,0] = m.x
                        new_frame[m.id,1,0] = m.y
                        new_frame[m.id,2,0] = m.z
                        # print("%d: %f %f %f" % (m.id, m.x, m.y, m.z))
                timestamp = np.zeros((1,))
                timestamp[0] = time.time()
                self._output_frame(new_frame, timestamp)


            if OWL.owlGetError() != OWL.OWL_NO_ERROR:
                print('A mocap read error occurred')
            if self._shutdown_flag:
                return

    def get_num_points(self):
        return self._num_points


class FileMocapSource(OfflineMocapSource):
    def __init__(self, input_data):
        """Loads a motion capture file (currently a C3D file) to create 
        a new MocapFile object
        """
        import btk
        all_frames = None
        timestamps = None
        
        #Initialize the file reader
        data = btk.btkAcquisitionFileReader()
        data.SetFilename(input_data)
        data.Update()
        data = data.GetOutput()
        
        #Get the number of markers tracked in the file
        num_points = 0
        run = True
        while run:
            try:
                data.GetPoint(num_points)
                num_points = num_points + 1
            except(RuntimeError):
                run = False
        
        #Get the length of the file
        length = data.GetPointFrameNumber()
        
        #Load the file data into an array
        all_frames = sp.empty((num_points, 3, length))
        for i in range(num_points):
            all_frames[i,:,:] = data.GetPoint(i).GetValues().T
        
        #Replace occluded markers (all zeros) with NaNs
        norms = la.norm(all_frames, axis=1)
        ones = sp.ones(norms.shape)
        nans = ones * sp.nan
        occluded_mask = np.where(norms != 0.0, ones, nans)
        occluded_mask = np.expand_dims(occluded_mask, axis=1)
        all_frames = all_frames * occluded_mask
        
        #Calculate and save the timestamps
        frequency = data.GetPointFrequency()
        period = 1/frequency
        timestamps = sp.array(range(length), dtype='float') * period
        
        # Initialize this OfflineMocapSource with the resulting frames and timestamps arrays
        super(FileMocapSource, self).__init__(frames, timestamps)

        
    # def set_sampling(self, num_samples, mode='uniform'):
    #     if mode == 'uniform':
    #         indices = sp.linspace(0, self.get_length()-1, num=num_samples)
    #         indices = sp.around(indices).astype(int)
    #     elif mode == 'random':
    #         indices = sp.random.randint(0, self.get_length(), num_samples)
    #         indices = sp.sort(indices)
    #     else:
    #         raise TypeError('A valid mode was not specified')
            
    #     self._all_frames = self._all_frames[:,:,indices]
    #     self._timestamps = self._timestamps[indices]
    #     self._all_frames.flags.writeable = False
    #     self._timestamps.flags.writeable = False
        

class ArrayMocapSource(OfflineMocapSource):
    def __init__(self, array, framerate):
        timestamps = np.array(range(array.shape[2])) * (1.0/framerate)
        super(ArrayMocapSource, self).__init__(array, timestamps)


class PointCloudMocapSource(OnlineMocapSource):
    def __init__(self, topic_name):
        super(PointCloudMocapSource, self).__init__()
        # ROS dependencies only needed here
        import rospy
        import sensor_msgs.msg as sensor_msgs

        self._frame_name = None

        # Start the subscriber thread
        self._sub = rospy.Subscriber(topic_name, sensor_msgs.PointCloud,
                self._new_frame_callback)
        self._start_time = time.time()

    def _new_frame_callback(self, message):
        new_frame = point_cloud_to_array(message)
        timestamp = np.zeros((1,))
        timestamp[0] = time.time()
        self._output_frame(new_frame, timestamp)
        self._num_points = new_frame.shape[0]
        self._frame_name = message.header.frame_id

    def close(self):
        self._sub.unregister()

    def get_num_points(self):
        return self._num_points

    def get_frame_name(self):
        return self._frame_name


class MocapStreamIterator():
    def __init__(self, mocap_stream):
        # Check that mocap_stream is a MocapStream instance
        if not hasattr(mocap_stream, 'read'):
            raise TypeError('A valid MocapStream instance was not given')

        # Define fields
        self.mocap_stream = mocap_stream

    def next(self):
        try:
            value = self.mocap_stream.read(block=True)
            return value
        except EOFError:
            raise StopIteration()


class RateTimer():
    def __init__(self, frequency):
        self._loop_time = 1.0 / frequency
        self._next_time = None

    def wait(self):
        if self._next_time is None:
            self._next_time = time.time() + self._loop_time
        else:
            wait_time = self._next_time - time.time()
            self._next_time += self._loop_time
            if wait_time > 0:
                # print('Waiting: ' + str(wait_time))
                time.sleep(wait_time)


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
        homog = transformations.superimposition_matrix(points_a.T, points_b.T)
        return homog, None


def transform_frame(frame, transform):
    """
    Transforms an entire frame of points with a homogenous tansform
    :param frame: the mocap frame (n,3)
    :param transform: the transform (4,4)
    :return: mocap frame (n,3)
    """

    frame = frame.copy()

    # Apply the transformation to the frame
    homog_coords = np.vstack((frame[:, :].T, np.ones((1, frame.shape[0]))))
    homog_coords = np.dot(transform, homog_coords)
    frame = homog_coords.T[:, 0:3]
    return frame


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
