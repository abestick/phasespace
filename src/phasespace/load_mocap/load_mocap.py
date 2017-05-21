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
from Queue import Queue
from threading import Thread, RLock
import time
from copy import deepcopy

# OWL is only needed by PhasespaceStream
try:
    import OWL
except(ImportError):
    pass


class MocapSource():
    __metaclass__=ABCMeta

    def __init__(self):
        # Initialize vars for coordinate change, if any
        self._coordinates_mode = {}
        self._desired_coords = {}
        self._desired_idxs = {}
        self._last_transform = {}

    @abstractmethod
    def read(self, frames, timestamps, coordinate_frame=None):
        """Reads data from the underlying mocap source. By default, this method 
        will block until the data is read. Returns a tuple (frames, timestamps) 
        where frames is a (num_points, 3, length) ndarray of mocap points, and 
        timestamps is a (length,) ndarray of the timestamps, in seconds, of the 
        corresponding mocap points.

        Once the end of the file/stream is reached, calls to read() will return 
        None. If the end of the stream is reached before length frames are read,
        the returned arrays may have fewer elements than expected, and all 
        future calls to read() will return None.

        If called with block=False and no data is available, returns arrays with 
        length=0.

        Once the end of the file/stream is reached, calls to read() will return 
        None.
        """
        # Compute and apply a coordinate transformation, if any
        if coordinate_frame is not None:
            trans_points = None
            if self._coordinates_mode[coordinate_frame] == 'time_varying':
                # Iterate over each frame
                trans_points = frames.copy()
                for i in range(trans_points.shape[2]):
                    # Find which of the specified markers are visible in this frame
                    visible_inds = np.where(~np.isnan(trans_points[self._desired_idxs[coordinate_frame],0,i]))[0]

                    # Compute the transformation
                    orig_points = trans_points[self._desired_idxs[coordinate_frame][visible_inds], :, i]
                    desired_points = self._desired_coords[coordinate_frame][visible_inds]
                    try:
                        homog = find_homog_trans(orig_points, desired_points, rot_0=None)[0]
                        self._last_transform[coordinate_frame] = homog
                    except ValueError:
                        # Not enough points visible for tf.transformations to compute the transform
                        homog = self._last_transform

                    #Apply the transformation to the frame
                    homog_coords = np.vstack((trans_points[:, :, i].T, np.ones((1, trans_points.shape[0]))))
                    homog_coords = np.dot(homog, homog_coords)
                    trans_points[:, :, i] = homog_coords.T[:, 0:3]
            else:
                raise TypeError('The specified mode is invalid')

            #Save the transformed points
            frames = trans_points

        # Return the original frames and timestamps with any transformations applied
        return frames, timestamps

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_num_points(self):
        pass

    @abstractmethod
    def get_length(self):
        pass

    @abstractmethod
    def get_framerate(self):
        pass

    @abstractmethod
    def set_sampling(self, num_samples, mode='uniform'):
        pass

    @abstractmethod
    def register_buffer(self, buffer, **kwargs):
        """
        Takes in the reference to a queue-like object and populates it with mocap data processed by the read function 
        with args and kwargs passed.
        :param buffer: queue like object
        :param args: arguments for the read function
        :param kwargs: keyword arguments for the read function
        :return: 
        """
        pass

    def set_coordinates(self, coordinate_frame_name, markers, new_coords, mode='time-varying'):
        """
        Designates a new coordinate frame defined by a subset of markers and their desired positions in the new frame.
        This frame can later be accessed by the read function by in order to determine which frame the points are being
        returned in
        :param string coordinate_frame_name: the name given to this new coordinate frame
        :param markers: the markers which define this frame
        :param new_coords: the desired coordinates of the designated markers when in this frame
        :param mode: whether this transform is done online or statically (I think?)
        :return: 
        """
        self._coordinates_mode[coordinate_frame_name] = mode
        self._desired_coords[coordinate_frame_name] = new_coords.squeeze()
        self._desired_idxs[coordinate_frame_name] = np.array(markers).squeeze()
        self._last_transform[coordinate_frame_name] = np.identity(4)

    def get_last_coordinates(self):
        return self._last_transform

    def read_dict(self, name_dict, block=True):
        """Returns a dict mapping marker names to numpy arrays of 
        marker coordinates. Argument name_dict is a dict mapping marker
        names to marker indices in this particular dataset
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

    def clone(self):
        """
        Provides a deep copy of itself.
        """
        return deepcopy(self)

    def __len__(self):
        return self.get_length()

    def __iter__(self):
        return MocapIterator(self)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def iterate(self, buffer_size, **kwargs):
        """
        Provides an iterator that will iterate through this source applying the args and kwargs passed to the read
        function at each frame. The iterator will have it's own buffer so that reading does not affect other iterators
        operating on the same source.
        """
        return MocapIterator(self, buffer_size, **kwargs)

    @abstractmethod
    def get_latest_frame(self):
        pass


class PhasespaceStream(MocapSource):
    def __init__(self, ip_address, num_points, framerate=None, buffer_length=2):
        super(PhasespaceStream, self).__init__()
        self._num_points = num_points
        self._shutdown_flag = False
        self._start_time = 0
        self._frame_count = 0

        # list for references to external buffers that may get populated with new data and their args and kwargs
        self._buffers = []

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

        # Initialize a circular read buffer
        self._read_buffer = _RingBuffer(buffer_length)

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
                timestamp = np.array(time.time())
                self._read_buffer.put((new_frame, timestamp))

                # populate each of the registered buffers with the new data
                for b in range(len(self._buffers)):
                    # extract the buffer reference, and the corresponding args and kwargs for the read call
                    buffer, args, kwargs = self._buffers[b]

                    '''
                    Here I call the super function directly. An alternative which would give the same would be
                    self.read(length=1, block=Irrelevant, **kwargs)
                    I wasn't sure which made more sense but putting the frames into the internal buffer just to pop them
                    out again seemed silly.
                    
                    We may choose to ditch the buffer, now that the external buffer provided by the iterator handles
                    the blocking. Or we may want to keep it so we can always get out the last N frames without iterating
                    or manually registering a buffer. 
                    
                    If we keep it, I suggest we change some naming. I think we should rename read in MocapSource to
                    _process_frames, since it doesn't actually read any frames internally when it's called, but rather
                    its children overloads do then call it to process the read data. And I think private because we 
                    never expect someone to call that function directly, we always expect the children to find the 
                    frames and pass them internally. We'd therefore make an abstract read method in MocapSource
                    
                    And if we put all the per-frame processing into a _process_frame function so _process_frames looks
                    like:
                    for frame in frames:
                        processed.append(self._process_frame(frame, options))
                        
                    then every child will be able to call this function for single frames when updating buffers and
                    the multi-frame functions can be left for reading chunks of data.
                    
                    Then the read function will take selection options (which frames) and processing options (eventually
                    passed to process_frame). And the iterators will be registed only with processing options, since 
                    the buffers will be updated with direct calls to _process_frames.
                    
                    If we keep the internal ring buffer we could also give it a get_last_n_frames function which peeks
                    at the latest frames, this way we can make it much larger than 2 and still get recent information.
                    '''
                    buffer.put(super(PhasespaceStream, self).read(new_frame, timestamp, **kwargs))


                self._frame_count += 1

            if OWL.owlGetError() != OWL.OWL_NO_ERROR:
                print('A mocap read error occurred')
            if self._shutdown_flag:
                return

    def read(self, length=1, block=True, coordinate_frame=None):
        """Reads data from the underlying mocap source. By default, this method 
        will block until the data is read. Returns a tuple (frames, timestamps) 
        where frames is a (num_points, 3, length) ndarray of mocap points, and 
        timestamps is a (length,) ndarray of the timestamps, in seconds, of the 
        corresponding mocap points.

        Once the end of the file/stream is reached, calls to read() will return 
        None. If the end of the stream is reached before length frames are read,
        the returned arrays may have fewer elements than expected, and all 
        future calls to read() will return None.

        If called with block=False and no data is available, returns arrays with 
        length=0.

        Once the end of the file/stream is reached, calls to read() will return 
        None.
        """
        frames = []
        timestamps = []
        for i in range(length):
            next_sample = self._read_buffer.get(block=block)
            frames.append(next_sample[0])
            timestamps.append(next_sample[1])
        frames, timestamps = np.dstack(frames), np.hstack(timestamps)

        # Let MocapSource.read() perform any remaining processing on the data before returning
        return super(PhasespaceStream, self).read(frames, timestamps, coordinate_frame)

    def close(self):
        self._shutdown_flag = True
        self._reader.join()
        OWL.owlDone()

    def get_num_points(self):
        return self._num_points

    def get_length(self):
        return 0

    def get_framerate(self):
        return self._frame_count / (time.time() - self._start_time)

    def set_sampling(self, num_samples, mode='uniform'):
        pass

    def register_buffer(self, buffer, **kwargs):
        """
        Registers a new buffer which will be updated with the latest frames
        :param buffer: a queue-like buffer which will receive all new frame data
        :return: 
        """

        self._buffers.append((buffer, kwargs))

    # def set_coordinates(self, coordinate_frame_name, markers, new_coords, mode='constant'):
        
    def iterate(self, buffer_size=2, **kwargs):
        return super(PhasespaceStream, self).iterate(buffer_size, **kwargs)

    def get_latest_frame(self, **kwargs):
        new_frame, timestamp = self._read_buffer.peek()
        return super(PhasespaceStream, self).read(new_frame, timestamp, **kwargs)


class MocapFile(MocapSource):
    def __init__(self, input_data):
        """Loads a motion capture file (currently a C3D file) to create 
        a new MocapFile object
        """
        import btk
        super(MocapFile, self).__init__()

        #Declare fields
        self._all_frames = None
        self._timestamps = None
        self._read_pointer = 0 #Next element that will be returned by read()
        
        #Check whether data is another MocapFile instance
        if hasattr(input_data, '_all_frames') and hasattr(input_data, '_timestamps'):
            self._all_frames = input_data._all_frames
            self._timestamps = input_data._timestamps
            
        #If not, treat data as a filepath to a C3D file
        else:
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
            self._all_frames = sp.empty((num_points, 3, length))
            for i in range(num_points):
                self._all_frames[i,:,:] = data.GetPoint(i).GetValues().T
            
            #Replace occluded markers (all zeros) with NaNs
            norms = la.norm(self._all_frames, axis=1)
            ones = sp.ones(norms.shape)
            nans = ones * sp.nan
            occluded_mask = np.where(norms != 0.0, ones, nans)
            occluded_mask = np.expand_dims(occluded_mask, axis=1)
            self._all_frames = self._all_frames * occluded_mask
            
            #Calculate and save the timestamps
            frequency = data.GetPointFrequency()
            period = 1/frequency
            self._timestamps = sp.array(range(length), dtype='float') * period
            
            #Make the arrays read-only
            self._all_frames.flags.writeable = False
            self._timestamps.flags.writeable = False
    
    def read(self, length=1, block=True, coordinate_frame=None):
        # Make sure we don't try to read past the end of the file
        file_len = self.get_length()
        if file_len == self._read_pointer:
            return None
        elif length > file_len - self._read_pointer:
            length = file_len - self._read_pointer

        # Read the frames and timestamps
        frames = self._get_frames()[:,:,self._read_pointer:self._read_pointer+length]
        timestamps = self.get_timestamps()[self._read_pointer:self._read_pointer+length]

        # Increment the read pointer
        self._read_pointer = self._read_pointer + length

        # Let MocapSource.read() perform any remaining processing on the data before returning
        return super(MocapFile, self).read(frames, timestamps, coordinate_frame)

    def close(self):
        pass
    
    # TODO: Should this method exist at all/use read() so coordinate changes are applied properly? 
    def _get_frames(self):
        """Returns a (num_points, 3, length) array of the mocap points.
        Always access mocap data through this method.
        """
        return self._all_frames
    
    def get_timestamps(self):
        """Returns a 1-D array of the timestamp (in seconds) for each frame
        """
        return self._timestamps
    
    def get_num_points(self):
        """Returns the total number of points tracked in the mocap file
        """
        return self._get_frames().shape[0]
    
    def get_length(self):
        """Returns the total number of frames in the mocap file
        """
        return self._get_frames().shape[2]
    
    def get_framerate(self):
        """Returns the average framerate of the mocap file in Hz"""
        duration = self._timestamps[-1] - self._timestamps[0]
        framerate = (self.get_length() - 1) / duration
        return framerate
    
    def set_start_end(self, start, end):
        """Trims the mocap sequence to remove data before/after the
        specified start/end frames, respectively.
        """
        self._all_frames = self._all_frames[:,:,start:end+1]
        self._timestamps = self._timestamps[start:end+1]
        self._all_frames.flags.writeable = False
        self._timestamps.flags.writeable = False
        
    def set_sampling(self, num_samples, mode='uniform'):
        if mode == 'uniform':
            indices = sp.linspace(0, self.get_length()-1, num=num_samples)
            indices = sp.around(indices).astype(int)
        elif mode == 'random':
            indices = sp.random.randint(0, self.get_length(), num_samples)
            indices = sp.sort(indices)
        else:
            raise TypeError('A valid mode was not specified')
            
        self._all_frames = self._all_frames[:,:,indices]
        self._timestamps = self._timestamps[indices]
        self._all_frames.flags.writeable = False
        self._timestamps.flags.writeable = False
    
    def plot_frame(self, frame_num, mark_num):
        """Plots the location of each marker in the specified frame
        """
        #Get the frame
        frame = self._get_frames()[:,:,frame_num]
        xs = frame[:,0]
        ys = frame[:,1]
        zs = frame[:,2]
        
        #Make the plot
        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')
        markers = ['r']*self.get_num_points()
        markers[mark_num] = 'b'
        axes.scatter(xs, ys, zs, c=markers, marker='o')
        axes.auto_scale_xyz([-1000,1000], [-1000, 1000], [0, 2000])
        axes.set_xlabel('X Label')
        axes.set_ylabel('Y Label')
        axes.set_zlabel('Z Label')
        axes.set_zlabel('Z Label')

    def register_buffer(self, buffer, **kwargs):
        """
        Registers a new buffer which will be loaded with all frames
        :param buffer: a queue-like buffer which will receive all frame data
        :return: 
        """
        for i in range(self.get_length()):
            frames = self._get_frames()[:, :, i:i+1]
            timestamps = self.get_timestamps()[i:i+1]
            buffer.put(super(MocapFile, self).read(frames, timestamps, **kwargs))

        buffer.put(None)

    def iterate(self, **kwargs):
        # We never want to limit the buffer
        return super(MocapFile, self).iterate(0, **kwargs)

    def get_latest_frame(self, **kwargs):
        return self.read(**kwargs)


class MocapArray(MocapFile):
    def __init__(self, array, framerate):
        super(MocapFile, self).__init__()
        if array.shape[1] != 3 or array.ndim != 3:
            raise TypeError('Input array is not the correct shape')

        self._all_frames = array
        self._timestamps = np.array(range(array.shape[2])) * (1.0/framerate)
        self._read_pointer = 0 #Next element that will be returned by read()


class PointCloudStream(MocapSource):
    def __init__(self, topic_name, buffer_length=2):
        super(PointCloudStream, self).__init__()
        # ROS dependencies only needed here
        import rospy
        import sensor_msgs.msg as sensor_msgs

        # Initialize counters
        self._num_points = -1
        self._start_time = 0
        self._frame_count = 0
        self._frame_name = None

        # Initialize a circular read buffer
        self._read_buffer = _RingBuffer(buffer_length)

        # list for references to external buffers that may get populated with new data and their args and kwargs
        self._buffers = []

        # Start the subscriber thread
        self._sub = rospy.Subscriber(topic_name, sensor_msgs.PointCloud,
                self._new_frame_callback)
        self._start_time = time.time()

    def _new_frame_callback(self, message):
        new_frame = point_cloud_to_array(message)
        timestamp = np.array(time.time())
        self._read_buffer.put((new_frame, timestamp))

        # populate each of the registered buffers with the new data
        for b in range(len(self._buffers)):
            # extract the buffer reference, and the corresponding args and kwargs for the read call
            buffer, args, kwargs = self._buffers[b]

            buffer.put(super(PointCloudStream, self).read(new_frame, timestamp, **kwargs))

        self._frame_count += 1
        self._num_points = new_frame.shape[0]
        self._frame_name = message.header.frame_id

    def read(self, length=1, block=True, coordinate_frame=None):
        """Reads data from the underlying mocap source. By default, this method 
        will block until the data is read. Returns a tuple (frames, timestamps) 
        where frames is a (num_points, 3, length) ndarray of mocap points, and 
        timestamps is a (length,) ndarray of the timestamps, in seconds, of the 
        corresponding mocap points.

        Once the end of the file/stream is reached, calls to read() will return 
        None. If the end of the stream is reached before length frames are read,
        the returned arrays may have fewer elements than expected, and all 
        future calls to read() will return None.

        If called with block=False and no data is available, returns arrays with 
        length=0.

        Once the end of the file/stream is reached, calls to read() will return 
        None.
        """
        frames = []
        timestamps = []
        for i in range(length):
            next_sample = self._read_buffer.get(block=True)
            frames.append(next_sample[0])
            timestamps.append(next_sample[1])
        frames = np.dstack(frames)
        timestamps = np.hstack(timestamps)

        # Let MocapSource.read() perform any remaining processing on the data before returning
        return super(PointCloudStream, self).read(frames, timestamps, coordinate_frame)

    def close(self):
        self._sub.unregister()

    def get_num_points(self):
        return self._num_points

    def get_length(self):
        return 0

    def get_framerate(self):
        return self._frame_count / (time.time() - self._start_time)

    def set_sampling(self, num_samples, mode='uniform'):
        pass

    def get_frame_name(self):
        return self._frame_name

    def register_buffer(self, buffer, **kwargs):
        """
        Registers a new buffer which will be updated with the latest frames
        :param buffer: a queue-like buffer which will receive all new frame data
        :return: 
        """

        self._buffers.append((buffer, kwargs))

    # def set_coordinates(self, coordinate_frame_name, markers, new_coords, mode='constant'):

    def iterate(self, buffer_size=2, **kwargs):
        return super(PointCloudStream, self).iterate(buffer_size, **kwargs)

    def get_latest_frame(self, **kwargs):
        new_frame, timestamp = self._read_buffer.peek()
        return super(PointCloudStream, self).read(new_frame, timestamp, **kwargs)


class MocapIterator():
    def __init__(self, mocap_obj, buffer_size=0, **kwargs):
        # Check that mocap_obj is a MocapFile instance
        if not hasattr(mocap_obj, 'read'):
            raise TypeError('A valid MocapSource instance was not given')

        # Define fields
        self.mocap_obj = mocap_obj
        self.buffer = _RingQueue(buffer_size)
        self.mocap_obj.register_buffer(self.buffer, **kwargs)

    def __iter__(self):
        return self

    def next(self):
        value = self.buffer.get(block=True)
        if value is not None:
            return value
        else:
            raise StopIteration()


class _RingQueue():
    def __init__(self, size):
        self._buffer = Queue(maxsize=size)

    def put(self, item):
        if self._buffer.full():
            #If the buffer is full, discard the oldest element to make room
            self._buffer.get()
        self._buffer.put(item)

    def get(self, block=True):
        return self._buffer.get(block=block)


class _RingBuffer(deque):
    def __init__(self, size):
        super(_RingBuffer, self).__init__(maxlen=size)

    def put(self, item):
        self.append(item)

    def get_oldest(self):
        return self.popleft()

    def get_latest(self):
        return self.pop()

    def get_oldest_n(self, n):
        return [self.popleft() for i in range(n)]

    def peek(self, i=-1):
        return self[i]

    def peek_latest_n(self, n):
        return self[-n:]

    def peek_oldest_n(self, n):
        return self[:n]

    def get(self, block=True):
        if block:
            while len(self) == 0:
                time.sleep(0.01)

        return self.peek(0)



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
        import tf.transformations as convert
        homog = convert.superimposition_matrix(points_a.T, points_b.T)
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
