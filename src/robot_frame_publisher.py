import rospy
import tf
import tf.transformations
import numpy as np
import numpy.linalg as npla

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

def vectorize_loss_function(loss_function):
	"""Returns a vectorized loss function which takes a single vector argument.
	params (10,) - [g_BM rotation (x,y,z,w), g_BM translation (x,y,z), p_H (x,y,z)]
	"""
	def vectorized_loss(params):
		rotation = params[0:4] / npla.norm(params[0:4])
		g_BM = tf.transformations.quaternion_matrix(rotation)
		g_BM[0:3,3] = params[4:7]
		p_H = np.hstack((params[7:10], np.array([1])))
		return loss_function(g_BM, p_H)
	return vectorized_loss

def main():
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

	vec_loss = vectorize_loss_function(loss_func)
	vec_loss(np.array([1,2,3,4,5,6,7,8,9,10]))

	1/0

if __name__ == '__main__':
	main()