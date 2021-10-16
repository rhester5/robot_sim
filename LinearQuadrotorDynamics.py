
import numpy as np

from Dynamics import Dynamics

class LinearQuadrotorDynamics(Dynamics):
	"""docstring for LinearQuadrotorDynamics"""
	def __init__(self, dt, motion_model_cov, meas_model_cov):
		super(LinearQuadrotorDynamics, self).__init__(dt)
		self.motion_model_cov = motion_model_cov
		self.motion_model_params = self.create_motion_model_params(self.motion_model_cov)
		self.meas_model_cov = meas_model_cov
		self.meas_model_params = self.create_meas_model_params(self.meas_model_cov)

	def get_params(self):
		A, B, Q = self.motion_model_params
		H, R = self.meas_model_params
		return A, B, Q, H, R

	def get_motion_model_params(self):
		return self.motion_model_params

	def get_meas_model_params(self):
		return self.meas_model_params

	def create_motion_model_params(self, motion_model_cov):
		# xdot = Ax + Bu
		# where x = [x, y, xdot, ydot]
		# and u = [xddot, yddot]
		A = np.block([[np.eye(2), self.dt * np.eye(2)],
					  [np.zeros((2, 2)), np.eye(2)]])
		B = np.block([[self.dt**2/2 * np.eye(2)], [self.dt * np.eye(2)]])

		Qpos = motion_model_cov[0]
		Qvel = motion_model_cov[1]
		Q = np.diag([Qpos, Qpos, Qvel, Qvel])

		return A, B, Q

	def create_meas_model_params(self, meas_model_cov):
		H = np.eye(4)
		# H[0, 0] = 0
		# H[1, 1] = 0

		Rpos = meas_model_cov[0]
		Rvel = meas_model_cov[1]
		R = np.diag([Rpos, Rpos, Rvel, Rvel])

		return H, R

	
if __name__ == '__main__':
	main()