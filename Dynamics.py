
import numpy as np

class Dynamics:
	"""docstring for Dynamics"""
	def __init__(self, dt):
		self.dt = dt

	def create_motion_model_params(motion_model_cov):
		raise NotImplementedError

	def create_meas_model_params(meas_cov):
		raise NotImplementedError

if __name__ == '__main__':
	main()