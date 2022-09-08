import numpy as np

class MotionModel(object):
	def __init__(self, args):
		if len(args) == 2:
			self.f = args[0]
			self.Q = args[1]
			self.linear = False
		elif len(args) == 3:
			self.A = args[0]
			self.B = args[1]
			self.Q = args[2]
			self.linear = True

		(m, _) = self.Q.shape
		self.m = m
		self.zero_mean = np.zeros(m)

	def __call__(self, x, u):
		if self.linear:
			new_state = np.matmul(self.A, x) + np.matmul(self.B, u) + np.random.multivariate_normal(self.zero_mean, self.Q)
		else:
			new_state = self.f(x, u) + np.random.multivariate_normal(self.zero_mean, self.Q)
		return new_state

	def get_state_size(self):
		return self.m