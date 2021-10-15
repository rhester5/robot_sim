import numpy as np

class MeasurementModel():
	def __init__(self, h, R, linear):
		self.h = h
		self.R = R
		self.linear = linear

		(n, _) = R.shape
		self.n = n
		self.zero_mean = np.zeros(n)

	def __call__(self, x):
		if self.linear:
			measurement = np.matmul(self.h, x) + np.random.multivariate_normal(self.zero_mean, self.R)
		else:
			measurement = self.h(x) + np.random.multivariate_normal(self.zero_mean, self.R)
		return measurement

	def get_meas_size(self):
		return self.n