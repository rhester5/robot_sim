
import numpy as np

class Controller(object):
	"""docstring for Controller"""
	def __init__(self, gains, actuator_limits, dt):
		self.set_gains(gains)
		self.set_actuator_limits(actuator_limits)
		self.dt = dt

	def __call__(self, x, sp):
		raise NotImplementedError

	def set_gains(self, gains):
		raise NotImplementedError

	def get_gains(self):
		raise NotImplementedError

	def set_actuator_limits(self, actuator_limits):
		raise NotImplementedError

	def get_actuator_limits(self):
		raise NotImplementedError


if __name__ == '__main__':
	main()