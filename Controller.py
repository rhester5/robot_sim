
import numpy as np

class Controller:
	"""docstring for Controller"""
	def __init__(self, gains, dt):
		self.set_gains(gains)
		self.dt = dt

	def __call__(self, x, sp):
		raise NotImplementedError

	def set_gains(self, gains):
		raise NotImplementedError

	def get_gains(self):
		raise NotImplementedError


if __name__ == '__main__':
	main()