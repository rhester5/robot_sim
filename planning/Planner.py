
import numpy as np

class Planner(object):
	"""docstring for Planner"""
	def __init__(self, env):
		self.env = env
		self.x = env.x
		self.y = env.y

	def plan(self, start, goal):
		raise NotImplementedError

if __name__ == '__main__':
	main()