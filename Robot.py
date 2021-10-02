
import numpy as np

class Robot:
	"""docstring for Robot"""
	def __init__(self, pose):
		self.pose = pose

	def get_pose(self):
		return self.pose

	def set_pose(self, pose):
		self.pose = pose
		

if __name__ == '__main__':
	main()