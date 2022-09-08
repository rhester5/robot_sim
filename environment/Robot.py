from environment.Polygon import Polygon

class Robot(Polygon):
	"""docstring for Robot"""
	def __init__(self, pose):
		super(Robot, self).__init__(pose)
		self.pose = pose

	def get_pose(self):
		return self.pose

	def set_pose(self, pose):
		self.pose = pose
		

if __name__ == '__main__':
	main()