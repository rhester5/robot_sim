
import numpy as np
from matplotlib import pyplot as plt

class Map(object):
	"""docstring for Map"""
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.occupancy_grid = np.zeros((x, y))

	def add_obstacle(self, obstacle):
		x1, y1, x2, y2  = obstacle.get_bounds()
		for i in range(x1, x2):
			for j in range(y1, y2):
				if obstacle.point_intersection((i, j)):
					self.occupancy_grid[i, j] = 1

	def render(self):
		plt.imshow(self.occupancy_grid, cmap='binary')
		

if __name__ == '__main__':
	main()