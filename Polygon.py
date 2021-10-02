
import numpy as np

class Polygon:
	"""docstring for Polygon"""
	def __init__(self, v):
		self.v = v
		if v.shape[0] > 1 and len(v.shape) > 1:
			self.e = [[v[i, :], v[i+1, :]] for i in range(v.shape[0]-1)] + [[v[-1, :], v[0, :]]]

	def __str__(self):
		return str(self.v) + '\n\n' + str(self.e)

	def get_bounds(self):
		x1, y1 = np.min(self.v, axis=0)
		x2, y2 = np.max(self.v, axis=0)
		return x1, y1, x2, y2

	def point_intersection(self, point):
		# http://geomalgorithms.com/a03-_inclusion.html
		wn = 0
		for edge in self.e:
			if edge[0][1] <= point[1]:
				if edge[1][1] > point[1]:
					if self.is_left(edge[0], edge[1], point) > 0:
						wn += 1
			else:
				if edge[1][1] <= point[1]:
					if self.is_left(edge[0], edge[1], point) < 0:
						wn -= 1
		return wn != 0

	def is_left(self, v_start, v_end, point):
		# http://geomalgorithms.com/a01-_area.html
		return (v_end[0] - v_start[0]) * (point[1] - v_start[1]) - (point[0] - v_start[0]) * (v_end[1] - v_start[1])

if __name__ == '__main__':
	main()