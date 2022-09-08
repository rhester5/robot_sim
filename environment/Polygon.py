
import numpy as np

class Polygon(object):
	"""docstring for Polygon"""
	def __init__(self, v):
		self.v = v
		if self.v.shape[0] > 1 and len(self.v.shape) > 1:
			if self.v.shape[0] > 2:
				self.v = self.graham_scan(self.v)
			self.e = [[self.v[i, :], self.v[i+1, :]] for i in range(self.v.shape[0]-1)] + [[self.v[-1, :], self.v[0, :]]]

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

	def less(self, p, p0):
		pp = p-p0
		return np.arctan2(pp[1], pp[0]), np.linalg.norm(pp)

	def graham_scan(self, pts):
		N, _ = pts.shape

		if N < 3:
			raise ValueError('At least 3 points are required to define a polygon')

		# get p0
		p0 = None
		x_low = np.inf
		y_low = np.inf
		for p in pts:
			if p[1] < y_low:
				y_low = p[1]
				x_low = p[0]
				p0 = p
			elif p[1] == y_low:
				if p[0] < x_low:
					x_low = p[0]
					p0 = p

		# sort
		pts_sorted = sorted(pts, key=lambda p: self.less(p, p0))

		if N == 3:
			# return Polygon(pts_sorted)
			return np.array(pts_sorted)

		# scan
		progress = np.zeros((N,))
		p0 = N-1
		p1 = 0
		p2 = 1
		p3 = 2
		p4 = 3
		convex_hull = [i for i in range(N)]
		while not np.all(progress):
			progress[p2] = 1
			if np.cross(pts_sorted[p1] - pts_sorted[p2], pts_sorted[p3] - pts_sorted[p2]) >= 0:
				convex_hull.remove(p2)
				# p4 = p4
				# p3 = p3
				p2 = p1
				p1 = p0
				p0 = convex_hull[convex_hull.index(p0) - 1]
			else:
				p0 = p1
				p1 = p2
				p2 = p3
				p3 = p4
				p4_ind = convex_hull.index(p4) + 1
				n = len(convex_hull)
				if p4_ind >= n:
					p4_ind -= n
				p4 = convex_hull[p4_ind]

		pts_sorted = np.array(pts_sorted)
		# polygon = Polygon(pts_sorted[convex_hull, :])
		# return polygon
		return pts_sorted[convex_hull, :]

if __name__ == '__main__':
	main()