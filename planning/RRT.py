
import numpy as np

from planning.Planner import Planner
from graph.Graph import Graph
from graph.Node import Node

class RRT(Planner):
	"""docstring for RRT"""
	def __init__(self, env, num_iter, radius, sample_goal):
		super(RRT, self).__init__(env)
		self.num_iter = num_iter
		self.radius = radius
		self.sample_goal = sample_goal
		self.disc = 10
		
	def plan(self, start, goal):
		self.start = start.astype(float)
		self.goal = goal.astype(float)
		self.tree = Graph([Node(self.start, None, 0)], [])
		xs = []
		ys = []
		for i in range(self.num_iter):
			print(i)
			if i % self.sample_goal:
				q_rand = Node(self.goal, None, 0)
			else:
				q_rand = self.sample_environment()
			q_near = self.nearest(q_rand)
			q_rand.set_parent(q_near)
			q_new = self.new(q_near, q_rand)
			if self.free(q_near, q_new):
				self.tree.add_vertex(q_new)
				self.tree.add_edge((q_near, q_new))
				x, y = self.render()
				if (len(xs) == 0 and len(ys) == 0) or (x != xs[-1] and y != ys[-1]):
					xs.append(x)
					ys.append(y)
				else:
					print('same')
				# xs.append(x)
				# ys.append(y)
				if np.all(q_new.get_state() == self.goal):
					return self.backtrack(q_new), xs, ys
		return None

	def sample_environment(self):
		free = False
		while not free:
			x, y = np.random.randint((self.env.x ,self.env.y))
			if not self.env.occupancy_grid[x, y]:
				free = True
		return Node(np.array([x, y]), None, 0)

	def distance(self, q1, q2):
		return np.linalg.norm(q2-q1)

	def nearest(self, q_rand):
		least_dist = np.inf
		q_near = None
		for v in self.tree.get_vertices():
			dist = self.distance(v.get_state(), q_rand.get_state())
			if dist < least_dist:
				least_dist = dist
				q_near = v
		return q_near

	def new(self, q_near, q_rand):
		dist = self.distance(q_near.get_state(), q_rand.get_state())
		if dist < self.radius:
			q_rand.set_cost(dist)
		else:
			p1 = q_near.get_state()
			p2 = q_rand.get_state()
			p3 = p1 + self.radius * (p2 - p1) / dist
			q_rand.set_state(p3)
			q_rand.set_cost(self.radius)
		return q_rand

	def free(self, q_near, q_new):
		pts = np.linspace(q_near.get_state(), q_new.get_state(), self.disc, dtype=int)
		# print(pts, self.env.occupancy_grid[pts[:, 0], pts[:, 1]], not np.any(self.env.occupancy_grid[pts[:, 0], pts[:, 1]]))
		return not np.any(self.env.occupancy_grid[pts[:, 1], pts[:, 0]])

	def backtrack(self, goal):
		path = []
		node = goal
		while node.get_parent() != None:
			path.append(node.get_state())
			node = node.get_parent()
		path.append(node.get_state())
		return np.array(path[::-1])

	def render(self):
		lines = []
		xs = []
		ys = []
		for edge in self.tree.get_edges():
			x1, y1 = edge[0].get_state()
			x2, y2 = edge[1].get_state()
			# lines.append(plt.plot([x1, x2], [y1, y2]))
			xs.append([x1, x2])
			ys.append([y1, y2])
		# return lines
		return xs, ys

if __name__ == '__main__':
	main()