
import numpy as np

class Graph(object):
	"""docstring for Graph"""
	def __init__(self, V, E):
		self.V = V
		self.E = E

	def add_vertex(self, v):
		self.V.append(v)

	def add_edge(self, e):
		self.E.append(e)

	def get_vertices(self):
		return self.V

	def get_edges(self):
		return self.E
		
if __name__ == '__main__':
	main()