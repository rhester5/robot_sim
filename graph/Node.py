
import numpy as np

class Node(object):
	"""docstring for Node"""
	def __init__(self, state, parent, cost):
		self.state = state
		self.parent = parent
		self.cost = cost

	def get_state(self):
		return self.state

	def get_parent(self):
		return self.parent

	def get_cost(self):
		return self.cost

	def set_state(self, state):
		self.state = state

	def set_parent(self, parent):
		self.parent = parent

	def set_cost(self, cost):
		self.cost = cost
		

if __name__ == '__main__':
	main()