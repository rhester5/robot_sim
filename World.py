
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path

from Map import Map
from Obstacle import Obstacle
from Robot import Robot

class World:
	"""docstring for World"""
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.map = Map(x, y)
		self.robots = []

	def add_obstacle(self, obstacle):
		self.map.add_obstacle(obstacle)

	def add_robot(self, robot):
		self.robots.append(robot)

	def set_start(self, start):
		self.start = start
		
	def set_goal(self, goal):
		self.goal = goal

	def set_trajectory(self, trajectory):
		self.trajectory = trajectory

	def render(self):
		plt.figure()
		self.map.render()
		plt.plot(self.trajectory[0, :], self.trajectory[1, :], 'b')
		plt.scatter(self.start[0], self.start[1], c='g')
		plt.scatter(self.goal[0], self.goal[1], c='r')
		for robot in self.robots:
			plt.scatter(robot.get_pose()[0], robot.get_pose()[1], c='b')
		plt.show()

	def animate(self, num_steps):
		fig, ax = plt.subplots()
		self.map.render()
		line = plt.plot(self.trajectory[:, 0], self.trajectory[:, 1], 'tab:blue')
		plt.scatter(self.start[0], self.start[1], c='g')
		plt.scatter(self.goal[0], self.goal[1], c='r')
		robots = []
		for robot in self.robots:
			robots.append(ax.scatter(robot.get_pose()[0], robot.get_pose()[1], c='b', marker='*'))

		def init():
			return robots # + line

		def update(i):
			for j, robot in enumerate(robots):
				robot.set_offsets(self.robot_trajectories[j][i].reshape((1, 2)))
			# line[0].set_data(self.trajectory[:, 0], self.trajectory[:, 1])
			return robots # + line

		anim = FuncAnimation(fig, update, frames=np.linspace(0, num_steps-1, num_steps, dtype=int), init_func=init, blit=True)

		plt.show()

		anim.save('filename.mp4')

	def simulate(self, num_steps, step_size):
		trajectory = np.linspace(self.start, self.goal, num_steps)
		self.set_trajectory(trajectory)
		self.robot_trajectories = [[] for robot in self.robots]
		for i in range(num_steps):
			for j, robot in enumerate(self.robots):
				robot.set_pose(self.trajectory[i])
				self.robot_trajectories[j].append(robot.get_pose())

if __name__ == '__main__':
	world = World(1000, 1000)
	start = np.array([100, 100])
	world.set_start(start)
	goal = np.array([900, 900])
	world.set_goal(goal)
	trajectory = np.array([[100, 200, 300, 400, 500, 600, 700, 800, 900], [100, 200, 300, 400, 500, 600, 700, 800, 900]])
	world.set_trajectory(trajectory)
	v = np.array([[300, 100], [400, 200], [300, 300], [200, 200]])
	obstacle = Obstacle(v)
	world.add_obstacle(obstacle)
	robot = Robot(start)
	world.add_robot(robot)
	num_steps = 100
	world.simulate(num_steps, None)
	world.animate(num_steps)