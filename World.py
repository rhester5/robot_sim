
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path

from Map import Map
from Obstacle import Obstacle
from Robot import Robot
from RRT import RRT
from MotionModel import MotionModel
from MeasurementModel import MeasurementModel
from LinearQuadrotorDynamics import LinearQuadrotorDynamics
from QuadrotorPID import QuadrotorPID

np.random.seed(0)

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

	def get_start(self):
		return self.start
		
	def set_goal(self, goal):
		self.goal = goal

	def get_goal(self):
		return self.goal

	def set_trajectory(self, trajectory):
		self.trajectory = trajectory

	def get_trajectory(self):
		return self.trajectory

	def set_plan(self, plan):
		self.plan = plan

	def get_plan(self):
		return self.plan

	def set_setpoint(self, setpoint):
		self.setpoint = setpoint

	def get_setpoint(self):
		return self.setpoint

	def render(self):
		plt.figure()
		self.map.render()
		plt.plot(self.trajectory[0, :], self.trajectory[1, :], 'b')
		plt.scatter(self.start[0], self.start[1], c='g')
		plt.scatter(self.goal[0], self.goal[1], c='r')
		for robot in self.robots:
			plt.scatter(robot.get_pose()[0], robot.get_pose()[1], c='b')
		plt.show()

	def animate(self):
		trajectory = world.get_trajectory()
		plan = world.get_plan()
		setpoint = world.get_setpoint()
		num_steps = trajectory.shape[0]
		fig, ax = plt.subplots()
		self.map.render()
		# line = plt.plot(self.trajectory[:, 0], self.trajectory[:, 1], 'tab:blue')
		line = plt.plot(plan[:, 0], plan[:, 1], 'tab:red')
		traj = plt.plot([], [], 'tab:blue')
		plt.scatter(self.start[0], self.start[1], c='g')
		plt.scatter(self.goal[0], self.goal[1], c='r')
		robots = []
		sp = ax.scatter(setpoint[0, 0], setpoint[0, 1], c='g', marker='*')
		for robot in self.robots:
			robots.append(ax.scatter(robot.get_pose()[0], robot.get_pose()[1], c='b', marker='*'))

		def init():
			# return robots # + line
			return traj + robots + [sp]

		# plot trajectory as you go
		# plot setpoint as you go

		def update(i):
			# print(i)
			for j, robot in enumerate(robots):
				robot.set_offsets(self.robot_trajectories[j][i].reshape((1, 2)))
			# line[0].set_data(self.trajectory[:, 0], self.trajectory[:, 1])
			sp.set_offsets([setpoint[i, 0], setpoint[i, 1]])
			# return robots # + line
			if not i:
				traj[0].set_data([], [])
			traj[0].set_data(trajectory[:i+1, 0], trajectory[:i+1, 1])
			return traj + robots + [sp]

		anim = FuncAnimation(fig, update, frames=np.linspace(0, num_steps-1, num_steps, dtype=int), init_func=init, blit=True, interval=10)
		
		print('showing')
		plt.show()
		print('saving')
		anim.save('follow.mp4')
		print('saved')

	# def animate_plan(self, xs, ys):
	# 	num_steps = self.trajectory.shape[0]
	# 	fig, ax = plt.subplots()
	# 	self.map.render()
	# 	# line = plt.plot(self.trajectory[:, 0], self.trajectory[:, 1], 'tab:blue')
	# 	plt.scatter(self.start[0], self.start[1], c='g')
	# 	plt.scatter(self.goal[0], self.goal[1], c='r')
	# 	robots = []
	# 	# lines = []
	# 	lines = set()
	# 	for robot in self.robots:
	# 		# robots.append(ax.scatter(robot.get_pose()[0], robot.get_pose()[1], c='b', marker='*'))
	# 		robots.append(ax.scatter(self.start[0], self.start[1], c='b', marker='*'))

	# 	def init():
	# 		return robots # + line
	# 		# return robots + lines

	# 	def update(i):
	# 		if i >= len(xs):
	# 			for j, robot in enumerate(robots):
	# 				robot.set_offsets(self.robot_trajectories[j][i-len(xs)].reshape((1, 2)))
	# 			# line[0].set_data(self.trajectory[:, 0], self.trajectory[:, 1])
	# 		else:
	# 			for j in range(len(xs[i])):
	# 				# lines.append(ax.plot(xs[i][j], ys[i][j], 'b'))
	# 				# lines.add(ax.plot(xs[i][j], ys[i][j], 'b'))
	# 				lines.add(ax.plot(xs[i][j], ys[i][j], 'b')[0])
	# 		# return robots # + line
	# 		# print(robots)
	# 		# print(lines)
	# 		# print(robots + [lines[k][0] for k in range(len(lines))])
	# 		# return robots + [lines[k][0] for k in range(len(lines))]
	# 		plot_lines = list(lines)
	# 		return robots + [plot_lines[k] for k in range(len(plot_lines))]

	# 	anim = FuncAnimation(fig, update, frames=np.linspace(0, len(xs)+num_steps-1, len(xs)+num_steps, dtype=int), init_func=init, blit=True)

	# 	plt.show()

	# 	anim.save('filename.mp4')

	def animate_plan(self, xs, ys):
		fig, ax = plt.subplots()
		self.map.render()
		# line = plt.plot(self.trajectory[:, 0], self.trajectory[:, 1], 'tab:blue')
		plt.scatter(self.start[0], self.start[1], c='g')
		plt.scatter(self.goal[0], self.goal[1], c='r')
		# robots = []
		# lines = []
		# # lines = set()
		# for robot in self.robots:
		# 	# robots.append(ax.scatter(robot.get_pose()[0], robot.get_pose()[1], c='b', marker='*'))
		# 	robots.append(ax.scatter(self.start[0], self.start[1], c='b', marker='*'))
		# lines = [ax.plot([], [], 'b')[0] for i in range(len(xs))]
		lines = [plt.plot([], [], 'b')[0] for i in range(len(xs))]

		def init():
			# return robots # + line
			# return robots + lines
			# lines = [ax.plot([], [], 'b')[0] for i in range(len(xs))]
			print('init')
			# lines = [plt.plot([], [], 'b')[0] for i in range(len(xs))]
			return lines

		def update(i):
			print(i)
			if not i:
				for j in range(len(xs)):
					lines[j].set_data([], [])
			# for j in range(len(xs[i])):
			# 	# lines.append(ax.plot(xs[i][j], ys[i][j], 'b'))
			# 	# lines.add(ax.plot(xs[i][j], ys[i][j], 'b'))
			# 	print(i, j)
			# 	lines.add(ax.plot(xs[i][j], ys[i][j], 'b')[0])
			# lines.append(ax.plot(xs[i][-1], ys[i][-1], 'b')[0])
			lines[i].set_data(xs[i][-1], ys[i][-1])
			# return robots # + line
			# print(robots)
			# print(lines)
			# print(robots + [lines[k][0] for k in range(len(lines))])
			# return robots + [lines[k][0] for k in range(len(lines))]
			# plot_lines = list(lines)
			# return robots + [plot_lines[k] for k in range(len(plot_lines))]
			return lines

		anim = FuncAnimation(fig, update, frames=np.linspace(0, len(xs)-1, len(xs), dtype=int), init_func=init, blit=True, interval=50)

		print('showing')
		plt.show()
		print('saving')
		anim.save('plan.mp4')
		print('saved')

	# def simulate(self, num_steps, step_size):
		# trajectory = np.linspace(self.start, self.goal, num_steps)
		# self.set_trajectory(trajectory)
		# num_steps = self.trajectory.shape[0]
		# self.robot_trajectories = [[] for robot in self.robots]
		# for i in range(num_steps):
		# 	for j, robot in enumerate(self.robots):
		# 		robot.set_pose(self.trajectory[i])
		# 		self.robot_trajectories[j].append(robot.get_pose())
	def simulate(self, K, x0, controller, motion_model, meas_model, lookahead, thresh):

		plan = self.get_plan()

		m = motion_model.get_state_size()
		n = meas_model.get_meas_size()

		state = np.zeros((K, m))
		meas = np.zeros((K, n))
		# setpoint = np.zeros((K, m))
		setpoint = np.zeros((K, 2))

		# initial state
		x = np.zeros(m)
		x[0:2] = x0
		i = 0
		waypoint = np.zeros(m)
		waypoint[0:2] = plan[i]
		if np.linalg.norm(waypoint[0:2] - x[0:2]) < thresh:
				i += 1
				if i < len(plan):
					waypoint[0:2] = plan[i]
		for k in range(K):
			print(k)
			# kp1 = k+1 if k < K-1 else k
			sp = x[0:2] + (waypoint[0:2] - x[0:2]) * lookahead / np.linalg.norm(waypoint[0:2] - x[0:2])
			# if i > 0:
			# 	last_waypoint = plan[i-1]
			# else:
			# 	last_waypoint = plan[0]
			# sp = last_waypoint + (waypoint[0:2] - x[0:2]) * lookahead / np.linalg.norm(waypoint[0:2] - x[0:2])
			u = controller(x[0:2], sp)
			if k == 0:
				control = np.zeros((K, len(u)))

			print(x, waypoint[0:2], sp, u)
			
			x = motion_model(x, u)
			z = meas_model(x)

			state[k, :] = x
			meas[k, :] = z
			setpoint[k, :] = sp
			control[k, :] = u

			if np.linalg.norm(waypoint[0:2] - x[0:2]) < thresh:
				i += 1
				if i < len(plan):
					waypoint[0:2] = plan[i]

		world.set_trajectory(state)
		world.set_setpoint(setpoint)
		# print('state', state)
		# print('setpoint', setpoint)

		self.robot_trajectories = [[] for robot in self.robots]
		for i in range(num_steps):
			for j, robot in enumerate(self.robots):
				robot.set_pose(self.trajectory[i, 0:2])
				self.robot_trajectories[j].append(robot.get_pose())

		return state, meas, setpoint, control

if __name__ == '__main__':
	x = 1000
	y = 1000
	world = World(x, y)
	start = np.array([100, 100])
	world.set_start(start)
	goal = np.array([900, 900])
	world.set_goal(goal)
	# trajectory = np.array([[100, 200, 300, 400, 500, 600, 700, 800, 900], [100, 200, 300, 400, 500, 600, 700, 800, 900]])
	# world.set_trajectory(trajectory)
	# v = np.array([[300, 100], [400, 200], [300, 300], [200, 200]])
	# obstacle = Obstacle(v)
	# world.add_obstacle(obstacle)
	num_obstacles = 25
	obs_radius = 100
	num_points = 10
	for i in range(num_obstacles):
		center = np.random.randint((x-obs_radius, y-obs_radius))
		pts = np.random.randint((obs_radius, obs_radius), size=(num_points, 2))
		obstacle = Obstacle(pts+center)
		if not obstacle.point_intersection(start) and not obstacle.point_intersection(goal):
			world.add_obstacle(obstacle)
	robot = Robot(start)
	world.add_robot(robot)
	num_iter = 10000
	rrt_radius = 25
	sample_goal = 10
	planner = RRT(world.map, num_iter, rrt_radius, sample_goal)
	# trajectory, xs, ys = planner.plan(start, goal)
	plan, xs, ys = planner.plan(start, goal)
	world.set_plan(plan)
	# print(xs, ys)
	# print(len(xs))
	# print(xs[0])
	# print(xs[-1])
	# world.set_trajectory(trajectory)
	num_steps = 5000
	dt = 0.1
	gains = [4, 4, 0.01]
	motion_model_cov = [0, 0]
	meas_model_cov = [0, 0]
	controller = QuadrotorPID(gains, dt)
	dynamics = LinearQuadrotorDynamics(dt, motion_model_cov, meas_model_cov)
	A, B, Q, H, R = dynamics.get_params()
	motion_model = MotionModel([A, B, Q])
	meas_model = MeasurementModel(H, R, True)
	lookahead = 1
	thresh = 1
	world.simulate(num_steps, world.get_start(), controller, motion_model, meas_model, lookahead, thresh)
	# print('animate plan')
	# world.animate_plan(xs, ys)
	# print('animate follow')
	world.animate()