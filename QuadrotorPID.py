
import numpy as np

from Controller import Controller

class QuadrotorPID(Controller):
	"""docstring for QuadrotorPID"""
	def __init__(self, gains, actuator_limits, dt):
		super(QuadrotorPID, self).__init__(gains, actuator_limits, dt)
		self.last_error = np.zeros(2)
		self.total_error = np.zeros(2)
		self.last_sp = None

	def __call__(self, x, sp):
		if np.any(sp != self.last_sp):
			self.total_error = 0

		p_error = sp - x
		d_error = (p_error - self.last_error) / self.dt
		i_error = p_error + self.total_error

		control = self.kp * p_error + self.kd * d_error + self.ki * i_error

		self.last_error = p_error
		self.total_error = i_error
		self.last_sp = sp

		if np.linalg.norm(control) > self.actuator_limits:
			control = control / np.linalg.norm(control) * self.actuator_limits

		return control

	def set_gains(self, gains):
		kp, kd, ki = gains
		self.kp = kp
		self.kd = kd
		self.ki = ki

	def get_gains(self):
		return self.kp, self.kd, self.ki

	def set_actuator_limits(self, actuator_limits):
		self.actuator_limits = actuator_limits

	def get_actuator_limits(self):
		return self.actuator_limits
		

if __name__ == '__main__':
	main()