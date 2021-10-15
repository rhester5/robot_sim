
import numpy as np

from Controller import Controller

class QuadrotorPID(Controller):
	"""docstring for QuadrotorPID"""
	def __init__(self, gains, dt):
		super(QuadrotorPID, self).__init__(gains, dt)
		self.last_error = np.zeros(2)
		self.total_error = np.zeros(2)

	def __call__(self, x, sp):
		p_error = sp - x
		d_error = (p_error - self.last_error) / self.dt
		i_error = p_error + self.total_error

		control = self.kp * p_error + self.kd * d_error + self.ki * i_error

		self.last_error = p_error
		self.total_error = i_error

		return control

	def set_gains(self, gains):
		kp, kd, ki = gains
		self.kp = kp
		self.kd = kd
		self.ki = ki

	def get_gains(self):
		return self.kp, self.kd, self.ki
		

if __name__ == '__main__':
	main()