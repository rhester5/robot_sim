import numpy as np


class KalmanFilter(object):
    def __init__(self, A, B, H, Q, R, x_0, P_0):
        # Model parameters
        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R

        # Initial state
        self._x = x_0
        self._P = P_0

    def predict(self, u):
        self._x = np.matmul(self.A, self._x) + np.matmul(self.B, u)
        self._P = np.matmul(self.A, np.matmul(self._P, self.A.transpose())) + self.Q

    def update(self, z):
        self.S = np.matmul(self.H, np.matmul(self._P, self.H.transpose())) + self.R
        self.V = z - np.matmul(self.H, self._x)
        self.K = np.matmul(self._P, np.matmul(self.H.transpose(), np.linalg.inv(self.S)))

        self._x = self._x + np.matmul(self.K, self.V)
        #self._P = self._P - np.matmul(self.K, np.matmul(self.S, self.K.transpose()))
        self._P = self._P - np.matmul(self.K, np.matmul(self.H, self._P))

    def get_state(self):
        return self._x, self._P

    def set_init(self, x_0, P_0):
        self._x = x_0
        self._P = P_0

    def set_init_state(self, x_0):
        self._x = x_0