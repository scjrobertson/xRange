"""
This module contains a class which implements
a basic (linear) Kalman Filter.

@author: scj robertson
@since: 08/03/16
"""
import numpy as np

class KalmanFilter(object):

    """
    Class for representing and implementing a (linear) Kalman filter.
    Currently does raise any exceptions!

    Parameters
    ----------
    A_t : (n, n) ndarray
        State transition matrix.

    B_t : (n, m) ndarray
        Control matrix. m is the dimension of 
        the control vector u_t

    C_t : (n, n) ndarray
        Process measurement matrix. k is the dimension
        of the process measuremnt vector z_t.

    Q_t : (n, n) ndarray
        Covariance of state transistion noise.

    R_t : (n, n) ndarray
        Covariance of the measurement noise.

    mu_0 : (n, 1) ndarray
        The initial state estimate.

    sig_0 : (n, n) ndarray
        The initial estimate of the state's
        covariance matrix.

    Methods
    ----------
    step 
        Kalman filter algorithm, predict and update, for 
        a single dicrete time step.

    Example
    ----------
    See the kalman_filtering notebook, Kalman Filtering section.
    """
    def __init__(self, A_t, B_t, C_t, R_0, Q_0, mu_0, sig_0):
        self.A_t = A_t
        self.B_t = B_t
        self.C_t = C_t
        self.R_t = R_0
        self.Q_t = Q_0
        self.mu_t = mu_0
        self.sig_t = sig_0 
        self.I = np.eye(mu_0.shape[0])

    def step(self, u_t, z_t):
        '''
        Implements a Kalman filter algorithm for a discrete
        time step, as presesnted in Probabilistic Robotics 
        Chapter 3.

        Parameters
        ----------
        u_t : (n, 1) ndarray
            Current control vector.

        z_t : (n, 1) ndarray
            Current state's observation.

        Returns
        ----------
        mu_t : (n, 1) ndarray
            Current state's mean.
        '''
        #Brevity
        [A_t, B_t, C_t, R_t, Q_t, mu_t, sig_t, I] = [self.A_t, self.B_t, 
                self.C_t, self.R_t, self.Q_t, self.mu_t, self.sig_t, self.I]

        #Ensure correct dimensions
        N = z_t.shape[0]
        u_t = u_t.reshape((N, 1))
        z_t = z_t.reshape((N, 1))

        #Prediction
        mu_bar = A_t@mu_t + B_t@u_t
        sig_bar = A_t@sig_t@(A_t.T) + R_t
        
        #Kalman gain
        innov = np.linalg.inv(C_t@sig_bar@(C_t.T) + Q_t)
        K_t = sig_bar@(C_t.T)@innov

        #Measurement update
        mu_t = mu_bar + K_t@(z_t - C_t@mu_bar)
        sig_t = (I - K_t@C_t)@(sig_bar.T)

        return mu_t
