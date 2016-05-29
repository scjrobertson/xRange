"""
Module containing the SensorArray class which 
models an array of FMCW radars.

@author: scj robertson
@since: 03/04/16
"""
import numpy as np

C = 3e8

class SensorArray(object):

    '''
    Class for representing an array of identical FMCW radars. For viable multilateration
    four or more sensors must always be specified at distinct locations.

    Parameters
    ----------
    sensor_locations : (N, 3) ndarray
        The locations of the sensors in 3D space. N >= 4 and at least
        four of these locations must be distinct.

    f_c : scalar
        The carrier frequency of the radar sensor.

    del_r : scalar
        The range resolution (m)

    r_max : scalar
        The maximum range (m) the radar can detect objects.
        Determines range bin width, not the Nyquist frequency for range.

    del_v : scalar
        The velocity resolution (m/s)

    v_max : scalar
        The maximum velocity (m/s) the radar can detect.
        Determines velocity bin width, not the Nyquist frequnecy for velocity.

    Methods
    ----------
    output
        Returns  range-Doppler maps for each sensor for a given collection of targets.

    Raises
    ----------
    ValueError
        If the are less than four distinct sensor locations.
    '''
    def __init__(self, sensor_locations, f_c, del_r, r_max, del_v, v_max):
        
        self.K, _ = sensor_locations.shape
        if (self.K < 4):
            ValueError('There must be K > 4 distinct sensor locations')
        
        self.f_c = f_c
        self.del_r = del_r
        self.r_max = r_max
        self.del_v = del_v
        self.v_max = v_max

        self.B = C/(2*del_r)
        self.T = C/(4*f_c*v_max)
        self.M = int((4*B*r_max)/c)
        self.N = int(C/(2*del_v*T*f_c))

    '''
    Determines a collection of range-Doppler maps for each sensor given 
    a simulated trajectory 
    '''
    def output(self, targets):
        I, J, _ = targets.shape
        dt = np.linspace(0, self.T, self.M)
        rd_map = np.zeros((self.K, self.M, self.N))

        return rd_map


