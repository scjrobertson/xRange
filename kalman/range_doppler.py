"""
Module for determining converting cartesian cooridnates and
velocity into Range-Doppler information and its
inverse transform, multilateration.

@author: scj robertson
@since: 23/02/16
"""
import numpy as np

def range_doppler(sensors, p, v):
    '''
    Convert a golf ball's trajectory and velocity information
    into the Range-Doppler information seen by the detector.
    This currently does not account for any measurement noise
    or missing information.

    Parameters
    ----------
    
    sensors : (m, 3) ndarray
        The 3D sensor locations.

    p : (n, 3) ndarray
        The golf ball's position in cartesian coordinates.

    v : (n, 3) ndarray
        The golf ball's velocity in cartesian coordinates.

    Returns
    ----------

    range_doppler : (m, n, 2)
        Range-Doppler for every position of the
        ball in the given trajectory.
    '''
    M, _ = sensors.shape
    N, _ = p.shape

    r_d = np.empty((M, N, 2))
    
    for i in np.arange(0, M):
        u = p - sensors[i, :]
        r_d[i, :, 0] = np.sqrt((u**2).sum(axis = 1))
        u = u/(r_d[i, :, 1][:, np.newaxis])
        r_d[i, :, 1] = np.multiply(v, u).sum(axis=1)

    return r_d
