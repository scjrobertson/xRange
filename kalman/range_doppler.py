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
    
    #Determine range and radial velocity as seen by each sensor
    for i in np.arange(0, M):
        u = p - sensors[i, :]
        r_d[i, :, 1] = np.linalg.norm(u, axis =1)
        u = u/(r_d[i, :, 1][:, np.newaxis])
        r_d[i, :, 0] = np.multiply(v, u).sum(axis=1)

    return r_d


def multilateration(s, d):
    '''
    Determine the golf ball's true position in 3D cartesian coordinates using the
    measured range data from the sensors. More than four sensor locations
    are required solve the system.

    Parameters
    ----------
    s : (m, 3) ndarray
        The 3D sensor locations.

    d : (n, m) ndarray
        The measure rnage information of the
        golf ball from each of the m-many sensors

    Returns
    ----------
    p : (n, 3) ndarray
        The golf ball's position in 3D space.
    '''
    
    #Pre-allocate a position matrix p
    N, M = d.shape
    p = np.empty((M, 3))

    #Determine A and perform QR decompisition
    A = s[1:, :] - s[0, :]
    Q, R = np.linalg.qr(A, mode='complete')
    R = 2*R

    #Pre-define a matrix n, square values of the sensor distances
    n = (s[1:, :]**2).sum(axis=1) - (s[0, :]**2).sum()
    n = np.repeat(n, M).reshape((N-1, M))

    #Determine the vector b and pre-multiply by Q.T
    b = (d[0, :]**2 -d[1:, :]**2) + n
    b = np.dot(Q.T, b)

    #Determine the 3D position for each set of range measurements
    #Is there a faster method?
    for i in np.arange(0, M):
        p[i, :] = np.linalg.lstsq(R, b[:, i])[0]
    
    return p

def determine_velocity(t, p, r_v):
    '''
    Determine the linear velocity of the golf ball
    from position and radial velocity measurements.

    Currently the function differentiates position
    measurements giving velocity.

    Parameters
    ----------
    t : (n, ) ndarray
        The sample intervals for the position
        vector p

    p : (n, 3) ndarray
        The position of the golf ball in cartesian 
        coordinates as determined by multilateration.

    r_v : (n, ) ndarray
        The radial velocity of the pall for each of the
        sample times.

    Returns
    ----------
    v_m : (n, 3) ndarray
        The linear velocity of the ball
        in cartesian coordinates.

    Raises
    ----------
    ValueError
        ValueError if the time, position and radial velocity 
        vectors dimensions are inconsistent.
    '''
    N, _ = p.shape

    if t.shape != (N, ) or t.shape != (N, 1):
        ValueError('Dimensions are inconsistent')

    return np.gradient(p, t.reshape(N, 1))[0]
