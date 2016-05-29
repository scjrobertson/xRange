"""
A module of use functions.

@author: scj robertson
@since 21/02/16
"""

def load_clubs(fname='clubs.csv'):
    '''
    Load the intial conditions for each club type. These are
    based on PGA tour averages as suppluied by the EMSS
    documentation.

    Parameters
    ----------

    fname : char
        The directory of the .csv (semicolon) file containing the club information.
        Format:
            -Club name : string
            -x-position : float
            -y-position : float
            -z-position : float
            -Launch angle : float (Degrees)
            -Direction angle: float (Degrees)
            -Speed : float (m/s)
            -Spin : float (RPM)
            -w-x : float (x component of angular velocity direction)
            -w-y : float (y component of angular velocity direction)
            -w-z : float (z component of angular velocity direction)

    Returns
    -----------

    names : (n, ) ndarray
        Names of the clubs

    p0 : (n, 3) ndarray
        Initial positions of the clubs

    v0 : (n, 3) ndarray
        Initial velocities of the clubs

    w0 : (n, 3) ndarray
        Initial positions of the clubs
    '''
    import numpy as np
    from pandas import read_csv

    raw = read_csv(fname, header=None, dtype = {0 : str}, delimiter=';').values
    i, j = raw.shape

    #Club names and initial positions
    names = raw[:, 0].reshape((i, 1))
    p0 = np.array(raw[:, 1:4], dtype=np.float64)

    #Determine velocity
    l_d = np.array(raw[:, 4:6], dtype=np.float64)*(np.pi/180)
    v0 = np.empty((i, 3), dtype=np.float64)
    v0[:, 0] = np.cos(l_d[:, 0])*np.cos(l_d[:, 1])*raw[:, 6]
    v0[:, 1] = np.cos(l_d[:, 0])*np.sin(l_d[:, 1])*raw[:, 6]
    v0[:, 2] = np.sin(l_d[:, 0])*raw[:, 6]
    #v0 = np.array(v0, dtype=np.float64)

    #Determine spin
    spin = np.array(raw[:, 7], dtype=np.float64).reshape((i, 1))*(np.pi/30)
    spin_axis = np.array(raw[:, 8:11], dtype=np.float64).reshape((i, 3))
    w0 = np.array(spin*spin_axis, dtype=np.float64)

    return names, p0, v0, w0

