"""
Module for calculating a golf ball's trajectory using EMSS's model.

@author: scj robertson

@since: 19/02/16
"""
import numpy as np

def calc_traj(p0, v0, w0, t):
    """
    Calculates the trajectory of a golf ball given its initial conditions
    and flight time. All the interchanges of the y, z are taken care of
    in this function, there is no need to alter p0, v0, w0.

    Parameters
    ----------
    p0 : (3, ) ndarray
        The initial position of the golf ball.
    
    v0 : (3, ) ndarray
        The initial velocity of the golf ball.

    w0 : (3, ) ndarray
        The initial angular velocity of the golf ball, given in radians.

    t : (m, ) ndarray
        Array of time m-many time instances. 
   
    Returns
    ----------
    t : (n, ) ndarray
        n-many valid time instances

    p : (n, 3) ndarray
        n-many position samples on the golf ball's trajectory. 

    v : (n, 3) ndarray
    	n-many velocity samples on the golf ball's trajectory.
    """
    from scipy.integrate import odeint

    #Drag and lift coefficients
    c_d1 = 0.20
    c_d2 = 0.18
    c_d3 = 0.06
    c_l1 = 0.54
    a_1 = 90000
    a_2 = 200000

    #Spin decay rate coefficient
    r_1 = 2e-5
    
    #Air density and viscosity
    rho = 1.225
    mu = 1.8e-5
    
    #Radius and area
    r = 42.67/2000
    a = np.pi*(r**2)
    
    #Mass and gravity
    g = 9.81
    m = 4.593e-2

    #Swap the coordinates
    p_s0 = np.array([p0[0], p0[2], p0[1]])
    v_s0 = np.array([v0[0], v0[2], v0[1]])

    #Solve the system of ODEs
    pvw0 = np.array((p_s0, v_s0, w0)).reshape((9, ))
    k = np.asarray([c_d1, c_d2, c_d3, c_l1, a_1, a_2, r_1, rho, mu, r, a, g, m])
    f = odeint(ode_sys, pvw0, t, args=(k,))

    #Swap the coordinates back
    p = np.array([f[:, 0], f[:, 2], f[:, 1]]).T
    v = np.array([f[:, 3], f[:, 5], f[:, 4]]).T

    #Only valid where z>0
    wl = f[:, 1] > 0

    return t[wl], p[wl, :], v[wl, :]

def ode_sys(w, t, k):
    """
    EMSS's model's system of differential equations. x,y and z equations are as 
    defined in the documentation.

    Parameters
    ----------
    w : (1, 9) ndarray
        Vector of previous state variables.
        (px, py, pz, vx, vy, vz, wx, wy, wz)

    t : float
        Time

    k : (1, 13) ndarray
        Vector of parameters 
        (cd_1, cd_2, cd_3, c_l1, a1, a2, srd, rho, mu, r, a, g, m)

    Returns
    ----------
    f : (1, 9) ndarray
        Vector of current state variables.
        (px, py, pz, vx, vy, vz, wx, wy, wz)

    """

    #Assign to local variables
    px, py, pz, vx, vy, vz, wx, wy, wz = w.tolist() 
    c_d1, c_d2, c_d3, c_l1, a_1, a_2, r_1, rho, mu, r, a, g, m = k.tolist()

    #Linear and angular speed
    v_s = np.sqrt(vx**2 + vy**2 + vz**2)
    w_s = np.sqrt(wx**2 + wy**2 + wz**2)

    #Reynolds number, spin, drag and lift
    r_e = (rho*v_s*2*r)/mu
    s = (w_s*r)/(v_s)
    srd = r_1*s
    c_l = c_l1*(s**0.45)
    c_d = c_d1 + c_d2*s + c_d3*np.sin(np.pi*((r_e - a_1)/a_2))

    #Pre-allocate the state variables
    f = np.zeros((9, ))

    #Position
    f[0] = vx
    f[1] = vy
    f[2] = vz

    #Velocity
    f[3] = -((rho*a*v_s)/(2*m))*(c_d*vx - (c_l/w_s)*(wy*vz - wz*vy)) 
    f[4] = -g - ((rho*a*v_s)/(2*m))*(c_d*vy - (c_l/w_s)*(wz*vx - wx*vz)) 
    f[5] = -((rho*a*v_s)/(2*m))*(c_d*vz - (c_l/w_s)*(wx*vy - wy*vx)) 

    #Spin
    f[6] = -(srd*wx*v_s)/r
    f[7] = -(srd*wy*v_s)/r
    f[8] = -(srd*wz*v_s)/r

    return f




    





