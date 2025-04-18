import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gam
import astropy.constants as const

import imageio
import os

"""
Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the structure of a star with SPH
"""

def W( x, h ):
    """
    Gausssian Smoothing kernel (3D)
    x     is a vector/matrix of x positions
    h     is the smoothing length
    w     is the evaluated smoothing function
    """
    
    w = (1.0 / (h*np.sqrt(np.pi))) * np.exp( -x**2 / h**2)
    
    return w
	
	
def gradW( x, h ):
    """
    Gradient of the Gausssian Smoothing kernel (3D)
    x     is a vector/matrix of x positions
    y     is a vector/matrix of y positions
    z     is a vector/matrix of z positions
    h     is the smoothing length
    wx, wy, wz     is the evaluated gradient
    """

    n = -2 * np.exp( -x**2 / h**2) / h**3 / (np.pi)**(1/2)
    wx = n * x

    return wx
	
	
def getPairwiseSeparations( ri, rj ):
    """
    Get pairwise desprations between 2 sets of coordinates
    ri    is an M x 3 matrix of positions
    rj    is an N x 3 matrix of positions
    dx, dy, dz   are M x N matrices of separations
    """

    M = ri.shape[0]
    N = rj.shape[0]

    # positions ri = (x,y,z)
    rix = ri[:,0].reshape((M,1))

    # other set of points positions rj = (x,y,z)
    rjx = rj[:,0].reshape((N,1))

    # matrices that store all pairwise particle separations: r_i - r_j
    dx = rix - rjx.T

    return dx
	

def getDensity( r, pos, m, h ):
    """
    Get Density at sampling loctions from SPH particle distribution
    r     is an M x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    m     is the particle mass
    h     is the smoothing length
    rho   is M x 1 vector of densities
    """

    M = r.shape[0]

    dx = getPairwiseSeparations( r, pos );

    rho = np.sum( m * W(dx, h), 1 ).reshape((M,1))

    return rho
	
	
def getPressure_polytropic(rho, k, n):
    """
    Equation of State
    rho   vector of densities
    k     equation of state constant
    n     polytropic index
    P     pressure
    """
    P = k * rho**(1+1/n)

    return P


def getPressure(rho, total_energy, gamma):
    """
    Equation of State
    rho   vector of densities
    k     equation of state constant
    n     polytropic index
    P     pressure
    """
    P = (gamma - 1) * rho * total_energy

    return P

def getdu_dt(rho, P, m, vel, dW, N):
    """
    Get the change in internal energy of the system

    rho   vector of densities
    K     equation of state constant
    gamma polytropic index
    du_dt  change in internal energy
    """

    vx = vel[:,0].reshape((N,1))

    du_dt = 0.5 * np.sum(m * ( P/rho**2 + P.T/rho.T**2) * (dW*(vx-vx.T)), 1).reshape((N,1))

    return du_dt


def getAcc( pos, vel, m, h, gamma, total_energy, lmbda, nu, dt, L ):
    """
    Calculate the acceleration on each SPH particle
    pos   is an N x 3 matrix of positions
    vel   is an N x 3 matrix of velocities
    m     is the particle mass
    h     is the smoothing length
    k     equation of state constant
    n     polytropic index
    lmbda external force constant
    nu    viscosity
    a     is N x 3 matrix of accelerations
    """

    N = pos.shape[0]

    # Get pairwise distances and gradients
    dx = getPairwiseSeparations( pos, pos )

    dx = dx - L* np.round(dx/L)

    dWx = gradW( dx, h )

    dW = dWx

    # Calculate densities at the position of the particles
    rho = getDensity( pos, pos, m, h )

    # Get the pressures
    P = getPressure(rho, total_energy, gamma)

    dudt = getdu_dt(rho, P, m, vel, dW, N)

    total_energy += dudt * dt

    # Add Pressure contribution to accelerations
    ax = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWx, 1).reshape((N,1))

    # pack together the acceleration components
    a = ax

    # Add external potential force
    #a -= lmbda * pos

    # Add viscosity
    #a -= nu * vel

    return a, total_energy
	


def main():
    """ SPH simulation """

    # Simulation parameters
    N         = 400    # Number of particles
    t         = 0      # current time of the simulation
    tEnd      = 12     # time at which simulation ends
    #dt        = 0.04   # timestep
    M         = 1      # star mass
    R         = 0.75   # star radius
    h         = 0.15    # smoothing length
    cfl = 0.9
    k         = 0.1    # equation of state constant
    n         = 1      # polytropic index
    nu        = 1      # damping
    gamma     = 5/3    # 
    plotRealTime = True # switch on for plotting as the simulation goes along

    # Generate Initial Conditions
    np.random.seed(42)            # set the random number generator seed

    lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gam(5/2+n)/R**3/gam(1+n))**(1/n) / R**2  # ~ 2.01
    m     = M/N                    # single particle mass

    # For a sound wave set up
    L = 1.0
    x = np.linspace(0, L, N)
    amplitude = 0.01
    rho0 = 1
    P0 = 1
    pos = x.reshape((N,1))   # randomly selected positions and velocities
    #print(pos.shape)
    k = 2*np.pi / L
    initial_rho = rho0 * (1 + amplitude * np.sin(k * pos))
    #print(initial_rho)
    

    cs = np.sqrt(gamma * P0 / rho0)
    dt = cfl * h / cs
    omega = k * cs
    initial_P = P0 * (initial_rho/rho0)**gamma

    vel   = amplitude * cs * np.sin(k*pos)

    x_true = np.linspace(0,1,100)
    true_density = rho0 * (1 + amplitude * np.sin(0 - k * x_true))

    # plt.figure()
    # plt.plot(pos, initial_rho, 'o')
    # plt.plot(x_true, true_density, '--')
    # plt.xlabel('x')
    # plt.ylabel('Density')
    # plt.show()

    #m = initial_rho * L / N

    total_energy = initial_P / initial_rho / (gamma - 1)  # initial internal energy

    # calculate initial gravitational accelerations
    acc, total_energy = getAcc( pos, vel, m, h, gamma, total_energy, lmbda, nu, dt, L )

    #print(acc.shape)

    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))

    rho_all = np.zeros((Nt+1, N))
    rho_all[0, :] = initial_rho.flatten()

    # prep figure
    fig = plt.figure(figsize=(10,5), dpi=80)
    grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0,0])
    plt.sca(ax1)
    plt.xlabel('x')
    plt.ylabel('density')
    # rr = np.zeros((100,3))
    # rlin = np.linspace(0,1,100)
    # rr[:,0] =rlin

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt/2
        
        # drift
        pos += vel * dt
        
        pos = pos % L
        
        # update accelerations
        acc, total_energy = getAcc( pos, vel, m, h, gamma, total_energy, lmbda, nu, dt, L )
        
        # (1/2) kick
        vel += acc * dt/2
        
        # update time
        t += dt
        
        # get density for plotting
        rho = getDensity( pos, pos, m, h )
        
        rho_all[i+1, :] = rho.flatten()
        true_density = rho0 * (1 + amplitude * np.sin(omega * t - k * x_true))
        
        # plot in real time
        if plotRealTime or (i == Nt-1):
            plt.sca(ax1)
            plt.cla()
            cval = np.minimum((rho-3)/3,1).flatten()
            plt.scatter(pos[:,0],rho[:,0], c='white', cmap=plt.cm.autumn, s=10, alpha=0.5)
            plt.plot(x_true, true_density, color='blue')
            ax1.set(xlim=(0, 1), ylim=(0, 2))
            # ax1.set_aspect('equal', 'box')
            # ax1.set_xticks([-1,0,1])
            ax1.set_facecolor('black')
            ax1.set_facecolor((.1,.1,.1))
            
            # plt.sca(ax2)
            # plt.cla()
            # ax2.set(xlim=(0, 1), ylim=(0, 3))
            # ax2.set_aspect(0.1)
            # plt.plot(rlin, rho_analytic, color='gray', linewidth=2)
            # rho_radial = getDensity( rr, pos, m, h )
            # plt.plot(rlin, rho_radial, color='blue')
            plt.savefig("frame_%04d.png" % i)
            plt.pause(0.001)
        
    # add labels/legend
    frames = []
    for i in range(Nt):
        filename = f"frame_%04d.png" % i
        image = imageio.imread(filename)
        frames.append(image)
    imageio.mimsave("animation2.gif", frames, loop=0)

    # Clean up individual frame files (optional)
    for i in range(Nt):
        os.remove(f"frame_%04d.png" % i)	
        
    plt.sca(ax1)
    # # Save figure
    plt.savefig('sph_1d.png',dpi=240)
    plt.show()
        
    return 0
	


  
if __name__== "__main__":
    main()
