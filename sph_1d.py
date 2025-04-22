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

def W_cubic( x, h ):
    """
    Cubic Smoothing kernel (1D)
    """
    q = np.abs(x) / h
    sigma = 2.0 / 3.0 / h
    w = np.zeros_like(q)
    mask1 = (q >= 0) & (q < 1)
    mask2 = (q >= 1) & (q < 2)
    w[mask1] = 1 - 1.5 * q[mask1]**2 + 0.75 * q[mask1]**3
    w[mask2] = 0.25 * (2 - q[mask2])**3
    return sigma * w

def gradW_cubic( x, h ):
    """
    Gradient of the smoothing kernel (1D)
    """
    q = np.abs(x) / h
    sigma = 2.0 / 3.0 / h
    dw = np.zeros_like(q)
    mask1 = (q >= 0) & (q < 1)
    mask2 = (q >= 1) & (q < 2)
    dw[mask1] = -3 * q[mask1] + 2.25 * q[mask1]**2
    dw[mask2] = -0.75 * (2 - q[mask2])**2
    return sigma * dw * np.sign(x) / h

	
	
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
	

def getDensity( r, pos, m, h, L ):
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

    dx = dx - L * np.round(dx / L)

    rho = np.sum( m * W_cubic(dx, h), 1 ).reshape((M,1))

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

    du_dt = 0.5 * np.sum(m.T * ( P/rho**2 + P.T/rho.T**2) * (vx-vx.T) * dW, axis=1, keepdims=True).reshape((N,1))

    return du_dt


def getAcc( pos, vel, m, h, gamma, total_energy, lmbda, nu, dt, L):
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

    dWx = gradW_cubic( dx, h )

    dW = dWx

    rho = getDensity( pos, pos, m, h, L )

    # Get the pressures
    P = getPressure(rho, total_energy, gamma)

    dudt = getdu_dt(rho, P, m, vel, dW, N)

    # Add Pressure contribution to accelerations
    ax = - np.sum( m.T * ( P/rho**2 + P.T/rho.T**2  ) * dWx, 1).reshape((N,1))

    # pack together the acceleration components
    a = ax

    # Add external potential force
    #a -= lmbda * pos

    # Add viscosity
    #a -= nu * vel

    return a, dudt, P

def sine_bump(x, x_start, x_end):
    bump = np.zeros_like(x)
    mask = (x >= x_start) & (x <= x_end)
    bump[mask] = np.sin(np.pi * (x[mask] - x_start) / (x_end - x_start)) 
    return bump
	


def main():
    """ SPH simulation """

    # Simulation parameters
    N         = 1000    # Number of particles
    t         = 0      # current time of the simulation
    tEnd      = 2    # time at which simulation ends
    #dt        = 0.04   # timestep
    M         = 1      # star mass
    R         = 0.75   # star radius
    h         = 0.1    # smoothing length
    cfl = 0.5
    k         = 0.1    # equation of state constant
    n         = 1      # polytropic index
    nu        = 0.05      # damping
    gamma     = 5/3    # 
    plotRealTime = True # switch on for plotting as the simulation goes along

    # Generate Initial Conditions
    np.random.seed(42)            # set the random number generator seed

    lmbda = 2*k*(1+n)*np.pi**(-3/(2*n)) * (M*gam(5/2+n)/R**3/gam(1+n))**(1/n) / R**2  # ~ 2.01
    #m     = M/N                    # single particle mass

    # For a sound wave set up
    L = 1.0
    x = np.linspace(0, L, N)
    amplitude = 0.01
    rho0 = 1
    P0 = 1
    pos = x.reshape((N,1))   # randomly selected positions and velocities
    #print(pos.shape)
    k = 3 * 2*np.pi / (L)

    #bump = sine_bump(pos, 0.0, 0.1)

    initial_rho = rho0 * (1 + amplitude * np.sin(k * pos))
    #initial_rho = rho0 * np.ones((N,1))
    #initial_rho = rho0 * (1 + amplitude*bump)
    #print(initial_rho)
    

    cs = np.sqrt(gamma * P0 / rho0)
    dt = cfl * h / cs
    omega = k * cs
    #initial_P = P0 * np.ones((N,1))
    #initial_P = P0 * (1 + gamma * amplitude * np.sin(k * pos))
    initial_P = P0 * (initial_rho / rho0)**gamma


   #vel   = np.zeros(pos.shape)
    vel = amplitude * cs * np.sin(k * pos)
    #vel = amplitude * cs * bump

    x_true = np.linspace(0,1,100)
    true_density = rho0 * (1 + amplitude * np.sin(0 - k * x_true))

    # plt.figure()
    # plt.plot(pos, initial_rho, 'o')
    # plt.plot(x_true, true_density, '--')
    # plt.xlabel('x')
    # plt.ylabel('Density')
    # plt.show()

    m = initial_rho * L / N
    #m = np.ones((N,1)) * L / N

    true_energy = 0.5 * amplitude**2 * omega**2 * 2 * np.pi / k * M / L

    total_energy = initial_P / initial_rho / (gamma - 1)  # initial internal energy

    # plt.plot(pos[:,0], initial_rho, 'o', label="initial")
    # plt.plot(pos[:,0], getDensity( pos, pos, m, h, L ), 'o', label="initial sph")
    # plt.xlabel('x')
    # plt.ylabel('Density')
    # plt.legend( )
    # plt.show()
    # calculate initial gravitational accelerations
    acc, dudt, P = getAcc( pos, vel, m, h, gamma, total_energy, lmbda, nu, dt, L)

    #print(acc.shape)

    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))

    # set up array to track density
    rho_all = np.zeros((Nt+1, N))
    rho_all[0, :] = initial_rho.flatten()

    # set up array to track pressure
    P_all = np.zeros((Nt+1, N))
    P_all[0, :] = initial_P.flatten()

    # set up array to track velocity
    vel_all = np.zeros((Nt+1, N))
    vel_all[0, :] = vel.flatten()

    # set up array to track energy
    energy_all = np.zeros((Nt+1))
    energy_all[0] = np.sum(0.5 * m * vel**2 + m * total_energy)

    timesteps = np.empty(Nt+1)
    timesteps[0] = 0.0

    # prep figure
    fig = plt.figure(figsize=(15,8), dpi=80)
    grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.3)
    ax1 = plt.subplot(grid[0,0])
    ax2 = plt.subplot(grid[1,0])
    ax3 = plt.subplot(grid[0,1])
    ax4 = plt.subplot(grid[1,1])
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
        acc, dudt, P = getAcc( pos, vel, m, h, gamma, total_energy, lmbda, nu, dt, L )
        
        # (1/2) kick
        vel += acc * dt/2

        # update total energy
        total_energy += dudt * dt
        
        # update time
        t += dt

        conserved_energy = np.sum(0.5 * m.flatten() * vel.flatten()**2 + m.flatten() * total_energy.flatten())
        
        # get density for plotting
        rho = getDensity( pos, pos, m, h, L )
        
        rho_all[i+1, :] = rho.flatten()
        P_all[i+1, :] = P.flatten()
        vel_all[i+1, :] = vel.flatten()
        energy_all[i+1] = conserved_energy
        timesteps[i+1] = t
        true_density = rho0 * (1 + amplitude * np.sin( - omega * t + k * x_true))
        true_velocity = cs * amplitude * np.sin( - omega * t + k * x_true)
        true_P = P0 * (true_density / rho0)**gamma
        
        # plot in real time
        if plotRealTime or (i == Nt-1):
            plt.sca(ax1)
            plt.cla()
            #cval = np.minimum((rho-3)/3,1).flatten()
            plt.scatter(pos[:,0],rho[:,0], c='k', cmap=plt.cm.autumn, s=10, alpha=0.5)
            plt.plot(x_true, true_density, color='blue')
            plt.xlabel('x')
            plt.ylabel('density')
            ax1.set(xlim=(0, 1), ylim=(0.9, 1.1))
            # ax1.set_aspect('equal', 'box')
            # ax1.set_xticks([-1,0,1])
            #ax1.set_facecolor((.1,.1,.1))

            plt.sca(ax2)
            plt.cla()
            #cval = np.minimum((rho-3)/3,1).flatten()
            plt.scatter(pos[:,0],P[:,0], c='k', cmap=plt.cm.autumn, s=10, alpha=0.5)
            plt.plot(x_true, true_P, color='blue')
            plt.xlabel('x')
            plt.ylabel('Pressure')
            ax2.set(xlim=(0, 1), ylim=(0.85, 1.2))
            # ax1.set_aspect('equal', 'box')
            # ax1.set_xticks([-1,0,1])
            #ax2.set_facecolor((.1,.1,.1))

            plt.sca(ax3)
            plt.cla()
            #cval = np.minimum((rho-3)/3,1).flatten()
            plt.scatter(pos[:,0],vel[:,0], c='k', cmap=plt.cm.autumn, s=10, alpha=0.5)
            plt.plot(x_true, true_velocity, color='blue')
            plt.xlabel('x')
            plt.ylabel('Velocity')
            ax3.set(xlim=(0, 1), ylim=(-0.2, 0.2))
            # ax1.set_aspect('equal', 'box')
            # ax1.set_xticks([-1,0,1])
            #ax3.set_facecolor((.1,.1,.1))

            plt.sca(ax4)
            plt.cla()
            #cval = np.minimum((rho-3)/3,1).flatten()
            plt.plot(timesteps[:i+1],energy_all[:i+1], c='k')
            #plt.plot(timesteps[:i+1], [true_energy for t in timesteps[:i+1]], color='blue')
            plt.xlabel('t')
            plt.ylabel('Energy')
            ax4.set(xlim=(0, tEnd))
            # ax1.set_aspect('equal', 'box')
            # ax1.set_xticks([-1,0,1])
            #ax4.set_facecolor((.1,.1,.1))
            
            fig.savefig("frame_%04d.png" % i)
            plt.pause(0.001)
        
    #add labels/legend
    frames = []
    for i in range(Nt):
        filename = f"frame_%04d.png" % i
        image = imageio.imread(filename)
        frames.append(image)
    imageio.mimsave("animation_all_paper_3_sound_wave.gif", frames, loop=0)

    # Clean up individual frame files (optional)
    for i in range(Nt):
        os.remove(f"frame_%04d.png" % i)	
        
    #Save figure
    plt.savefig('sph_1d_paper_3_sound_wave.png',dpi=240)
    plt.show()
        
    return 0
	


  
if __name__== "__main__":
    main()
