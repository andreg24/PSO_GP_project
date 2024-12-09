import numpy as np
import matplotlib.pyplot as plt
import random
from collections.abc import Callable
from landscapes import *



# utils function

def compute_magnitude(pts):
    """Return a vector of magnitudes of an array of points in R^2"""
    return np.sqrt(np.power(pts[:, 0], 2) + np.power(pts[:, 1], 2))   

def compute_center_mass(pts):
    """Return the center of mass of points in R^2"""
    return np.array([pts.mean(axis=0)])

def compute_dispersion(pts, center_mass):
    """Return the dispersion of the points as the average distance from the center of mass"""
    return compute_magnitude(pts - center_mass).mean()

def distance(pts, reference_point):
    """Returns the distances of each point in pts from the reference point"""
    return compute_magnitude(pts - reference_point)


# Standard Swarm Class

class Swarm:
    """Class swarm that implementes PSO for 2D problems"""

    def __init__(
        self,
        n,
        fit,
        soc_factor=1.49,
        cog_factor=1.49,
        inertia_weight=0.9,
        inertia_decay=0.99,
        inertia_min = 0,
        vel_magnitude_limits=[0.01, 5],
        ):

        self.n = n # number of particles
        self.fit = fit # fitness function

        self.soc_factor = soc_factor
        self.cog_factor = cog_factor
        self.inertia_weight = inertia_weight
        self.inertia_decay = inertia_decay
        self.inertia_min = inertia_min

        self.vel_magnitude_limits = vel_magnitude_limits # min and max velocity magnitude

        # initialize
        self.swarm = None
        self.velocity = None
        self.initialization()

        self.personal_best = self.swarm
        self.global_best = np.array([self.personal_best[np.argmin(self.fit(self.personal_best))]])
        self.center_mass = compute_center_mass(self.swarm)
        self.dispersion = compute_dispersion(self.swarm, self.center_mass) # average distance from center of mass

    def initialization(self, gen_position_limits=np.array([[-5, 5]]*2)):
        
        """Function dealing with the initialization of the particles initial positions and velocities"""

        # initialize population
        self.swarm = np.random.uniform(
            gen_position_limits[:, 0], 
            gen_position_limits[:, 1], 
            size=(self.n, 2))
        
        # initialize velocity
        self.velocity = np.zeros((self.n, 2))

    def update_velocity(self):
        # stochastic elements
        r1 = np.random.uniform(np.zeros(2), np.ones(2), size=(self.n, 2))
        r2 = np.random.uniform(np.zeros(2), np.ones(2), size=(self.n, 2))
        
        # factors
        inertia = self.velocity * self.inertia_weight
        social = self.soc_factor * r1 * (self.global_best - self.swarm)
        cognitive = self.cog_factor * r2 * (self.personal_best - self.swarm)

        # update velocity
        self.velocity = inertia + social + cognitive
        # self.inertia_weight = self.inertia_weight - self.inertia_decay
        # if self.inertia_weight < self.inertia_min:
            # self.inertia_weight = self.inertia_min

        # check velocity lower/upper magnitude limit
        # MODIFY TO DEAL WITH ZERO VELOCITY!!!!
        magnitude = compute_magnitude(self.velocity)
        large_vel = magnitude > self.vel_magnitude_limits[1]
        small_vel = (magnitude < self.vel_magnitude_limits[0]) & (magnitude > 0)
        if large_vel.sum() > 0:
            self.velocity[large_vel] = self.velocity[large_vel] / np.array([[magnitude[large_vel]]]).reshape(large_vel.sum(), 1) * self.vel_magnitude_limits[1]
        if small_vel.sum() > 0:
            self.velocity[small_vel] = self.velocity[small_vel] / np.array([[magnitude[small_vel]]]).reshape(small_vel.sum(), 1) * self.vel_magnitude_limits[0]

    def update_position(self):
        self.swarm = self.swarm + self.velocity

        # update best
        indexes = self.fit(self.personal_best) > self.fit(self.swarm)
        self.personal_best[indexes] = self.swarm[indexes]
        self.global_best = np.array([self.personal_best[np.argmin(self.fit(self.personal_best))]])

        # update center mass
        self.center_mass = compute_center_mass(self.swarm)

        # compute dispersion
        self.dispersion = compute_dispersion(self.swarm, self.center_mass)

    def generate(self, cycles=1):

        for _ in range(cycles):
            self.update_velocity()
            self.update_position()

    def plot(self):
        plt.scatter(self.swarm[:, 0], self.swarm[:, 1], s=1, color='black')
        plt.scatter(self.center_mass[:, 0], self.center_mass[:, 1], s=16, color='red', marker='s', alpha=0.5)
        plt.scatter(self.global_best[:, 0], self.global_best[:, 1], s=16, color='green', marker='+')
    
    def describe(self):
        print(f'best fit value = {self.fit(self.global_best)[0]}')
        print(f'best solution = {self.global_best[0]}')
        print(f'center of mass = {self.center_mass[0]}')
        print(f'dispersion = {self.dispersion}')



def original_update(n, swarm, velocity, global_best, personal_best, center_mass, dispersion, soc_factor=0.5015298256056491, cog_factor=2.302762690261078, inertia_weight=0.16394384037124668):

    # stochastic elements
    r1 = np.random.uniform(np.zeros(2), np.ones(2), size=(n, 2))
    r2 = np.random.uniform(np.zeros(2), np.ones(2), size=(n, 2))

    # best:
    '''soc_factor = 0.5015298256056491
    cog_factor = 2.302762690261078
    inertia_weight = 0.16394384037124668'''
    inertia_decay = 0.01
    inertia_min = 0
    
    # factors
    inertia = velocity * inertia_weight
    social = soc_factor * r1 * (global_best - swarm)
    cognitive = cog_factor * r2 * (personal_best - swarm)

    # update velocity
    velocity = inertia + social + cognitive
    return velocity

class SwarmGP:
    """Class swarm that implementes PSO for 2D problems"""

    def __init__(
        self,
        n: int,
        land: Landscape = None,
        vel_update: Callable[[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]=original_update,
        vel_magnitude_limits: list=[1e-10, 5],
        ):

        self.n = n # number of particles
        self.vel_update = vel_update # velocity update function
        self.land = None # fitness function

        self.vel_magnitude_limits = vel_magnitude_limits # min and max velocity magnitude

        # quantities to be initialized
        self.swarm = None
        self.velocity = None
        self.personal_best = None
        self.global_best = None
        self.center_mass = None
        self.dispersion = None
        # self.default_start() # initialize values

    def default_start(self, gen_position_limits=np.array([[-5, 5]]*2), land=None):
        
        """Function dealing with the initialization of the particles initial positions and velocities"""

        self.land = land

        # initialize population
        self.swarm = np.random.uniform(
            gen_position_limits[:, 0], 
            gen_position_limits[:, 1], 
            size=(self.n, 2))
        
        # initialize velocity
        self.velocity = np.zeros((self.n, 2))

        # other quantities
        self.personal_best = self.swarm
        self.global_best = np.array([self.personal_best[np.argmin(self.land(self.personal_best))]])
        self.center_mass = compute_center_mass(self.swarm)
        self.dispersion = compute_dispersion(self.swarm, self.center_mass) # average distance from center of mass
    
    def check_velocity(self):
        """Function that checks if the vecoloticies of each particles are between the fixed bounds. Otherwise normalizes them to align to the boundaries"""
        magnitude = compute_magnitude(self.velocity)
        large_vel = magnitude > self.vel_magnitude_limits[1]
        small_vel = (magnitude < self.vel_magnitude_limits[0]) & (magnitude > 0)
        if large_vel.sum() > 0:
            self.velocity[large_vel] = self.velocity[large_vel] / np.array([[magnitude[large_vel]]]).reshape(large_vel.sum(), 1) * self.vel_magnitude_limits[1]
        if small_vel.sum() > 0:
            self.velocity[small_vel] = self.velocity[small_vel] / np.array([[magnitude[small_vel]]]).reshape(small_vel.sum(), 1) * self.vel_magnitude_limits[0]

    def update_position(self):
        """Updates position without checking for boundaries"""
        self.swarm = self.swarm + self.velocity

        # update best
        indexes = self.land(self.personal_best) > self.land(self.swarm)
        self.personal_best[indexes] = self.swarm[indexes]
        self.global_best = np.array([self.personal_best[np.argmin(self.land(self.personal_best))]])

        # update center mass
        self.center_mass = compute_center_mass(self.swarm)

        # compute dispersion
        self.dispersion = compute_dispersion(self.swarm, self.center_mass)

    def generate(self, cycles=1):

        for _ in range(cycles):
            self.velocity = self.vel_update(self.n, self.swarm, self.velocity, self.global_best, self.personal_best, self.center_mass, self.dispersion)
            self.check_velocity()
            self.update_position()

    def plot(self):
        plt.scatter(self.swarm[:, 0], self.swarm[:, 1], s=1, color='black')
        plt.scatter(self.center_mass[:, 0], self.center_mass[:, 1], s=16, color='red', marker='s', alpha=0.5)
        plt.scatter(self.global_best[:, 0], self.global_best[:, 1], s=16, color='green', marker='+')
    
    def describe(self):
        print(f'best fit value = {self.land(self.global_best)[0]}')
        print(f'best solution = {self.global_best[0]}')
        print(f'center of mass = {self.center_mass[0]}')
        print(f'dispersion = {self.dispersion}')