import numpy as np
import matplotlib.pyplot as plt
import random
from collections.abc import Callable
from landscapes import *


def compute_magnitude(pts):
    """Return a vector of magnitudes of an array of points in R^2"""
    # type consistency
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

def original_update(swarm, velocity, global_best, personal_best, center_mass, dispersion, soc_factor=1.49, cog_factor=1.49, inertia_weight=0.9):
    """Returns the bnew velocity vectors for the particle using the standard velocity update formula"""

    # best parameters found via hyperparemeter optimization:
    # soc_factor = 0.5015298256056491
    # cog_factor = 2.302762690261078
    # inertia_weight = 0.16394384037124668

    n = 50

    # stochastic elements
    r1 = np.random.uniform(np.zeros(2), np.ones(2), size=(n, 2))
    r2 = np.random.uniform(np.zeros(2), np.ones(2), size=(n, 2))
    
    # factors
    inertia = velocity * inertia_weight
    social = soc_factor * r1 * (global_best - swarm)
    cognitive = cog_factor * r2 * (personal_best - swarm)

    # update velocity
    velocity = inertia + social + cognitive
    return velocity

class SwarmGP:
    """Class swarm that implementes PSO for 2D problems with modular velocity update"""

    def __init__(
        self,
        n: int,
        land: Landscape = None,
        vel_update: Callable[[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]=original_update,
        vel_magnitude_limits: list=[1e-10, 5],
        ):

        self.n = n # num particles
        self.vel_update = vel_update # velocity update function
        self.land = None # fitness function

        self.vel_magnitude_limits = vel_magnitude_limits # min and max velocity magnitude

        # quantities to be initialized
        self.swarm = None
        self.velocity = None
        self.personal_best = None
        self.global_best = None
        self.center_mass = None
        self.dispersion = None # average distance from center of mass

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
        self.dispersion = compute_dispersion(self.swarm, self.center_mass)
    
    def check_velocity(self):
        """Function that checks if the velocity of each particles is between the fixed bounds. 
        Otherwise normalizes them to align to the boundaries"""

        magnitude = compute_magnitude(self.velocity)
        large_vel = magnitude > self.vel_magnitude_limits[1]
        small_vel = (magnitude < self.vel_magnitude_limits[0]) & (magnitude > 0)
        if large_vel.sum() > 0:
            self.velocity[large_vel] = self.velocity[large_vel] / np.array([[magnitude[large_vel]]]).reshape(large_vel.sum(), 1) * self.vel_magnitude_limits[1]
        if small_vel.sum() > 0:
            self.velocity[small_vel] = self.velocity[small_vel] / np.array([[magnitude[small_vel]]]).reshape(small_vel.sum(), 1) * self.vel_magnitude_limits[0]

    def update_position(self):
        """Updates position with invisible oundary conditions"""

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
        """Runs the PSO cycles for the specified number of times"""

        for _ in range(cycles):
            self.velocity = self.vel_update(self.swarm, self.velocity, self.global_best, self.personal_best, self.center_mass, self.dispersion)
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

def generate_problems(NUM_PROBLEMS):
    """Stochastically generates a sequence of landscapes/problems for testing the performance of PSO"""
    
    problems = []
    new_minima = np.random.uniform(-2, 2, size=(NUM_PROBLEMS, 2))
    angles = np.random.uniform(0, 2, (NUM_PROBLEMS,))
    for i in range(NUM_PROBLEMS):
        name = random.choice(landscapes_names)
        land = Landscape(name, landscapes_dict[name])
        land.minimum = new_minima[i]
        angle = angles[i]
        land.rotation = generate_rotation_matrix(angle*np.pi)
        problems.append(land)
    return problems

def PSOResults(swarm: SwarmGP, problems, num_iterations):
    """Returns an array of the best result obtained by the PSO on the problems contained in the problems list"""

    n_problems = len(problems)
    
    results = np.zeros(n_problems)
    swarms_best = np.zeros((len(problems), 2))
    problems_minima = np.zeros((n_problems, 2))

    for i in range(n_problems):
        problems_minima = problems[i].minimum
        swarm.default_start(land=problems[i])
        swarm.generate(num_iterations)
        swarms_best[i] = swarm.global_best

    # Process results
    results = distance(swarms_best, problems_minima)

    return results