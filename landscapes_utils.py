import numpy as np
import matplotlib.pyplot as plt
import random
from collections.abc import Callable

# landscapes fucntions
def circle(pts: np.ndarray) -> np.ndarray:
    """
    Info:
    - global minimum = f(0, 0) = 0
    - local minima: no
    """
    x: np.ndarray = pts[:, 0]
    y: np.ndarray = pts[:, 1]
    return np.power(x, 2) + np.power(y, 2)

def rastrigin(pts: np.ndarray) -> np.ndarray:
    """
    Info:
    - global minimum = f(0, 0) = 0
    - local minima: yes
    """
    A: int = 10
    x: np.ndarray = pts[:, 0]
    y: np.ndarray = pts[:, 1]
    return A*2 + (np.power(pts, 2) - A*np.cos(pts * 2 * np.pi)).sum(axis=1)

# def eggholder(pts: np.ndarray) -> np.ndarray:
#     """
#     Info:
#     - global minimum = f(0, 0) = 0
#     - local minima: no
#     """
#     x: np.ndarray = pts[:, 0]
#     y: np.ndarray = pts[:, 1]
#     return - (y + 47)* np.sin(np.sqrt(np.abs(x* 0.5 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))

def schaffer(pts: np.ndarray) -> np.ndarray:
    x: np.ndarray = pts[:, 0]
    y: np.ndarray = pts[:, 1]
    return 0.5 + (np.power(
        np.sin(np.power(x, 2) - np.power(y, 2)) - 0.5, 2)
        ) / np.power(
            1 + 0.001*(np.power(x, 2) + np.power(y, 2)), 2
        )

def rosenbrock(pts: np.ndarray) -> np.ndarray:
    """
    Info:
    - global minimum = f(a, a**2) = 0
    - local minima: boh
    """
    a: int = 0
    b: int = 100
    x: np.ndarray = pts[:, 0]
    y: np.ndarray = pts[:, 1]
    return np.power((a - x), 2)  + b * np.power((y - np.power(x, 2)), 2)

def easom(pts: np.ndarray) -> np.ndarray:
    """
    Info:
    - global minimum = f(pi, pi) = -1
    - local minima: yes
    """
    x: np.ndarray = pts[:, 0] + np.pi
    y: np.ndarray = pts[:, 1] + np.pi
    return -np.cos(x)*np.cos(y)*np.exp(-np.power((x-np.pi), 2) - np.power((y - np.pi), 2))

def griewank(pts: np.ndarray) -> np.ndarray:
    """
    Info:
    - global minimum = f(0, 0) = -1
    - local minima: yes
    """
    x: np.ndarray = pts[:, 0]
    y: np.ndarray = pts[:, 1]
    return 1 + 1/4000 * (np.power(x, 2) + np.power(y, 2)) - np.cos(x) * np.cos(y / np.sqrt(2))


def generate_rotation_matrix(angle: float) -> np.ndarray:
    """Functions that given an angle returns the rotation matrix corresponding to that angle. Rotation is performed as vector @ rotation"""
    
    return np.array([[np.cos(angle), np.sin(angle)],
                     [-np.sin(angle), np.cos(angle)]])

# landscape class
class Landscape:
    """Class storing a landscape function, its minimum and name, allowing to perform translations and rotations"""

    def __init__(
        self, name: str, 
        func: Callable[[np.ndarray], np.ndarray], 
        minimum: np.ndarray=np.array([[0, 0]]),
        rotation: np.ndarray=np.eye(2, 2),
        ):
        self.name = name
        self.func = func
        # standard minima is [[0, 0]]
        self.minimum = minimum # np array of shape (1, 2)
        self.rotation = rotation # np array of type [[cosx, sinx], [-sinx, cosx]]

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        # rotation
        pts = pts - self.minimum
        pts = pts @ self.rotation
        
        return self.func(pts)
    

basic_landscapes = [
    Landscape('circle', circle, np.array([[0, 0]])),
    Landscape('rastrigin', rastrigin, np.array([[0, 0]])),
    Landscape('schaffer', schaffer, np.array([[0, 0]])),
    Landscape('rosenbrock', rosenbrock, np.array([[0, 0]])),
    Landscape('easom', easom, np.array([[0, 0]])),
    Landscape('griewank', griewank, np.array([[0, 0]]))
]

landscapes_dict = {
    'circle': circle,
    'rastrigin': rastrigin,
    'schaffer': schaffer,
    'rosenbrock': rosenbrock,
    'easom': easom,
    'griewank': griewank,
}

landscapes_names = [
    'circle',
    'rastrigin',
    'schaffer',
    'rosenbrock',
    'easom',
    'griewank',
]

CIRCLE = Landscape('circle', circle)
RASTRIGIN = Landscape('rastrigin', rastrigin)
SCHAFFER = Landscape('schaffer', schaffer)
ROSENBROCK = Landscape('rosenbrock', rosenbrock)
EASOM = Landscape('easom', easom)
GRIEWANK = Landscape('griewank', griewank)


# OLD VERSION

# # fitness function modifier

# def fitness_decorator(fit, var_translation, var_scale=1, amplitude=1, intercept=0):
#     """var_translation and var_scale have (2,) shape"""
#     def modified_fit(solutions):
#         solutions = solutions - var_translation
#         solutions = solutions * var_scale

#         return amplitude * fit(solutions) + intercept
    
#     return modified_fit

# # landscape and fitness functions

# class Landscape:

#     def __init__(self, name, func, minima):
#         self.name = name
#         self.func = func
#         self.minima = minima
    
#     def modify_minima(self, new_minima):
#         self.func = fitness_decorator(self.func, new_minima[0] - self.minima[0])
#         self.minima = new_minima