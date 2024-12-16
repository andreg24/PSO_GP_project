import operator
import math
import random

import numpy as np
import matplotlib.pyplot as plt

import sys
import gc

import optuna
import logging
import pickle

from collections.abc import Callable

from pso_utils import *
from landscapes_utils import *

from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


# functions to save and load results
def save_log(log, path='logs/log_uniform.pkl'):
    with open(path, 'wb') as lb_file:
        pickle.dump(log, lb_file)

def load_log(path='logs/log_uniform.pkl'):
    with open(path, 'rb') as lb_file:
        return pickle.load(lb_file)

def save_hof(hof, path='logs/hof_uniform.pkl'):
    with open(path, 'wb') as lb_file:
        pickle.dump(hof, lb_file)

def load_hof(path='logs/hof_uniform.pkl'):
    with open(path, 'rb') as lb_file:
        return pickle.load(lb_file)
    
def save_pop(pop, path='logs/hof_uniform.pkl'):
    with open(path, 'wb') as lb_file:
        pickle.dump(pop, lb_file)

def load_pop(path='logs/hof_uniform.pkl'):
    with open(path, 'rb') as lb_file:
        return pickle.load(lb_file)


# plotting function for results
def plot_log(log, ax1=None):
    gen = log.select("gen")
    fit_mins = log.chapters["fitness"].select("min")
    size_avgs = log.chapters["height"].select("avg")

    # If no ax1 is provided, create a new figure and axis
    if ax1 is None:
        fig, ax1 = plt.subplots()
    else:
        fig = ax1.figure

    # Plot minimum fitness on ax1
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    ax1.axhline(y=2.21, color="black", linestyle=":")
    ax1.axhline(y=4.66, color="black", linestyle=":")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    # Create a twin axis for height on the right
    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "r-", label="Average Height")
    ax2.set_ylabel("Height", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    # Combine legends from both axes
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    return fig, (ax1, ax2)


