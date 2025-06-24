from pymoo.algorithms.hyperparameters import SingleObjectiveSingleRun, HyperparameterProblem
from pymoo.algorithms.soo.nonconvex.g3pcx import G3PCX
from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.problem import Problem
from pymoo.core.parameters import set_params, hierarchical
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.single import Sphere
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("Hello World")

# This makes a problem which acts as a 3d porabola
#It is lowest at 2,2
class bowl(Problem):
    #This initilizes the problem and sets (in order):
    #The number of variables, the number of objectives, the number of constraints, and the boundries of the function
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array([-10, -10]),
                         xu=np.array([10, 10]))
    
    def _evaluate(self, X, out):
        f = (X[:, 0] - 2)**2 + (X[:, 1] - 2)**2
        out["F"] = f

problem = bowl
ackley = get_problem("ackley", n_var=2, a=20, b=1/5, c=2 * np.pi)

problem = ackley
F, G = problem.evaluate(np.random.rand(3, 10))