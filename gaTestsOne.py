import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

#Best at 2,2
class bowl(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array([-10, -10]),
                         xu=np.array([10, 10]))
    
    def _evaluate(self, X, out):
        f = (X[:, 0] - 2)**2 + (X[:, 1] - 2)**2
        out["F"] = f

class EggHolderProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array([-512, -512]),
                         xu=np.array([512, 512]),
                         )

    def _evaluate(self, X, out, *args, **kwargs):
        x1 = X[:, 0]
        x2 = X[:, 1]
        term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47))))
        term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
        f = term1 + term2
        out["F"] = f.reshape(-1, 1)   # (n_individuals, 1)

#Best at 0,0
ackley = get_problem("ackley", n_var=2, a=20, b=1/5, c=2 * np.pi)

#Best at 0,0
rastrigin = get_problem("rastrigin", n_var=2)

#Best at 0,0
griewank = get_problem("griewank")

#Best at 512, 404.2319
eggholder = EggHolderProblem()

#Best at 2,2
sheet = bowl()


#Sets the algorithim and problem
def roll():

    algorithm = NSGA2(pop_size=100)
    problem = eggholder
    return minimize(problem, algorithm, ('n_gen', 100), verbose=False,), problem

def map(inp):

    
    res, problem = inp

    x = np.linspace(problem.xl[0], problem.xu[0], 100)
    y = np.linspace(problem.xl[1], problem.xu[1], 100)
    X, Y = np.meshgrid(x, y)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([problem.evaluate(xi) for xi in XY]).reshape(X.shape)
    XY = np.column_stack([X.ravel(), Y.ravel()])
    Z = np.array([problem.evaluate(xi) for xi in XY]).reshape(X.shape)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha = 0.5)
    ax.set_title("3D Plot of Single-Objective Function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Target (x, y) location where you want to add a point
    x_target = res.X[0]
    y_target = res.X[1]

    # This finds the point with the closest x and y to our target x and y, and then sets it's z as the z for our point
    ix = (np.abs(x - x_target)).argmin()
    iy = (np.abs(y - y_target)).argmin()
    z_value = Z[iy, ix]

    ax.scatter(x_target, y_target, z_value, color='red', s=50, label='Target Point', depthshade = False, zorder = 10)
    ax.text(x_target, y_target, z_value + 0.5, f'({x_target:.2f}, {y_target:.2f}, {z_value:.2f})', color='red')

    ax.legend()
    plt.show()

map(roll())