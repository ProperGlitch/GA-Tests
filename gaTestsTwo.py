import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output

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

#This gets the ackley function that already exists within pymoo
#This is lowest at 0,0
ackley = get_problem("ackley", n_var=2, a=20, b=1/5, c=2 * np.pi)
rastrigin = get_problem("rastrigin", n_var=2)
eggholder = get_problem("Griewank")
#This gets the NSGA2 algorithm and sets it as the algorithm to be used in minimzation
algorithm = NSGA2(pop_size=20)

problem = eggholder

#plot_problem_surface(problem, 100, plot_type="wireframe+contour")

class MyOutput(Output):

    def __init__(self):
        super().__init__()
        self.found_x = Column("found_x", width=13)
        self.found_y = Column("found_y", width=13)
        self.columns += [self.found_x, self.found_y]

    def update(self, algorithm):
        super().update(algorithm)
        bestValue = algorithm.pop.get("X")[np.argmin(algorithm.pop.get("F"))]
        self.found_x.set(bestValue[0])
        self.found_y.set(bestValue[1])


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
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x1, x2)")

for i in range(1):
    res = minimize(problem,
                algorithm,
                ('n_gen', 100),
                seed=i+1,
                output = MyOutput(),
                verbose=True,
                )

    print("X: " + str(res.X[0]) + " | Y: " + str(res.X[1]))

# Target (x, y) location where you want to add a point
x_target = res.X[0]
y_target = res.X[1]

# Find the index of the closest point in the grid
ix = (np.abs(x - x_target)).argmin()
iy = (np.abs(y - y_target)).argmin()

# Get the exact x, y, z from the grid
x_closest = x[ix]
y_closest = y[iy]
z_value = Z[iy, ix]  # Note: rows = y, columns = x in meshgrid

# Plot the point
ax.scatter(x_target, y_target, z_value, color='red', s=50, label='Target Point', depthshade = False, zorder = 10)

# Optional: label the point
ax.text(x_target, y_target, z_value + 0.5, f'({x_target:.2f}, {y_target:.2f}, {z_value:.2f})', color='red')

# Show legend and plot
ax.legend()
plt.show()