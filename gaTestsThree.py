import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import time
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.selection.tournament import TournamentSelection


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


#Sets the algorithim and problem
def roll(prob = 0):

    problems = {
    0: get_problem("ackley", n_var=2, a=20, b=1/5, c=2 * np.pi),
    1: get_problem("rastrigin", n_var=2),
    2: get_problem("griewank", n_var=2),
    3: EggHolderProblem(),
    4: bowl(),
    5: get_problem("dtlz2", n_var=11, n_obj=3)
    }
    problem = problems[prob]
    algorithm = NSGA2(
        pop_size=100,
        crossover=PointCrossover(n_points=1),
        mutation=PolynomialMutation(eta=15),
        eliminate_duplicates=True
        )
    return minimize(problem, algorithm, ('n_gen', 100), verbose=False,), problem

#---

start = time.time()

fig = plt.figure(figsize=(18, 10))  # Optional: widen a bit to fit 6 plots better

for i in range(6):  # Now includes DTLZ2
    if i < 5:
        res, problem = roll(i)

        x = np.linspace(problem.xl[0], problem.xu[0], 100)
        y = np.linspace(problem.xl[1], problem.xu[1], 100)
        X, Y = np.meshgrid(x, y)
        XY = np.column_stack([X.ravel(), Y.ravel()])
        Z = np.array([problem.evaluate(xi) for xi in XY]).reshape(X.shape)

        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.33)

        # Extract solution point
        if isinstance(res.X[0], np.float64):
            x_target = res.X[0]
            y_target = res.X[1]
        else:
            x_target = res.X[0][0]
            y_target = res.X[0][1]

        ix = (np.abs(x - x_target)).argmin()
        iy = (np.abs(y - y_target)).argmin()
        z_value = Z[iy, ix]

        # Plot and label point
        ax.scatter(x_target, y_target, z_value, color='red', s=30, label='Best', depthshade=False, zorder=10)
        ax.text(x_target, y_target, z_value + 5, f'({x_target:.2f}, {y_target:.2f}, {z_value:.2f})',
                color='black', fontsize=8, ha='center')

        titles = [
            "0, 0",
            "0, 0",
            "0, 0",
            "512, 404.2319",
            "2,2"
        ]
        ax.set_title(f"Ideal: {titles[i]}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")

    else:
        # ✅ Special case: DTLZ2 (multi-objective)
        res, problem = roll(5)
        F = res.F  # shape (n_solutions, 3)

        ax = fig.add_subplot(2, 3, 6, projection='3d')
        ax.scatter(F[:, 0], F[:, 1], F[:, 2], c='blue', s=15, alpha=0.7)
        ax.set_title("Pareto Front - DTLZ2")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        ax.view_init(elev=30, azim=45)


end = time.time()  # ✅ End the timer
print(f"\nTotal time to generate all plots: {end - start:.2f} seconds")

plt.tight_layout()
plt.show()


