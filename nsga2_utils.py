import numpy as np
import time
import matplotlib.pyplot as plt
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD


def generate_true_front(problem_name="dtlz2", n_var=11, n_obj=3):
    """
    Generate true Pareto front points for a given problem.
    Works for problems like DTLZ2, DTLZ7, etc.
    """
    problem = get_problem(problem_name, n_var=n_var, n_obj=n_obj)
    return problem.pareto_front()


def get_metrics(F, true_front):
    """Calculate HV and IGD for a given front."""
    ref_point = np.array([1.1] * F.shape[1])
    hv = HV(ref_point=ref_point).do(F)
    print(str(hv))
    igd = IGD(true_front).do(F)
    return hv, igd


def run_nsga2(problem_name="DTLZ2", n_var=11, n_obj=3, pop_size=100, eta_crossover=15, eta_mutation=20, prob_mutation=0.1, seed=1, n_gen=200):
    """Run NSGA-II with given parameters and return results."""
    problem = get_problem(problem_name, n_var=n_var, n_obj=n_obj)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=eta_crossover),
        mutation=PolynomialMutation(prob=prob_mutation, eta=eta_mutation),
        eliminate_duplicates=True
    )

    start = time.time()
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    elapsed = time.time() - start
    return res.F, elapsed


def plot_nsga2_result(ax, F, true_front, title_info="", highlight=False):
    """
    Plot NSGA-II result and true Pareto front (as points or surface).
    """
    # Plot true Pareto front as points instead of surface
    ax.scatter(true_front[:, 0], true_front[:, 1], true_front[:, 2], 
               c='red', s=5, alpha=0.2, label='True Front')

    ax.scatter(F[:, 0], F[:, 1], F[:, 2], c='blue', s=20, label='NSGA-II Result')

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.view_init(elev=30, azim=45)

    ax.set_title(title_info, fontsize=10,
                 fontweight='bold' if highlight else 'normal',
                 color='darkgreen' if highlight else 'black')
