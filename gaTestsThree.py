import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


# --- Compute Performance Metrics ---
def get_metrics(F, true_front):
    ref_point = np.array([1.1] * F.shape[1])
    hv = HV(ref_point=ref_point).do(F)
    igd = IGD(true_front).do(F)
    return hv, igd


# --- Generate Pareto Surface for DTLZ2 ---
def generate_true_pareto_surface(res=50):
    phi = np.linspace(0, np.pi / 2, res)
    theta = np.linspace(0, np.pi / 2, res)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z


# --- NSGA-II Run and Plot with Traits ---
def run_nsga2_and_plot(ax, eta_crossover, true_front):
    problem = get_problem("dtlz2", n_var=11, n_obj=3)

    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=eta_crossover),
        mutation=PolynomialMutation(prob=0.1, eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem, algorithm, ("n_gen", 200), seed=1, verbose=False)
    F = res.F

    # Metrics
    hv, igd = get_metrics(F, true_front)
    n_points = len(F)

    # Surface
    x, y, z = generate_true_pareto_surface()
    ax.plot_surface(x, y, z, alpha=0.15, color='red', edgecolor='none')

    # Solutions
    ax.scatter(F[:, 0], F[:, 1], F[:, 2], c='blue', s=20)

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.view_init(elev=30, azim=45)

    ax.set_title(f"η_c: {eta_crossover} | HV: {hv:.3f} | IGD: {igd:.3f} | N: {n_points}", fontsize=9)


# --- Setup Figure ---
fig = plt.figure(figsize=(18, 10))
axes = [fig.add_subplot(2, 3, i + 1, projection='3d') for i in range(6)]

# --- Get True Front Once ---
problem = get_problem("dtlz2", n_var=11, n_obj=3)
true_front = problem.pareto_front(use_cache=False)

# --- Vary eta_crossover ---
eta_vals = [5, 10, 15, 20, 25, 30]
for ax, eta in zip(axes, eta_vals):
    run_nsga2_and_plot(ax, eta, true_front)

# --- Add Legend ---
custom_legend = [
    Line2D([0], [0], marker='s', color='w', label='True Pareto Surface', markerfacecolor='red', markersize=10, alpha=0.3),
    Line2D([0], [0], marker='o', color='w', label='NSGA-II Result', markerfacecolor='blue', markersize=8)
]
fig.legend(handles=custom_legend, loc='upper center', ncol=2, fontsize=12)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.suptitle("DTLZ2 - NSGA-II Results with Varying Crossover η (eta_crossover)", fontsize=16)
plt.show()
