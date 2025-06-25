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
from matplotlib.lines import Line2D
import time

def get_metrics(F, true_front):
    ref_point = np.array([1.1] * F.shape[1])
    hv = HV(ref_point=ref_point).do(F)
    igd = IGD(true_front).do(F)
    return hv, igd

def generate_true_pareto_surface(res=50):
    phi = np.linspace(0, np.pi / 2, res)
    theta = np.linspace(0, np.pi / 2, res)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z

def run_nsga2_and_plot(ax, eta_crossover):
    problem = get_problem("dtlz2", n_var=11, n_obj=3)

    algorithm = NSGA2(
        pop_size=100,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=eta_crossover),
        mutation=PolynomialMutation(prob=0.1, eta=20),
        eliminate_duplicates=True
    )

    start = time.time()
    res = minimize(problem, algorithm, ("n_gen", 200), seed=5, verbose=False)
    elapsed = time.time() - start

    F = res.F
    true_x, true_y, true_z = generate_true_pareto_surface()
    hv, igd = get_metrics(F, np.column_stack([true_x.ravel(), true_y.ravel(), true_z.ravel()]))

    ax.plot_surface(true_x, true_y, true_z, alpha=0.15, color='red', edgecolor='none')
    ax.scatter(F[:, 0], F[:, 1], F[:, 2], c='blue', s=20)

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.view_init(elev=30, azim=45)

    metrics_str = f"η_c: {eta_crossover} | HV: {hv:.3f} | IGD: {igd:.3f} | N: 100\nTime: {elapsed:.2f}s"
    return hv, ax, metrics_str

# --- Main Figure with Subplots ---
fig = plt.figure(figsize=(18, 10))
axes = [fig.add_subplot(2, 3, i + 1, projection='3d') for i in range(6)]
eta_values = [5, 10, 25, 50, 75, 100]

results = []

for ax, eta in zip(axes, eta_values):
    hv, ax_obj, label = run_nsga2_and_plot(ax, eta_crossover=eta)
    results.append((hv, ax_obj, label))

# --- Highlight Best HV ---
best_result = max(results, key=lambda x: x[0])
for hv, ax_obj, label in results:
    if ax_obj == best_result[1]:
        ax_obj.set_title(label, fontsize=10, fontweight='bold', color='darkgreen')
    else:
        ax_obj.set_title(label, fontsize=9)

# --- Legend ---
custom_legend = [
    Line2D([0], [0], marker='s', color='w', label='True Pareto Surface', markerfacecolor='red', markersize=10, alpha=0.3),
    Line2D([0], [0], marker='o', color='w', label='NSGA-II Result', markerfacecolor='blue', markersize=8)
]
fig.legend(handles=custom_legend, loc='upper center', ncol=2, fontsize=12)

plt.subplots_adjust(top=0.90)
plt.suptitle("DTLZ2 - NSGA-II Across Different η_c (Best HV Highlighted)", fontsize=16)
plt.tight_layout()
plt.show()
