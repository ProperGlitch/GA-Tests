import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from nsga2_utils import run_nsga2, get_metrics, plot_nsga2_result, generate_true_front
import numpy as np

# Define eta_crossover values to test
eta_values = [15, 30, 45, 60, 75, 90]

# Generate true Pareto surface once for all plots
true_front_points = generate_true_front()

# Prepare figure
fig = plt.figure(figsize=(18, 10))
axes = [fig.add_subplot(2, 3, i + 1, projection='3d') for i in range(6)]

# Store results to find best
results = []

# Run study
for ax, eta in zip(axes, eta_values):
    F, duration = run_nsga2(eta_crossover=eta, seed=6)
    hv, igd = get_metrics(F, true_front_points)
    label = f"η_c: {eta} | HV: {hv:.3f} | IGD: {igd:.3f}\nTime: {duration:.2f}s"
    results.append((hv, ax, label, F))

# Highlight best HV
best_result = max(results, key=lambda x: x[0])

# Plot all
for hv, ax, label, F in results:
    plot_nsga2_result(ax, F, generate_true_front(), label, highlight=(ax == best_result[1]))

# Legend
custom_legend = [
    Line2D([0], [0], marker='s', color='w', label='True Pareto Surface', markerfacecolor='red', markersize=10, alpha=0.3),
    Line2D([0], [0], marker='o', color='w', label='NSGA-II Result', markerfacecolor='blue', markersize=8)
]
fig.legend(handles=custom_legend, loc='upper center', ncol=2, fontsize=12)

# Title
plt.subplots_adjust(top=0.90)
plt.suptitle("DTLZ2 - NSGA-II with Varying η_c (Crossover Distribution Index)", fontsize=16)
plt.tight_layout()
plt.show()