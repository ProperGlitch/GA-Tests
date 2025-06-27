import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from nsga2_utils import run_nsga2, get_metrics, plot_nsga2_result, generate_true_front
import numpy as np


# Define mutation probabilities to test
generations = [0, 0.1, 0.3, 0.6, 0.7, 1.0]
problem_name = "DTLZ7"


# Generate true Pareto surface once
true_front_points = generate_true_front(problem_name)

# Prepare figure
fig = plt.figure(figsize=(18, 10))
axes = [fig.add_subplot(2, 3, i + 1, projection='3d') for i in range(6)]

# Store results for comparison
results = []

# Run study
for ax, p_mut in zip(axes, generations):
    F, duration = run_nsga2(prob_mutation=p_mut, problem_name="DTLZ7", seed=42, n_gen=3)
    hv, igd = get_metrics(F, true_front_points)
    label = f"p_mut: {p_mut:.1f} | HV: {hv:.3f} | IGD: {igd:.3f}\nTime: {duration:.2f}s"
    results.append((hv, ax, label, F))

# Highlight best HV
best_result = max(results, key=lambda x: x[0])

# Plot
for hv, ax, label, F in results:
    plot_nsga2_result(ax, F, true_front_points, label, highlight=(ax == best_result[1]))

# Legend
custom_legend = [
    Line2D([0], [0], marker='s', color='w', label='True Pareto Surface', markerfacecolor='red', markersize=10, alpha=0.3),
    Line2D([0], [0], marker='o', color='w', label='NSGA-II Result', markerfacecolor='blue', markersize=8)
]
fig.legend(handles=custom_legend, loc='upper center', ncol=2, fontsize=12)

# Title and layout
plt.subplots_adjust(top=0.90)
plt.suptitle(f"{problem_name} - NSGA-II with Varying Number of Generations (n_gen)", fontsize=16)
plt.tight_layout()
plt.show()