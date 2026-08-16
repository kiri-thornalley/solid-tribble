# code to create automated plots go here.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="paper", style="ticks", palette="viridis")

def convergence_regret_plot(best_observations, benchmark, acq, optimal_value=None):
    """
    Create a convergence and/or regret plot.
    """
    OPTIMAL_VALUES = {
        "hartmann": +3.322,
        "branin": -0.397,
        "ackley": 0.0
    }
    if benchmark is not None:
        optimal_value = OPTIMAL_VALUES[benchmark]
    elif benchmark is None:
        if optimal_value is None:
            raise ValueError("Optimal value not specified")
    
    print(f"Optimal Value {optimal_value}")

    ## == Convergence plot == ##
    """
    Best observed value of the objective function found so far, against iteration number. 
    Demonstrates how fast the optimizer approaches the maximum value. 
    """
    iters = np.arange(len(best_observations))

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(iters, best_observations, linewidth=1.5, label=acq)
    ax.axhline(y=optimal_value, linewidth=1.5, label="Optimal Value")
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Best Observation")
    ax.set_xlim(0, len(best_observations) - 1)
    ax.legend()
    plt.savefig(f"results/{benchmark}_convergence_plot.png")

    ## == Regret plot == ##
    """
    Measures the difference between the optimum value and the best observed value found so far.
    Simple regret tends towards 0 over time.
    """
    # Calculate regret at each step
    global_maximum = optimal_value
    regret = global_maximum - np.asarray(best_observations)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(iters, regret, linewidth=1.5, label=acq)
    ax.set_xlabel("Number of observations")
    ax.set_ylabel("Regret")
    ax.set_xlim(0, len(best_observations) - 1)
    ax.legend()
    plt.savefig(f"results/{benchmark}_regret_plot.png")