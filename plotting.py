# code to create automated plots go here.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def convergence_regret_plot(csv_file, benchmark=None, acq=None, optimal_value=None):
    """
    Create a convergence and/or regret plot.
    Parameter(s):
        csv_file (str): Path to the CSV file containing the data.
        benchmark (str): The name of the benchmark function.
        acq (str): The name of the acquisition function.
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
    sns.set_theme(context="paper", style="ticks", palette="viridis")
    palette = sns.color_palette()

    OPTIMAL_VALUES = {
                "hartmann": +3.322,
                "branin": -0.397,
                "ackley": 0.0
            }
    df = pd.read_csv(csv_file)  

    benchmarks = df["objective"].unique()
    acqs = df["method"].unique()
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(15, 5))

    for col, benchmark in enumerate(benchmarks):
        ax = axes[col]
        for acq in acqs: 
            data = df[
                (df["objective"] == benchmark) & (df["method"] == acq)
            ]
            
            optimal_value = OPTIMAL_VALUES[benchmark]
            
            summary = (data.groupby("iteration")["best_value"].agg(median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            ) )
            
            ax.plot(summary.index, summary["median"], label=acq)
            ax.fill_between(
                summary.index,
                summary["q25"],
                summary["q75"],
                alpha=0.3
            )
        ax.axhline(y=optimal_value, color=palette[5], linestyle="--", label="Optimal Value")
        ax.set_title(f"Convergence Plot for {benchmark.capitalize()}")
        ax.set_xlabel("Number of observations")
        ax.set_ylabel("Best Observation")
        ax.set_xlim(1, data['iteration'].max())
        if col == len(benchmarks) - 1:
            ax.legend()

    plt.tight_layout()
    plt.savefig("results/convergence_plot.png")

    ## == Regret plot == ##
    """Measures the difference between the optimum value and the best observed value found so far.
    Simple regret tends towards 0 over time.
    """
    benchmarks = df["objective"].unique()
    acqs = df["method"].unique()
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(15, 5))

    for col, benchmark in enumerate(benchmarks):
        ax = axes[col]
        for acq in acqs: 
            data = df[
                (df["objective"] == benchmark) & (df["method"] == acq)
            ]
            
            optimal_value = OPTIMAL_VALUES[benchmark]
            
            # Calculate regret at each step
            data = data.copy()
            data["regret"] = optimal_value - data["best_value"]

            summary = (
                data
                .groupby("iteration")["regret"]
                .agg(
                    median="median",
                    q25=lambda x: x.quantile(0.25),
                    q75=lambda x: x.quantile(0.75),
                )
            )
            
            ax.plot(summary.index, summary["median"], label=acq)
            ax.fill_between(
                summary.index,
                summary["q25"],
                summary["q75"],
                alpha=0.3
            )
        ax.axhline(y=0, color="black", linestyle="--")
        ax.set_title(f"Regret Plot for {benchmark.capitalize()}")
        ax.set_xlabel("Number of observations")
        ax.set_ylabel("Regret")
        ax.set_xlim(1, data['iteration'].max())
        ax.set_ylim(0, data["regret"].max())
        if col == len(benchmarks) - 1:
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"results/regret_plot.png")

    
def random_vs_BO(best_history, best_value, objective):
    """
    Plot the comparison between random search and Bayesian optimisation.
    Parameter(s):
        best_history (list): A list of the best values found at each iteration during random search.
        best_value (list): A list of the best values found at each iteration during Bayesian optimisation.
        objective (str): The name of the objective function.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(best_history, label="Random Search")
    ax.plot(best_value, label="Bayesian Optimisation")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best value so far")
    ax.set_title(f"Random Search vs BO on {objective if objective is not None else 'Unknown'}")

    ax.legend()

    plt.savefig(f"results/comparison_{objective if objective is not None else 'Unknown'}.png")
