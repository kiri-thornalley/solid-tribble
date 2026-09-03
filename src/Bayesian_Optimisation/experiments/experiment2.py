
from src.Bayesian_Optimisation.bo_core import *
from src.Bayesian_Optimisation.data_pipeline import materials_pipeline
from src.Bayesian_Optimisation.thompson_sampling import thompson_sampling
from src.Bayesian_Optimisation.random_search import random_search

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from botorch.exceptions.warnings import InputDataWarning

""" turn off "Input features is not contained to the unit cube" warning. 
I know, deliberate choice to work in the native domain to simplify the code. """
warnings.filterwarnings("ignore", category=InputDataWarning)

results = []

n_iter = 20

for seed in range(10):

    # Create data for this seed
    init_X, init_Y, candidate_X, candidate_pool_df = materials_pipeline(
        seed=seed
    )

    for acq in ["EI", "UCB", "TS"]:

        # Fresh training data for this acquisition function
        train_X = init_X.clone()
        train_Y = init_Y.clone()

        # Fresh candidate pool
        current_candidate_X = candidate_X.clone()
        current_candidate_pool_df = candidate_pool_df.copy()

        # Check data before fitting GP
        print(
            f"Seed {seed}, {acq}: "
            f"NaNs in X = {torch.isnan(train_X).sum().item()}, "
            f"NaNs in Y = {torch.isnan(train_Y).sum().item()}"
        )
        
        history = bayesian_optimization(n_iter, acq, seed, objective=None, generate=False, train_X=train_X, train_Y=train_Y, current_candidate_X=current_candidate_X, current_candidate_pool_df=current_candidate_pool_df)
        for iteration, value in enumerate(history, start=1):
        
            results.append({
                "seed": seed,
                "method": f"BO-{acq}",
                "iteration": iteration,
                "best_value": value,
            })

    df = pd.DataFrame(results)

    df.to_csv(
        "data/materials.csv",
        index=False
    )

# Plot
sns.set_theme(context="paper", style="ticks", palette="viridis")
fig, ax = plt.subplots(figsize=(8, 6))

for method in df["method"].unique():

    method_df = df[df["method"] == method]

    summary = (
        method_df
        .groupby("iteration")["best_value"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
    )

    ax.plot(
        summary.index,
        summary["median"],
        label=method
    )

    ax.fill_between(
        summary.index,
        summary["q25"],
        summary["q75"],
        alpha=0.3
    )


ax.set_xlabel("BO iterations")
ax.set_ylabel("Best formation energy")
ax.legend()

plt.tight_layout()

plt.savefig(
    "results/materials_convergence_plot.png"
)

plt.close()