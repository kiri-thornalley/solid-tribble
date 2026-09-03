
from src.Bayesian_Optimisation.bo_core import *
from src.Bayesian_Optimisation.data_pipeline import materials_pipeline
from src.Bayesian_Optimisation.thompson_sampling import thompson_sampling
from src.Bayesian_Optimisation.random_search import random_search

import pandas as pd
import matplotlib.pyplot as plt
import warnings
from botorch.exceptions.warnings import InputDataWarning

""" turn off "Input features is not contained to the unit cube" warning. 
I know, deliberate choice to work in the native domain to simplify the code. """
warnings.filterwarnings("ignore", category=InputDataWarning)

def optimize_candidate(acquisition, candidate_X):

    acq_values = acquisition(candidate_X.unsqueeze(1))

    best_idx = acq_values.argmax().item()
    candidate = candidate_X[best_idx].unsqueeze(0)

    return candidate, best_idx



results = []

n_iter = 40

for seed in range(3):

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
        
        for iteration in range(n_iter):

            model = build_GP_surrogate(
                train_X,
                train_Y
            )
            if acq == "TS":

                candidate, candidate_idx = thompson_sampling(
                    model,
                    current_candidate_X
                )

            else:
                acquisition = build_acquisition(
                    model,
                    train_Y,
                    acq=acq
                )

                candidate, candidate_idx = optimize_candidate(
                    acquisition,
                    current_candidate_X
                )

            formation_energy = current_candidate_pool_df.iloc[
                candidate_idx
            ]["formation_energy"]

            observation = torch.tensor(
                [[-formation_energy]],
                dtype=torch.double
            )

            train_X, train_Y = update_dataset(
                train_X,
                train_Y,
                candidate,
                observation
            )

            # Remove selected material
            current_candidate_X = torch.cat([
                current_candidate_X[:candidate_idx],
                current_candidate_X[candidate_idx + 1:]
            ])

            current_candidate_pool_df = current_candidate_pool_df.drop(
                current_candidate_pool_df.index[candidate_idx]
            ).reset_index(drop=True)

            best_value = -train_Y.max().item()

            results.append({
                "seed": seed,
                "method": f"BO-{acq}",
                "iteration": iteration + 1,
                "best_value": best_value,
            })

            print(
                f"Iteration {iteration + 1}: "
                f"candidate index = {candidate_idx}, "
                f"formation energy = {formation_energy:.4f},"
                f"best value = {best_value:.4f}"
)
    df = pd.DataFrame(results)

    df.to_csv(
        "data/materials.csv",
        index=False
    )

# Plot
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