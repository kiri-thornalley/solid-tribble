from src.Bayesian_Optimisation.bo_core import *
from experiments.random_search import random_search
from plotting import *
import pandas as pd
from botorch.test_functions import Branin, Ackley, Hartmann


import warnings
from botorch.exceptions.warnings import InputDataWarning

""" turn off "Input features is not contained to the unit cube" warning. 
I know, deliberate choice to work in the native domain to simplify the code. """
warnings.filterwarnings("ignore", category=InputDataWarning)

OBJECTIVES = {
    "branin": Branin(negate=True).to(device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), dtype=torch.double),
    "ackley": Ackley(negate=True).to(device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), dtype=torch.double),
    "hartmann": Hartmann(negate=True).to(device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"), dtype=torch.double)
}


results = []

n_iter=20

for seed in range(10):
    for objective_name in ["branin", "ackley", "hartmann"]:
            objective = OBJECTIVES[objective_name]
            print(f"Running experiment for {objective_name}, method: Random, with seed {seed}")
            random_history = random_search(n_iter=n_iter, objective=objective, seed=seed)
            for iteration, value in enumerate(random_history, start=1):
                results.append({
                    "seed": seed,
                    "objective": objective_name,
                    "method": "random",
                    "iteration": iteration,
                    "best_value": value,
                })
            for acq in ["EI", "UCB"]:
                print(f"Running experiment for {objective_name}, method: BO-{acq}, with seed {seed}")
                bo_history = bayesian_optimization(n_iter=n_iter, objective=objective, acq=acq, seed=seed)
                for iteration, value in enumerate(bo_history, start=1):
                    results.append({
                        "seed": seed,
                        "objective": objective_name,
                        "method": f"BO-{acq}",
                        "iteration": iteration,
                        "best_value": value,
                    })

df = pd.DataFrame(results)
df.to_csv("results/benchmark.csv", index=False)
df = pd.read_csv("results/benchmark.csv")


convergence_regret_plot("results/benchmark.csv")
