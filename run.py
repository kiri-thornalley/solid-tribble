import argparse

import torch
from plotting import convergence_regret_plot
from src.Bayesian_Optimisation.bo_core import (
    build_GP_surrogate, build_acquisition,
    optimize_candidate, evaluate_candidate,
    update_dataset,
    generate_initial_data
    )
from experiments.random_search import random_search, plot
from botorch.test_functions import Hartmann, Ackley, Branin

import warnings
from botorch.exceptions.warnings import InputDataWarning

torch.manual_seed(42)

""" turn off "Input features is not contained to the unit cube" warning. 
I know, deliberate choice to work in the native domain to simplify the code. """
warnings.filterwarnings("ignore", category=InputDataWarning)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

if argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", choices=["hartmann", "ackley", "branin"], default="hartmann")
    parser.add_argument("--acq", choices=["EI", "UCB"], default="EI")
    args = parser.parse_args()

if args.function == "hartmann":
    hartmann = Hartmann(negate=True).to(device=device)
    objective = hartmann
elif args.function == "ackley":
    ackley = Ackley(negate=True).to(device=device)
    objective = ackley
elif args.function == "branin":
    branin = Branin(negate=True).to(device=device)
    objective = branin

acq = args.acq
bounds = objective.bounds.to(device=device, dtype=dtype)

# Generate initial observations
train_X, train_Y = generate_initial_data(n=10, objective=objective, bounds=bounds)

if __name__ == "__main__":
    history = []
    n_iter = 20
    # Run BO
    for iteration in range(n_iter):
        model = build_GP_surrogate(train_X,train_Y)
        acquisition = build_acquisition(model, train_Y, acq=acq)
        candidate = optimize_candidate(acquisition, bounds)
        observation = evaluate_candidate(candidate, objective)
        train_X, train_Y = update_dataset(train_X, train_Y, candidate, observation)

        best_value = train_Y.max()

        print(
            f"Iteration {iteration + 1}: "
            f"best = {best_value.item():.4f}"
        )
        history.append(best_value.item())

    convergence_regret_plot(history, benchmark=args.function, acq=acq)

    best_history = random_search(n_iter=n_iter, objective=objective, seed=42)
    plot(best_history, best_value, objective)      
