import argparse

import torch
from plotting import convergence_regret_plot
from src.Bayesian_Optimisation.bo_core import (
    build_GP_surrogate, build_acquisition,
    optimize_candidate, evaluate_candidate,
    update_dataset,
    )
from botorch.test_functions import Hartmann, Ackley, Branin

import warnings
from botorch.exceptions.warnings import InputDataWarning

""" turn off "Input features is not contained to the unit cube" warning. 
I know, deliberate choice to work in the native domain to simplify the code. """
warnings.filterwarnings("ignore", category=InputDataWarning)

def generate_initial_data(n=10, objective=None, bounds=None):
    """ Generate training data """
    d = bounds.shape[1]
    train_X = torch.rand(n, d, device=device, dtype=dtype)
    # scale values of x if 0-1 isn't actually the bounds of the problem
    train_X = bounds[0] + (bounds[1] - bounds[0]) * train_X
    train_Y = objective(train_X).unsqueeze(-1)
    return train_X, train_Y

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
    best_observations = []
    # Run BO
    for iteration in range(20):
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
        best_observations.append(best_value.item())

    convergence_regret_plot(best_observations, benchmark=args.function, acq=acq)
