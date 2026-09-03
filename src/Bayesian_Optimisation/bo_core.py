# Components for Bayesian Optimisation go here. 

import torch
import numpy as np 
from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_mll
from gpytorch import ExactMarginalLogLikelihood
from botorch.acquisition import (
    LogExpectedImprovement,
    UpperConfidenceBound,
    )
from botorch.optim import optimize_acqf
from botorch.test_functions import Hartmann, Ackley, Branin


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


def generate_initial_data(n=10, objective=None, bounds=None):
    """ Generate training data """
    d = bounds.shape[1]
    train_X = torch.rand(n, d, device=device, dtype=dtype)
    # scale values of x if 0-1 isn't actually the bounds of the problem
    train_X = bounds[0] + (bounds[1] - bounds[0]) * train_X
    train_Y = objective(train_X).unsqueeze(-1)
    return train_X, train_Y

def build_GP_surrogate(train_X, train_Y):
    """
    Build and fit a Gaussian Process surrogate model for Bayesian optimisation.
    Parameter(s):
        train_X (torch.Tensor): Training input data
        train_Y (torch.Tensor): Training output data
    Returns:
        model (SingleTaskGP): The fitted Gaussian Process model
    """
    # Create the GP model
    model = SingleTaskGP(train_X, train_Y)
    
    # Fit the model
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    
    return model

def build_acquisition(model, train_Y, acq="EI"):
    """ Construct and acquisition function. 
    Parameter(s):
        model (SingleTaskGP): The fitted Gaussian Process model
        train_Y (torch.Tensor): Training output data
        acq (str): The type of acquisition function to construct
    Returns:
        acquisition (AcquisitionFunction): The constructed acquisition function
    """
    acq = acq.upper()

    if acq == "EI":
        """using log expected improvement to solve numerical issues that lead to suboptimal optimization performance.
        https://arxiv.org/abs/2310.20708 for details."""
        acquisition = LogExpectedImprovement(
            model=model,
            best_f=train_Y.max(),
        )
    elif acq =="UCB":
        acquisition = UpperConfidenceBound(
            model=model,
            beta=2.0,
        )
    else:
        raise ValueError(f"{acq} is an invalid acquisition function. Expected EI or UCB")

    return acquisition

def optimize_candidate(acquisition, bounds, q=1):
    """
    Optimize the acquisition function to find the next candidate point.
    Parameter(s):
        acquisition (AcquisitionFunction): The acquisition function to optimize
        bounds (torch.Tensor): The bounds for the optimization
        q (int): The number of candidates to optimize
    Returns:
        candidates (torch.Tensor): The optimized candidate points
    """
    candidates, _ = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds,
        q=q,
        num_restarts=10,
        raw_samples=512,
    )
    return candidates

def evaluate_candidate(candidate, objective):
    """
    Evaluate the objective function at the given candidate point.
    Parameter(s):
        candidate (torch.Tensor): The candidate point to evaluate
        objective (callable): The objective function to evaluate
    Returns:
        value (torch.Tensor): The evaluated objective value
    """
    observation = objective(candidate).unsqueeze(-1)
    return observation   

def update_dataset(train_X, train_Y, candidate, observation):
    """
    Update the training dataset with the new candidate and its observation.
    Parameter(s):
        train_X (torch.Tensor): The current training input data
        train_Y (torch.Tensor): The current training output data
        candidate (torch.Tensor): The new candidate point
        observation (torch.Tensor): The observation at the new candidate point
    Returns:
        train_X (torch.Tensor): The updated training input data
        train_Y (torch.Tensor): The updated training output data
    """
    train_X = torch.cat([train_X, candidate], dim=0)
    train_Y = torch.cat([train_Y, observation], dim=0)
    return train_X, train_Y 

def bayesian_optimization(n_iter, objective, acq, seed):
    """
    Performs Bayesian Optimization. 
    Parameter(s):
        n_iter (int): The number of iterations to run
        objective (callable): The objective function to optimize
        acq (str): The acquisition function to use
        seed (int): The random seed for reproducibility
    Returns:
        history (list): A list of the best observations at each iteration
    """
    history = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    torch.manual_seed(seed)
    if objective == "hartmann":
        hartmann = Hartmann(negate=True).to(device=device)
        objective = hartmann
    elif objective == "ackley":
        ackley = Ackley(negate=True).to(device=device)
        objective = ackley
    elif objective == "branin":
        branin = Branin(negate=True).to(device=device)
        objective = branin

    bounds = objective.bounds.to(device=device, dtype=dtype)

    # Generate initial observations
    train_X, train_Y = generate_initial_data(n=10, objective=objective, bounds=bounds)

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

    return history