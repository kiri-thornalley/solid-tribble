import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


def random_search(n_iter=20, objective=None, seed=None):
    """
    Perform random search optimization.
    Parameter(s):
        n_iter (int): Number of iterations to run.
        objective (callable): The objective function to optimize.
        seed (int): The random seed for reproducibility.
    Returns:
        best_history (list): A list of the best values found at each iteration.
    """

    if seed is not None:
        torch.manual_seed(seed)

    best_history = []
    best_value = None
    bounds = objective.bounds.to(device=device, dtype=dtype)

    for i in range(n_iter):
        x = torch.rand(
            1, bounds.shape[1],
            device=device,
            dtype=dtype
        )
        #scale to actual bounds
        x = bounds[0] + (bounds[1] - bounds[0]) * x
        y = objective(x)

        if best_value is None:
            best_value = y.item()
        else:
            best_value = max(best_value, y.item())

        best_history.append(best_value)
        print(f"Iteration {i + 1}: best = {best_value:.4f}")
    return best_history

