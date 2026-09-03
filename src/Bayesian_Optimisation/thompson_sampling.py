def thompson_sampling(model, candidate_X):

    if candidate_X.shape[0] == 0:
        raise ValueError("Candidate pool is empty.")

    posterior = model.posterior(
        candidate_X.unsqueeze(1)
    )

    sampled_values = posterior.rsample()

    candidate_idx = sampled_values.argmax().item()

    candidate = candidate_X[candidate_idx].unsqueeze(0)

    return candidate, candidate_idx