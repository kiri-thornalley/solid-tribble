# Bayesian Optimisation for Materials Discovery
## Motivation
Developing novel materials and predicting their properties can be both time-consuming and expensive in terms of materials consumed. 
## Method
A Gaussian Process surrogate was fitted to the search space and iteratively updated as decisions were made. At each step, the acquisition function is evaluated and the material with the highest value was selected as the next point to explore. 

## Experimental Setup
Finite set of 200 materials were sampled from the Materials Project. Initially, 10 materials were used to initially train the GP, a further 140 form part of the candidate pool the model can choose from and the remainder form the held-out test set.
Features band_gap, Volume, density, nElements and eFermi were used to train the Gaussian Process, and formation_energy was used as the target variable. 10 random seeds were applied to each experiment. 

## Results
### Experiment 1
Benchmarked BO against Random Search on analytical test functions (Branin, Ackley, Hartmann6). 
Median best-so-far on a 10-seed study, shows that variations of BO (EI, UCB) converge on known optima much faster than random search. 
<img width="1491" height="491" alt="13d3389d-5704-4dbc-aa85-a5593d3a8c06" src="https://github.com/user-attachments/assets/4aa99ea4-cae9-4b60-82c3-5193c5ca4f31" />

### Experiment 2
Evaluated BO-EI, BO-UCB and Thompson Sampling on a discrete BO problem, to investigate how efficiently each method discovers low-formation energy materials from a finite candidate set. 

## Limitations
Materials Project API is used as a stand-in for the expensive experimental evaluation of novel materials; the present work does not integrate physical synthesis or characterisation hardware.  
