
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
<img width="1500" height="500" alt="convergence_plot" src="https://github.com/user-attachments/assets/3833aa2d-f892-4dac-b79c-0525626274b8" />

### Experiment 2
Evaluated BO-EI, BO-UCB and Thompson Sampling on a discrete BO problem, to investigate how efficiently each method discovers low-formation energy materials from a finite candidate set. 
<img width="800" height="600" alt="materials_convergence_plot" src="https://github.com/user-attachments/assets/490f1777-737d-423f-b041-03b0a257ef40" />

Median best-so-far on a 10-seed study, UCB showed greater exploratory behaviour in the discrete materials search, selecting materials with higher posterior uncertainty; and exploring areas not well represented by the initial data the GP was trained on. 
## Limitations
Materials Project API is used as a stand-in for the expensive experimental evaluation of novel materials; the present work does not integrate physical synthesis or characterisation hardware.  
