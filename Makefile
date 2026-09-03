.PHONY: benchmark
benchmark: 
	python3 -m src.Bayesian_Optimisation.experiments.experiment1
.PHONY: materials
materials:
	python3 -m src.Bayesian_Optimisation.experiments.experiment2

# remove all files created and return to just input files
.PHONY : clean
clean:
	rm -f results/*; rm -f data/*