.PHONY: benchmark
benchmark: 
	python3 experiment1.py


# remove all files created and return to just input files
.PHONY : clean
clean:
	rm -f results/*