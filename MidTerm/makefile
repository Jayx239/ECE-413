all:
		gcc -fopenmp chol.c chol_gold.c chol.h mt.h -lpthread -std=c99 -lm -o chol
		gcc -o solver solver.c solver_gold.c -fopenmp -std=c99 -lm -lpthread
