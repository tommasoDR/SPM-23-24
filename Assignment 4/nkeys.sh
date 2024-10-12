#!/bin/sh

#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH -o results.log
#SBATCH -e errors.err

srun --mpi=pmix -n 2 ./nkeyspar 2 1000000
srun --mpi=pmix -n 3 ./nkeyspar 2 1000000
srun --mpi=pmix -n 4 ./nkeyspar 2 1000000
srun --mpi=pmix -n 8 ./nkeyspar 2 1000000
srun --mpi=pmix -n 16 ./nkeyspar 2 1000000
srun --mpi=pmix -n 24 ./nkeyspar 2 1000000

srun --mpi=pmix -n 2 ./nkeyspar 100 1000000
srun --mpi=pmix -n 3 ./nkeyspar 100 1000000
srun --mpi=pmix -n 4 ./nkeyspar 100 1000000
srun --mpi=pmix -n 8 ./nkeyspar 100 1000000
srun --mpi=pmix -n 16 ./nkeyspar 100 1000000
srun --mpi=pmix -n 24 ./nkeyspar 100 1000000

srun --mpi=pmix -n 2 ./nkeyspar 1000 1000000
srun --mpi=pmix -n 3 ./nkeyspar 1000 1000000
srun --mpi=pmix -n 4 ./nkeyspar 1000 1000000
srun --mpi=pmix -n 8 ./nkeyspar 1000 1000000
srun --mpi=pmix -n 16 ./nkeyspar 1000 1000000
srun --mpi=pmix -n 24 ./nkeyspar 1000 1000000