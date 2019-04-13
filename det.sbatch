#!/bin/bash
#
#SBATCH --partition=normal
#sbatch --nodes=1 
#SBATCH --ntasks=4
#SBATCH --mem=1024
#SBATCH --output=det_%J_stdout.txt
#SBATCH --error=det_%J_stderr.txt
#SBATCH --time=00:01:00
#SBATCH --job-name=det
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/oucs044/mpi_determinant/
##################################################
mpirun -n 4 det 16 > det_output16.txt
