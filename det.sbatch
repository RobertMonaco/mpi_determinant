#!/bin/bash
#
#SBATCH --partition=normal
#sbatch --nodes=1 
#SBATCH --ntasks=16
#SBATCH --mem=1024
#SBATCH --output=det_%J_stdout.txt
#SBATCH --error=det_%J_stderr.txt
#SBATCH --time=00:30:00
#SBATCH --job-name=det
#SBATCH --mail-user=robertmonaco@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/oucs016/SIPE/det_output
##################################################
mpirun -n 16 mpi_determinant/det 16 > det_output16.txt