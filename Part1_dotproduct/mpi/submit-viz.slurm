#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=24
#SBATCH --threads-per-core=1
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH -p visu 

module purge
module load nvhpc/21.9
module load gcc/9.2.0-c7

ni=`printf "%07d\n" ${SLURM_NPROCS}`
echo $ni 

mpirun -np ${SLURM_NPROCS} ./aviz.out -n 600000000 -m 1000 > outViz_${ni}c.txt
