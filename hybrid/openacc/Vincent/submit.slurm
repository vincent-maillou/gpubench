#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=2
#SBATCH --threads-per-core=1
#SBATCH --time=00:10:00
##SBATCH --exclusive
#SBATCH -p gpus 

module purge
module load nvhpc/21.9
module load gcc/9.2.0-c7

ni=`printf "%07d\n" ${SLURM_NPROCS}`
echo $ni 

mpirun -np ${SLURM_NPROCS} ./dotp_openacc.out -n 9600000000 -m 1 > out_${ni}c.txt
#mpirun -np ${SLURM_NPROCS} ./dotp_openacc.out -n 240000000 -m 1 > out_${ni}c.txt
