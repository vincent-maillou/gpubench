sbatch -n 4 -w turing[02-03] -A gpu-bench submit.slurm 
module purge 
module load nvhpc/21.9  gcc/9.2.0-c7
mpicxx -O3 -acc -Minfo=acc ps_hybrid_acc.cpp

