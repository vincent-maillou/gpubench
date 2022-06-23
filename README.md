> This project is under development
# gpubench
Benchmarking of performances and usability of several heterogenous computing languages on hpc mini-apps

## Overview
This project aims to evaluate the portability of classical HPC algorithms from pure-CPU
applications to GPU and hybrid technologies. 


NOTES:
Compilation Kokkos sur Liger dans le module singularity
-  salloc -p gpus -w turing03
-  module load singularity
-  singularity shell --nv ../jupyter-nvhpc20.9_latest.sif 
-  export CMAKE=/install/cmake-3.22.1-linux-x86_64/bin/cmake
-  $CMAKE -DCMAKE_CXX_EXTENSIONS=Off ..

