#module load cmake
#module load nvhpc/22.9
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
#export OMP_DISPLAY_ENV=true
export APEX_KOKKOS_TUNING_WINDOW=10
export APEX_KOKKOS_TUNING_POLICY=random

