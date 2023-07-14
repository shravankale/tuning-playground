
export APEX_PROC_SELF_STATUS=0
export APEX_PROC_STAT=0
export APEX_PROC_LOADAVG=0

# If you want to capture  scatterplot during execution, only capture 10% of the timers.
export APEX_SCATTERPLOT_FRACTION=0.01

# Do some cleanup
rm -rf core.* /tmp/ah_se*.log tuned untuned nokokkos *.pdf *.png apex_converged_tuning.yaml *.pdf *.csv
set -x

#./install/apex/bin/apex_exec --apex:kokkos --apex:kokkos_tuning \ --apex:kokkos_fence
#./install/apex/bin/apex_exec --apex:debug --apex:kokkos --apex:kokkos_tuning --apex:scatter --apex:postprocess \
#--apex:cuda --apex:cuda_counters --apex:cuda_details \
#install/tuning-playground/bin/simple_mdrange.exe --kokkos-tune-internals

#--apex:cuda --apex:cuda_counters --apex:cuda_details \
#Run
/home/users/shravank/projects/KokkosXApex_updated/install/apex/bin/apex_exec --apex:csv --apex:kokkos --apex:kokkos_tuning --apex:kokkos_fence --apex:scatter --apex:postprocess \
--apex:ompt --apex:ompt_details \
/home/users/shravank/projects/KokkosXApex_updated/install/tuning-playground/bin/mm2d_large_tilling.exe 
#--kokkos-tune-internals

##Debug
##./install/apex/bin/apex_exec --apex:debug --apex:kokkos --apex:kokkos_tuning \
##--apex:cuda --apex:cuda_counters --apex:cuda_details \
##install/tuning-playground/bin/simple_mdrange.exe --kokkos-tune-internals

#apex_kokkos_tuning.cpp:271