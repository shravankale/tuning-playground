#ifndef TUNINGPLAYRGROUND_HPP
#define TUNINGPLAYRGROUND_HPP

#include<Kokkos_Core.hpp>

template<typename Setup, typename Tunable>
void tuned_kernel(int argc, char* argv[], Setup setup, Tunable tunable){
  int num_iters = 1000;
  bool tuned_internals;
  bool found_tuning_tool;
  bool print_progress;
  Kokkos::initialize(argc, argv);
  {
    auto kernel_data = setup(num_iters);
    for(int x =0; x < num_iters; ++x) {
      tunable(x, num_iters, kernel_data);
    } 
  
  }
  Kokkos::finalize();
}

#endif
