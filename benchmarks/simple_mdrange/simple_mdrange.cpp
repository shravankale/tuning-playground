/**
 * simple_mdrange
 *
 * Complexity: medium
 *
 * Requires: GPU architecture (tested with cuda)
 *
 * Tuning problem:
 *
 * Kokkos has an "MDRangePolicy" to model tightly nested loops,
 * a lot of tuning problems are about tuning the tile sizes in
 * that construct
 *
 * The primary use case of this is to validate that tuning MDRanges
 * is feasible. For tuning folks it has the benefit of being
 * *hugely* multidimensional
 *
 * Note that this currently involves no features.
 *
 */
#include <tuning_playground.hpp>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <iostream>
#include <random>
#include <tuple>
int main(int argc, char *argv[]) {
  constexpr const int data_size = 1000;
  using view_type =
      Kokkos::View<float ******, Kokkos::DefaultExecutionSpace::memory_space>;

  tuned_kernel(
      argc, argv,
      [&](const int total_iters) {
        view_type process("process_this", 10, 10, 10, 10, 10, 10);
        return std::make_tuple(process);
      },
      [&](const int x, const int total_iters, view_type A) {
        Kokkos::MDRangePolicy<Kokkos::Cuda,Kokkos::Rank<6>> p({0,0,0,0,0,0},{10,10,10,10,10,10});
        Kokkos::parallel_for(
            "MDRange", p, KOKKOS_LAMBDA(int i1,int i2,int i3,int i4,int i5,int i6 ) {
              A( i1, i2, i3, i4, i5, i6) += 1.0f;
            });
      });
}
