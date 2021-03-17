/**
 * Deep Copy
 *
 * Complexity: simple
 * Tuning problem:
 *
 * Kokkos transforms data layouts of View depending on the architecture.
 *
 * That is, in a 3D view, we change which dimension is stride 1 access.
 * This means that in some cases, we need to transpose data if it's
 * operated on in multiple ExecutionSpaces
 *
 * It does so using an "MDRangePolicy," a set of tightly nested loops.
 *
 * These "MDRangePolicy's" have tile sizes, which you're picking.
 * Currently no features, but plans are to vary the size of these Views
 * to enable you to see whether optimal tile sizes vary with View shapes
 *
 * This is basically a smoke-test, can your tool tune tile sizes
 */
#include <tuning_playground.hpp>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <random>
#include <tuple>
#include <iostream>
int main(int argc, char* argv[]) {
  tuned_kernel(argc, argv, 
		  [&](const int total_iters) {
  Kokkos::View<float***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space> left("left", 100, 100, 100);
  Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> right("right", 100, 100, 100);
                    return std::make_pair(left, right); 
		  }, 
		  [&](const int x, const int total_iters, auto data) {
              auto left = data.first;
	      auto right = data.second;
	      Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, right, left); 
		  });
}
