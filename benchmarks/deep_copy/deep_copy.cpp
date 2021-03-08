/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

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
  Kokkos::View<float***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace::memory_space> left("left", total_iters, total_iters, total_iters);
  Kokkos::View<float***, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace::memory_space> right("right", total_iters, total_iters, total_iters);
                    return std::make_pair(left, right); 
		  }, 
		  [&](const int x, const int total_iters, auto data) {
              auto left = data.first;
	      auto right = data.second;
	      Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{}, right, left); 
		  });
}
