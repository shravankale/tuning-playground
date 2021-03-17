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
#include <iostream>
#include <random>
#include <tuple>
int main(int argc, char *argv[]) {
  constexpr const int data_size = 1000;
  tuned_kernel(
      argc, argv,
      [&](const int total_iters) {
        Kokkos::View<float **, Kokkos::DefaultExecutionSpace::memory_space>
            left("left_inp", data_size, data_size);
        Kokkos::View<float **, Kokkos::DefaultExecutionSpace::memory_space>
            right("right_inp", data_size, data_size);
        Kokkos::View<float **, Kokkos::DefaultExecutionSpace::memory_space>
            output("output", data_size, data_size);
        return std::make_tuple(left, right, output);
      },
      [&](const int x, const int total_iters, auto data) {
        auto left = std::get<0>(data);
        auto right = std::get<1>(data);
        auto output = std::get<2>(data);

        fastest_of(
            "bad_gemms",
            [&]() {
              using team_policy =
                  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
              using team_member = team_policy::member_type;
              Kokkos::parallel_for(
                  "bad_team_gemm",
                  team_policy(data_size * data_size, Kokkos::AUTO,
                              Kokkos::AUTO),
                  KOKKOS_LAMBDA(const team_member &member) {
                    auto index = member.league_rank();
                    auto x = index % data_size;
                    auto y = index / data_size;
                    float sum = 0;
                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(member, data_size),
                        [&](int &i, float &lsum) {
                          lsum += left(x, i) * right(i, y);
                        },
                        sum);
                    output(x, y) = sum;
                  });
            },
            [&]() {
              Kokkos::parallel_for(
                  "bad_mdrange_gemm",
                  Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace,
                                        Kokkos::Rank<2>>(
                      {0, 0}, {data_size, data_size}),
                  KOKKOS_LAMBDA(const int x, const int y) {
                    for (int z = 0; z < data_size; ++z) {
                      output(x, y) += left(x, z) * right(z, y);
                    }
                  });
            });
      });
}
