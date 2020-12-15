//========================================================================================
// Copyright (c) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//
// This program is open source under the BSD-3 License. See LICENSE file for details.
//========================================================================================

#ifndef ETHON_UNIFORM_MESH_BOUNDARY_HPP_
#define ETHON_UNIFORM_MESH_BOUNDARY_HPP_

#include <cstdio>
#include <stdexcept>

#include <Kokkos_Core.hpp>

using namespace Kokkos;

enum class BoundaryConditionType { Outflow, Reflecting, Periodic };

enum class Side { Lower, Upper };

namespace {

/**
 * @brief Apply boundary condition in 1D on one side of the given 1D vector of states
 *
 * N is the number of real cells
 */
template <typename State, size_t N_GHOST, BoundaryConditionType BDRY, Side SIDE>
KOKKOS_FORCEINLINE_FUNCTION void Apply1DBdry(View<State *, LayoutStride, Kokkos::HostSpace> U, size_t N) {
  // use lower-case for real indices and upper-case for logical indices
  // logical index L corresponds to real index l = L + N_GHOST
  if (SIDE == Side::Lower) {
    for (size_t i = 0; i < N_GHOST; ++i) {
      int64_t G = i - N_GHOST; // logical index of ghost cell: -1, -2, -3, ..., -N_GHOST
      if (BDRY == BoundaryConditionType::Outflow) {
        U[i] = U[0 + N_GHOST]; // just copy logical cell 0
      } else if (BDRY == BoundaryConditionType::Reflecting) {
        U[i] = U[-(G + 1) + N_GHOST]; // copy logical cell -(G + 1) and invert velocity
        U[i].mu[0] =
            -U[i].mu[0]; // FIXME this is a bug, assumes boundary condition is in x-direction
      } else if (BDRY == BoundaryConditionType::Periodic) {
        U[i] = U[(N + G) + N_GHOST]; // copy logical cell N + G
      }
    }
  } else if (SIDE == Side::Upper) {
    // for (size_t i = N + N_GHOST; i < U.size(); ++i) {
    // replace i with ip so the loop can be unrolled
    for (size_t ip = 0; ip < N_GHOST; ++ip) {
      size_t i = ip + U.size() - N_GHOST;
      int64_t G = i - N_GHOST; // logical index of ghost cell: N, N+1, N+2, ..., N+N_GHOST-1
      if (BDRY == BoundaryConditionType::Outflow) {
        U[i] = U[(N - 1) + N_GHOST]; // just copy logical cell N - 1
      } else if (BDRY == BoundaryConditionType::Reflecting) {
        U[i] = U[(N - (G - N + 1)) + N_GHOST]; // copy logical cell N - (G - N + 1) and invert u
        U[i].mu[0] =
            -U[i].mu[0]; // FIXME this is a bug, assumes boundary condition is in x-direction
      } else if (BDRY == BoundaryConditionType::Periodic) {
        U[i] = U[(G - N) + N_GHOST]; // copy logical cell G - N
      }
    }
  }
}

} // namespace

template <typename State, size_t N_GHOST, BoundaryConditionType BDRY, Side SIDE>
class Boundary1D {
public:
  static KOKKOS_IMPL_FORCEINLINE void Apply(std::vector<State> &U) {
    auto N = U.size() - 2 * N_GHOST; // number of real cells
    View<State *, Kokkos::HostSpace, MemoryTraits<Unmanaged>> view(U.data(), U.size());
    Apply1DBdry<State, N_GHOST, BDRY, SIDE>(view, N);
  }
};

template <typename State, size_t N_GHOST, size_t DIR, BoundaryConditionType BDRY, Side SIDE>
class Boundary3D {
  static_assert((DIR >= 0) && (DIR <= 2), "DIR must be 0, 1, or 2");

public:
  using ExecSpace = Kokkos::OpenMP;

  static KOKKOS_IMPL_FORCEINLINE void Apply(Kokkos::View<State ***, Kokkos::HostSpace> &U) {
    auto N = U.extent(DIR) - 2 * N_GHOST;

    if (DIR == 0) {
      parallel_for(
          MDRangePolicy<ExecSpace, Rank<2>>(
              {N_GHOST, N_GHOST}, {U.extent(1) - N_GHOST, U.extent(2) - N_GHOST}),
          KOKKOS_LAMBDA(const int &j, const int &k) {
            auto sub = subview(U, ALL(), j, k);
            Apply1DBdry<State, N_GHOST, BDRY, SIDE>(sub, N);
          });
    } else if (DIR == 1) {
      parallel_for(
          MDRangePolicy<ExecSpace, Rank<2>>(
              {N_GHOST, N_GHOST}, {U.extent(0) - N_GHOST, U.extent(2) - N_GHOST}),
          KOKKOS_LAMBDA(const int &i, const int &k) {
            auto sub = subview(U, i, ALL(), k);
            Apply1DBdry<State, N_GHOST, BDRY, SIDE>(sub, N);
          });
    } else if (DIR == 2) {
      parallel_for(
          MDRangePolicy<ExecSpace, Rank<2>>(
              {N_GHOST, N_GHOST}, {U.extent(0) - N_GHOST, U.extent(1) - N_GHOST}),
          KOKKOS_LAMBDA(const int &i, const int &j) {
            auto sub = subview(U, i, j, ALL());
            Apply1DBdry<State, N_GHOST, BDRY, SIDE>(sub, N);
          });
    }
  }
};

#endif // ETHON_UNIFORM_MESH_BOUNDARY_HPP_
