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

#ifndef ETHON_KOKKOS_OMP_BLOCK_POOL_HPP_
#define ETHON_KOKKOS_OMP_BLOCK_POOL_HPP_

#include <mutex>
#include <stack>
#include <vector>

#include <Kokkos_Core.hpp>

#include "kokkos_omp/block.hpp"
#include "state.hpp"

template <typename Block_t>
class BlockPool {
public:
  constexpr static int N_blocks = Block_t::N_blocks;
  using CellData_t = typename Block_t::CellData_t;
  using FluxData_t = typename Block_t::FluxData_t;

  BlockPool()
      : blocks_(N_blocks),
        refinement_flags_("Refinement flags"),
        cell_data("cell_data"),
        x_fluxes("x_fluxes"),
        y_fluxes("y_fluxes"),
        z_fluxes("z_fluxes"),
        lower_boundary("lower_boundary"),
        upper_boundary("upper_boundary") {
    std::lock_guard<std::mutex> lock(mutex_);

    // push from back to front, so that we'll use the chunk from beginning to end (not that it
    // matters...)
    for (int i = 0; i < N_blocks; ++i) {
      free_ids_.push(N_blocks - 1 - i);
    }
  }

  int size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return N_blocks - free_ids_.size();
  }

  auto refinement_flags() { return refinement_flags_; }
  auto refinement_flag(int b_id) const {
    return refinement_flags_(b_id);
  }

  Block_t *Allocate(const Kokkos::Array<double, 3> lower_bounds,
      const Kokkos::Array<double, 3> cell_size,
      const int level) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_ids_.size() == 0)
      throw std::out_of_range("Requested more blocks than there are available");

    auto b_id = free_ids_.top();
    free_ids_.pop();

    blocks_[b_id].Init(b_id, lower_bounds, cell_size, level);
    refinement_flags_(b_id) = 0;
    return &blocks_[b_id];
  }

  void Free(int bid) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_ids_.push(bid);
  }

private:
  std::vector<Block_t> blocks_;

  // stack of pointers to free blocks
  std::stack<int> free_ids_;

  // refinement flags
  Kokkos::View<int[N_blocks], Kokkos::HostSpace> refinement_flags_;

  // mutex to serialize access to the pool
  mutable std::mutex mutex_;

public:
  //   [0]    mass density in g cm^{-3}
  // [1,2,3]  momentum density (mu_i = rho * u_i, where u_i is fluid velocity) in g cm^{-2} s^{-1}
  //   [4]    total energy density (rho * e + 1/2 * rho * u_i u^i, where e is the internal specific
  //          energy) in erg cm^{-3} = g cm^{-1} s^{-2}
  CellData_t cell_data;

  CellData_t x_fluxes, y_fluxes, z_fluxes;
  FluxData_t lower_boundary, upper_boundary;
};

#endif // ETHON_KOKKOS_OMP_BLOCK_POOL_HPP_
