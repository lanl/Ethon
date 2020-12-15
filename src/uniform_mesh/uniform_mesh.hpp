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

#ifndef ETHON_UNIFORM_MESH_MESH_HPP_
#define ETHON_UNIFORM_MESH_MESH_HPP_

#include <cassert>

using Kokkos::Array;

template <size_t DIM>
class Mesh_u {
public:
  Mesh_u(const Array<double, DIM> lower_bounds,
      const Array<double, DIM> upper_bounds,
      const Array<size_t, DIM> N)
      : lower_bounds_(lower_bounds), upper_bounds_(upper_bounds), N_(N) {
    for (size_t d = 0; d < DIM; ++d)
      cell_size_[d] = (upper_bounds_[d] - lower_bounds_[d]) / double(N_[d]);
  }

  static Mesh_u FromLowerBoundsAndCellSize(const Array<double, DIM> lower_bounds,
      const Array<double, DIM> cell_size,
      const Array<size_t, DIM> N) {
    Array<double, DIM> upper_bnds;
    for (size_t d = 0; d < DIM; ++d)
      upper_bnds[d] = cell_size[d] * double(N[d]) + lower_bounds[d];

    return Mesh_u(lower_bounds, upper_bnds, N);
  }

  auto lower_bounds() const { return lower_bounds_; }
  auto upper_bounds() const { return upper_bounds_; }
  auto N() const { return N_; }
  auto cell_size() const { return cell_size_; }

  KOKKOS_FORCEINLINE_FUNCTION Array<double, DIM> x(const Array<size_t, DIM> &index) const {
    Array<double, DIM> res;
    for (size_t d = 0; d < DIM; ++d) {
      assert(index[d] < N_[d]);
      res[d] = lower_bounds_[d] + cell_size_[d] * (0.5 + double(index[d]));
    }
    return res;
  }

private:
  Array<double, DIM> lower_bounds_;
  Array<double, DIM> upper_bounds_;
  Array<size_t, DIM> N_;
  Array<double, DIM> cell_size_;
};

#endif // ETHON_UNIFORM_MESH_MESH_HPP_
