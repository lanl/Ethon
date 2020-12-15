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

#ifndef ETHON_KOKKOS_OMP_AMR_MESH_INFO_HPP_
#define ETHON_KOKKOS_OMP_AMR_MESH_INFO_HPP_

#include <cassert>

#include <Kokkos_Core.hpp>

#include "uniform_mesh/uniform_mesh.hpp"

using Kokkos::Array;

template <size_t BLOCK_SIZE>
class AMRMeshInfo {
public:
  AMRMeshInfo(const Array<double, 3> lower_bounds,
      const Array<double, 3> upper_bounds,
      const Array<size_t, 3> num_cells)
      : lower_bounds_(lower_bounds), upper_bounds_(upper_bounds), num_cells_(num_cells) {
    std::vector<std::string> dir_str{"x", "y", "z"};
    for (size_t d = 0; d < 3; ++d) {
      num_blocks_[d] = num_cells_[d] / BLOCK_SIZE;
      if (num_blocks_[d] * BLOCK_SIZE != num_cells_[d]) {
        throw std::invalid_argument("Number of cells in the " + dir_str[d] + "-direction (" +
                                    std::to_string(num_cells_[d]) +
                                    ") is not a multiple of the block size (" +
                                    std::to_string(BLOCK_SIZE) + ")");
      }
      cell_size_[d] = (upper_bounds_[d] - lower_bounds_[d]) / double(num_cells_[d]);
    }
  }

  AMRMeshInfo(const Mesh_u<3> &uniform_mesh)
      : AMRMeshInfo(uniform_mesh.lower_bounds(), uniform_mesh.upper_bounds(), uniform_mesh.N()) {}

  static AMRMeshInfo FromLowerBoundsAndCellSize(const Array<double, 3> lower_bounds,
      const Array<double, 3> cell_size,
      const Array<size_t, 3> num_cells) {
    Array<double, 3> upper_bnds;
    for (size_t d = 0; d < 3; ++d)
      upper_bnds[d] = cell_size[d] * double(num_cells[d]) + lower_bounds[d];

    return AMRMeshInfo(lower_bounds, upper_bnds, num_cells);
  }

  auto lower_bounds() const { return lower_bounds_; }
  auto upper_bounds() const { return upper_bounds_; }
  auto num_cells() const { return num_cells_; }
  auto num_blocks() const { return num_blocks_; }
  auto N() const { return num_cells_; }
  auto cell_size() const { return cell_size_; }

  // return cell size for the given AMR level
  auto cell_size(const int level) const {
    Kokkos::Array<double, 3> cell_size;

    double divisor = 1.0 / double(1 << level);
    for (int i = 0; i < 3; ++i) {
      cell_size[i] = cell_size_[i] * divisor;
    }

    return cell_size;
  }

  // p4est vertex coordinates range from 0.0 to num_blocks[i] in dimension i
  Array<double, 3> TransformP4estVertex(const double vertex[3]) const {
    Array<double, 3> res;
    for (size_t d = 0; d < 3; ++d)
      res[d] = lower_bounds_[d] +
               vertex[d] * (upper_bounds_[d] - lower_bounds_[d]) / double(num_blocks_[d]);

    return res;
  }

  Array<double, 3> CellCenter(const Array<size_t, 3> &index) const {
    Array<double, 3> res;
    for (size_t d = 0; d < 3; ++d) {
      assert(index[d] < num_cells_[d]);
      res[d] = lower_bounds_[d] + cell_size_[d] * (0.5 + double(index[d]));
    }
    return res;
  }

  Array<double, 3> CellLowerBounds(const Array<size_t, 3> &index) const {
    Array<double, 3> res;
    for (size_t d = 0; d < 3; ++d) {
      assert(index[d] < num_cells_[d]);
      res[d] = lower_bounds_[d] + cell_size_[d] * double(index[d]);
    }
    return res;
  }

private:
  Array<double, 3> lower_bounds_;
  Array<double, 3> upper_bounds_;
  Array<size_t, 3> num_cells_;
  Array<size_t, 3> num_blocks_;
  Array<double, 3> cell_size_;
};

#endif // ETHON_KOKKOS_OMP_AMR_MESH_INFO_HPP_
