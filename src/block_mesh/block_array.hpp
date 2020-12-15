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

#ifndef ETHON_BLOCK_MESH_BLOCK_ARRAY_HPP_
#define ETHON_BLOCK_MESH_BLOCK_ARRAY_HPP_

#include <array>

template <typename T, size_t N1, size_t N2, size_t N3>
class Array3D {
public:
  Array3D() = default;

  // don't allow copy or assingment, since that would copy all the data
  Array3D(const Array3D &) = delete;
  Array3D &operator=(Array3D const &) = delete;

  inline T &operator()(uint i, uint j, uint k) { return data_[index(i, j, k)]; }
  inline const T &operator()(uint i, uint j, uint k) const { return data_[index(i, j, k)]; }

private:
  inline uint index(uint i, uint j, uint k) const { return (k * N2 + j) * N1 + i; }

  std::array<T, N1 * N2 * N3> data_;
};

template <typename T, size_t N1, size_t N2, size_t N3, size_t N4>
class Array4D {
public:
  Array4D() = default;

  // don't allow copy or assingment, since that would copy all the data
  Array4D(const Array4D &) = delete;
  Array4D &operator=(Array4D const &) = delete;

  inline T &operator()(uint i, uint j, uint k, uint l) { return data_[index(i, j, k, l)]; }
  inline const T &operator()(uint i, uint j, uint k, uint l) const {
    return data_[index(i, j, k, l)];
  }

private:
  inline uint index(uint i, uint j, uint k, uint l) const {
    return ((l * N3 + k) * N2 + j) * N1 + i;
  }

  std::array<T, N1 * N2 * N3 * N4> data_;
};

#endif // ETHON_BLOCK_MESH_BLOCK_ARRAY_HPP_
