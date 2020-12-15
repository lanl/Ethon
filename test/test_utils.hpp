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

#ifndef ETHON_TEST_TEST_UTILS_HPP_
#define ETHON_TEST_TEST_UTILS_HPP_

#include <string>
#include <vector>

#include "state.hpp"
#include "uniform_mesh/uniform_mesh.hpp"

std::vector<std::vector<double>> ReadAthenaResults(const std::string path, const size_t N);

void CompareAthenaResults(const std::string path,
    const Mesh_u<1> &mesh,
    const std::vector<State_u<1>> &res,
    const double tol);

#endif // ETHON_TEST_TEST_UTILS_HPP_
