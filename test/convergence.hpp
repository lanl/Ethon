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

#ifndef ETHON_TEST_CONVERGENCE_HPP_
#define ETHON_TEST_CONVERGENCE_HPP_

#include <cassert>
#include <cmath>
#include <vector>

struct Errors {
  std::vector<double> dx_, Linf_, L1_, L2_;

  void push(
      const std::vector<double> &result, const std::vector<double> &expected, const double dx);
};

struct ConvergenceOrder {
  double conv_Linf, conv_L1, conv_L2; // order of convergence in the different norms
  double R2_Linf, R2_L1, R2_L2;       // correlation coefficients in the different norms

  ConvergenceOrder(const Errors &errs);

  void ComputeConvergence(
      const std::vector<double> &x, const std::vector<double> &y, double *conv, double *R2);
};

#endif // ETHON_TEST_CONVERGENCE_HPP_
