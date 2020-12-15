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

#include "convergence.hpp"

#include <cstdio>

void Errors::push(
    const std::vector<double> &result, const std::vector<double> &expected, const double dx) {
  assert(result.size() == expected.size());
  double Linf = 0.0;
  double L1 = 0.0;
  double L2 = 0.0;

  // linearly scale number to be in the range [0, 1] defined by expected min and max
  // auto minmax = std::minmax_element(expected.begin(), expected.end());
  for (size_t i = 0; i < result.size(); ++i) {
    // double res = (result[i] - *minmax.first) / (*minmax.second - *minmax.first);
    // double exp = (expected[i] - *minmax.first) / (*minmax.second - *minmax.first);
    double diff = fabs(result[i] - expected[i]);

    Linf = std::max(Linf, diff);
    L1 += diff;
    L2 += diff * diff;
  }

  // printf("dx = %.4e: Linf = %.6e, L1 = %.6e, L2 = %.6e\n", dx, Linf, dx * L1, sqrt(dx * L2));

  dx_.push_back(dx);
  Linf_.push_back(Linf);
  L1_.push_back(dx * L1);
  L2_.push_back(sqrt(dx * L2));
}

ConvergenceOrder::ConvergenceOrder(const Errors &errs) {
  ComputeConvergence(errs.dx_, errs.Linf_, &conv_Linf, &R2_Linf);
  ComputeConvergence(errs.dx_, errs.L1_, &conv_L1, &R2_L1);
  ComputeConvergence(errs.dx_, errs.L2_, &conv_L2, &R2_L2);
}

void ConvergenceOrder::ComputeConvergence(
    const std::vector<double> &x, const std::vector<double> &y, double *conv, double *R2) {
  assert(x.size() == y.size());
  double n = double(x.size());
  double Sx = 0.0;
  double Sy = 0.0;
  double Sxx = 0.0;
  double Syy = 0.0;
  double Sxy = 0.0;

  for (size_t i = 0; i < x.size(); ++i) {
    double lx = log(x[i]);
    double ly = log(y[i]);
    Sx += lx;
    Sxx += lx * lx;
    Sy += ly;
    Syy += ly * ly;
    Sxy += lx * ly;
  }

  *conv = (n * Sxy - Sx * Sy) / (n * Sxx - Sx * Sx);
  double r = (n * Sxy - Sx * Sy) / sqrt((n * Sxx - Sx * Sx) * (n * Syy - Sy * Sy));
  *R2 = r * r;
}
