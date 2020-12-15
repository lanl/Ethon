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

#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

std::vector<std::vector<double>> ReadAthenaResults(const std::string path, const size_t N) {
  FILE *f = fopen(path.c_str(), "r");
  if (f == nullptr) throw std::runtime_error("File '" + path + "' does not exist");

  std::vector<size_t> idx(N);
  std::vector<double> x(N), rho(N), eps(N), mu0(N), mu1(N), mu2(N);

  char buffer[1024];
  size_t i = 0;
  while (!feof(f)) {
    fgets(buffer, 1024, f);
    if (feof(f)) break;

    if (buffer[0] == '#') continue; // comment

    if (sscanf(buffer,
            "%04lu %le %le %le %le %le %le\n",
            &idx[i],
            &x[i],
            &rho[i],
            &eps[i],
            &mu0[i],
            &mu1[i],
            &mu2[i]) != 7) {
      throw std::runtime_error("An error occurred while reading data file '" + path + "'");
    }

    if (idx[i] != i + 2)
      throw std::runtime_error("Unexpected index at i = " + std::to_string(i) + " in file " + path);

    ++i;
  }
  fclose(f);

  return {x, rho, mu0, eps};
}

void CompareAthenaResults(const std::string path,
    const Mesh_u<1> &mesh,
    const std::vector<State_u<1>> &res,
    const double tol) {
  const auto N = mesh.N()[0];
  auto athena = ReadAthenaResults(path, N);

  auto &athena_x = athena[0];
  auto &athena_rho = athena[1];
  auto &athena_mu = athena[2];
  auto &athena_eps = athena[3];

  double max_rho_err = 0.0;
  double max_mu_err = 0.0;
  double max_eps_err = 0.0;
  for (size_t i = 0; i < N; ++i) {
    EXPECT_NEAR(mesh.x({i})[0], athena_x[i], 1.0e-6) << "x-coordinate mismatch at " << i;

    max_rho_err = std::max(max_rho_err, fabs(res[i].rho - athena_rho[i]));
    max_mu_err = std::max(max_mu_err, fabs(res[i].mu[0] - athena_mu[i]));
    max_eps_err = std::max(max_eps_err, fabs(res[i].epsilon - athena_eps[i]));
  }

  printf("max athena diffs: rho = %.10e, mu = %.10e, eps = %.10e\n",
      max_rho_err,
      max_mu_err,
      max_eps_err);

  EXPECT_LE(max_rho_err, tol) << "rho diff to Athena too big";
  EXPECT_LE(max_mu_err, tol) << "mu diff to Athena too big";
  EXPECT_LE(max_eps_err, tol) << "eps diff to Athena too big";
}
