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

#include <cstdio>
#include <sstream>

#include <gtest/gtest.h>

#include "../convergence.hpp"
#include "../regression.hpp"
#include "../test_utils.hpp"

#include "hllc.hpp"
#include "ideal_gas_eos.hpp"
#include "slopes.hpp"

#include "uniform_mesh/godunov_1d.hpp"
#include "uniform_mesh/muscl_1d.hpp"

using Mesh = Mesh_u<1>;
using State = State_u<1>;
using EOS = IdealGas<State>;
using HLLC = HLLC_u<State, EOS>;

const EOS eos(1.4); // gamma = 1.4
const double cfl = 0.9;

/**
 * @brief Create an initialized mesh for the Sedov problem
 *
 * @param N   Number of cells
 * @return std::pair<Mesh, std::vector<State>>  Mesh and data
 */
std::pair<Mesh, std::vector<State>> MakeInitMesh(const size_t N) {
  const double rho0 = 1.0;
  const double p0 = 1.0e-4;
  const double Eblast = 0.8;

  Mesh mesh({-1.0}, {1.0}, {N});
  double dx = mesh.cell_size()[0];
  std::vector<State> init(N);

  auto init_state = State::FromPrimitive(rho0, {0.0}, p0, eos);
  // printf("Initial state: rho = %.10e, u = %.10e, eps = %.10e, e = %.10e, p = %.10e, cs =
  // %.10e\n",
  //     init_state.rho,
  //     init_state.u(0),
  //     init_state.epsilon,
  //     init_state.specific_internal_energy(),
  //     eos.pressure(init_state),
  //     eos.sound_speed(init_state));

  for (size_t i = 0; i < N; ++i) {
    init[i] = init_state;
  }

  init[N / 2 - 1].epsilon = 0.5 * Eblast / dx;
  init[N / 2].epsilon = 0.5 * Eblast / dx;

  return {mesh, init};
}

std::vector<std::vector<double>> ReadTimmesResults(const size_t N) {
  std::ostringstream ss;
  ss << "sedov_1d_files/timmes_sedov_" << N;

  FILE *f = fopen(ss.str().c_str(), "r");
  if (f == nullptr) throw std::runtime_error("File '" + ss.str() + "' does not exist");

  std::vector<int> idx(N);
  std::vector<double> x(N), rho(N), e(N), p(N), u(N), cs(N);

  char buffer[1024];
  size_t i = 0;
  while (!feof(f)) {
    fgets(buffer, 1024, f);
    if (feof(f)) break;

    if (buffer[0] == '#') continue; // comment

    if (sscanf(buffer,
            "%i %le %le %le %le %le %le\n",
            &idx[i],
            &x[i],
            &rho[i],
            &e[i],
            &p[i],
            &u[i],
            &cs[i]) != 7) {
      throw std::runtime_error("An error occurred while reading data file '" + ss.str() + "'");
    }
    if (idx[i] != int(i + 1))
      throw std::runtime_error(
          "Unexpected index at i = " + std::to_string(i) + " in file " + ss.str());

    ++i;
  }
  fclose(f);

  return {x, rho, u, p, e, cs};
}

void CompareTimmesResults(
    const Mesh &mesh, const std::vector<State> &res, std::map<std::string, Errors> *errors) {
  const auto N = mesh.N()[0] / 2;
  auto timmes = ReadTimmesResults(N);

  auto &timmes_x = timmes[0];
  auto &timmes_rho = timmes[1];
  auto &timmes_u = timmes[2];
  auto &timmes_p = timmes[3];

  std::vector<double> rho(N), u(N), p(N), e(N), cs(N);
  for (size_t i = 0; i < N; ++i) {
    EXPECT_NEAR(mesh.x({N + i})[0], timmes_x[i], 1.0e-7) << "x-coordinate mismatch at " << i;

    rho[i] = res[i].rho;
    u[i] = res[i].u(0);
    p[i] = eos.pressure(res[i]);
    e[i] = res[i].specific_internal_energy();
    cs[i] = eos.sound_speed(res[i]);
  }

  double dx = mesh.cell_size()[0];
  (*errors)["rho"].push(rho, timmes_rho, dx);
  (*errors)["u"].push(u, timmes_u, dx);
  (*errors)["p"].push(p, timmes_p, dx);
}

// check whether solution is symmetric and return the right half to be compared against the Timmes
// result
std::vector<State> CheckSymmetric(const Mesh &mesh, const std::vector<State> &res) {
  const auto N = mesh.N()[0];

  for (size_t i = 0; i < N / 2; ++i) {
    auto &left = res[N / 2 + i];
    auto &right = res[N / 2 - i - 1];

    EXPECT_NEAR(left.rho, right.rho, 1.0e-12);
    EXPECT_NEAR(-left.mu[0], right.mu[0], 1.0e-12);
    EXPECT_NEAR(left.epsilon, right.epsilon, 1.0e-12);
  }

  return std::vector<State>(res.begin() + N / 2, res.end());
}

template <typename Solver>
void CheckConvergenceOrder(const double L1_conv_tol,
    const double L2_conv_tol,
    const double R2_tol,
    const double athena_tol,
    const std::string &output) {
  const double t_final = 0.5;
  std::map<std::string, Errors> errors;

  for (size_t n = 200; n <= 3200; n *= 2) {
    auto init = MakeInitMesh(n);
    auto mesh = init.first;
    auto res = Solver::Evolve(mesh, eos, cfl, init.second, 0.0, t_final);
    auto right_half = CheckSymmetric(mesh, res);
    CompareTimmesResults(mesh, right_half, &errors);

    CompareAthenaResults("sedov_1d_files/athena_sedov_" + std::to_string(n), mesh, res, athena_tol);

    auto name = "sedov_1d_files/" + output + "_" + std::to_string(n);
    auto err = RegressionTest(name, mesh, res, false);
    EXPECT_LE(err, 1.0e-10) << "Regression error too big: " << err;
  }

  for (auto &itm : errors) {
    ConvergenceOrder conv(itm.second);
    auto var = itm.first;
    printf("%s:\n", var.c_str());
    printf("  Linf: %.3f, R2 = %.4f\n", conv.conv_Linf, conv.R2_Linf);
    printf("    L1: %.3f, R2 = %.4f\n", conv.conv_L1, conv.R2_L1);
    printf("    L2: %.3f, R2 = %.4f\n", conv.conv_L2, conv.R2_L2);
    printf("\n");

    printf("L1: ");
    for (size_t i = 0; i < itm.second.L1_.size(); ++i)
      printf("%.10e, ", itm.second.L1_[i]);
    printf("\n");
    printf("L2: ");
    for (size_t i = 0; i < itm.second.L2_.size(); ++i)
      printf("%.10e, ", itm.second.L2_[i]);
    printf("\n");
    printf("Linf: ");
    for (size_t i = 0; i < itm.second.Linf_.size(); ++i)
      printf("%.10e, ", itm.second.Linf_[i]);
    printf("\n");

    EXPECT_GE(conv.conv_L1, L1_conv_tol) << "L1 error of " << var;
    EXPECT_GE(conv.conv_L2, L2_conv_tol) << "L2 error of " << var;
    EXPECT_GE(conv.R2_L1, R2_tol) << "L1 R2 of " << var;
    EXPECT_GE(conv.R2_L2, R2_tol) << "L2 R2 of " << var;
  }
}

TEST(Sedov_1D, Godunov) {
  CheckConvergenceOrder<
      Godunov1D<HLLC, BoundaryConditionType::Outflow, BoundaryConditionType::Outflow>>(
      0.57, 0.20, 0.88, 3.0, "godunov");
}

TEST(Sedov_1D, MUSCL_HLLC_MC) {
  CheckConvergenceOrder<
      MUSCL1D<HLLC, BoundaryConditionType::Outflow, BoundaryConditionType::Outflow,
      SlopeType::MC>>( 0.80, 0.39, 0.89, 3.1e-1, "muscl_HLLC_MC");
}

TEST(Sedov_1D, MUSCL_HLLC_SuperBee) {
  CheckConvergenceOrder<MUSCL1D<HLLC,
      BoundaryConditionType::Outflow,
      BoundaryConditionType::Outflow,
      SlopeType::SuperBee>>(0.90, 0.43, 0.84, 1.0, "muscl_HLLC_SuperBee");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
