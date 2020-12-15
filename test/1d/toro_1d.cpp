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
 * @brief Create an initialized mesh. The state is L to the left of x0 and R to the right.
 *
 * @param N   Number of cells
 * @param x0  Location of diaphragm
 * @param L   Initial state to the left of diaphragm
 * @param R   Initial state to the right of diaphragm
 * @return std::pair<Mesh, std::vector<State>>  Mesh and data
 */
std::pair<Mesh, std::vector<State>> MakeInitMesh(
    const size_t N, const double x0, const State &L, const State &R) {
  Mesh mesh({0.0}, {1.0}, {N});
  std::vector<State> dat(N);

  for (size_t i = 0; i < N; ++i) {
    double x = mesh.x({i})[0];
    dat[i] = (x <= x0) ? L : R;
  }

  return {mesh, dat};
}

std::vector<std::vector<double>> ReadToroResults(const std::string &prefix, const size_t N) {
  std::ostringstream ss;
  ss << "toro_1d_files/" << prefix << std::setw(4) << std::setfill('0') << N;

  FILE *f = fopen(ss.str().c_str(), "r");
  if (f == nullptr) throw std::runtime_error("File '" + ss.str() + "' does not exist");

  std::vector<double> x(N), rho(N), u(N), p(N), e(N);
  for (size_t i = 0; i < N; ++i) {
    if (fscanf(f, "%le %le %le %le %le\n", &x[i], &rho[i], &u[i], &p[i], &e[i]) != 5)
      throw std::runtime_error("Could not read Toro result file " + ss.str());
  }
  fclose(f);

  return {x, rho, u, p, e};
}

void CompareToroResults(const std::string &prefix,
    const Mesh &mesh,
    const std::vector<State> &res,
    std::map<std::string, Errors> *errors) {
  const auto N = mesh.N()[0];
  auto toro = ReadToroResults(prefix, N);

  auto &toro_x = toro[0];
  auto &toro_rho = toro[1];
  auto &toro_u = toro[2];
  auto &toro_p = toro[3];
  auto &toro_e = toro[4];

  std::vector<double> rho(N), u(N), p(N), e(N);
  for (size_t i = 0; i < N; ++i) {
    double x = mesh.lower_bounds()[0] + (0.5 + double(i)) * mesh.cell_size()[0];
    EXPECT_NEAR(x, toro_x[i], 1.0e-7) << "x-coordinate mismatch at " << i;

    rho[i] = res[i].rho;
    u[i] = res[i].u(0);
    p[i] = eos.pressure(res[i]);
    e[i] = res[i].specific_internal_energy();
  }

  double dx = mesh.cell_size()[0];
  (*errors)["rho"].push(rho, toro_rho, dx);
  (*errors)["u"].push(u, toro_u, dx);
  (*errors)["p"].push(p, toro_p, dx);
  (*errors)["e"].push(e, toro_e, dx);
}

template <typename Solver>
void CheckConvergenceOrder(const double x0,
    const double t_final,
    const State &L,
    const State &R,
    const std::string &toro_path,
    const double L1_conv_tol,
    const double L2_conv_tol,
    const double R2_tol,
    const std::string &output) {
  std::map<std::string, Errors> errors;

  for (size_t n = 100; n <= 3200; n *= 2) {
    auto init = MakeInitMesh(n, x0, L, R);
    auto mesh = init.first;
    auto res = Solver::Evolve(mesh, eos, cfl, init.second, 0.0, t_final);
    CompareToroResults(toro_path, mesh, res, &errors);

    auto name = "toro_1d_files/" + output + "_" + toro_path + std::to_string(n);
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

    EXPECT_GE(conv.conv_L1, L1_conv_tol) << "L1 error of " << var;
    EXPECT_GE(conv.conv_L2, L2_conv_tol) << "L2 error of " << var;
    EXPECT_GE(conv.R2_L1, R2_tol) << "L1 R2 of " << var;
    EXPECT_GE(conv.R2_L2, R2_tol) << "L2 R2 of " << var;
  }
}

using Godunov = Godunov1D<HLLC, BoundaryConditionType::Outflow, BoundaryConditionType::Outflow>;
using MUSCL =
    MUSCL1D<HLLC, BoundaryConditionType::Outflow, BoundaryConditionType::Outflow, SlopeType::MC>;

TEST(Toro_1d_Godunov, Test_1) {
  auto L = State::FromPrimitive(1.0, {0.75}, 1.0, eos);
  auto R = State::FromPrimitive(0.125, {0.0}, 0.1, eos);
  CheckConvergenceOrder<Godunov>(0.3, 0.2, L, R, "toro_1_", 0.56, 0.28, 0.94, "godunov");
}

TEST(Toro_1d_Godunov, Test_2) {
  auto L = State::FromPrimitive(1.0, {-2.0}, 0.4, eos);
  auto R = State::FromPrimitive(1.0, {2.0}, 0.4, eos);
  CheckConvergenceOrder<Godunov>(0.5, 0.15, L, R, "toro_2_", 0.62, 0.51, 0.98, "godunov");
}

TEST(Toro_1d_Godunov, Test_3) {
  auto L = State::FromPrimitive(1.0, {0.0}, 1000.0, eos);
  auto R = State::FromPrimitive(1.0, {0.0}, 0.01, eos);
  CheckConvergenceOrder<Godunov>(0.5, 0.012, L, R, "toro_3_", 0.49, 0.22, 0.98, "godunov");
}

TEST(Toro_1d_Godunov, Test_4) {
  auto L = State::FromPrimitive(5.99924, {19.5975}, 460.894, eos);
  auto R = State::FromPrimitive(5.99242, {-6.19633}, 46.0950, eos);
  CheckConvergenceOrder<Godunov>(0.4, 0.035, L, R, "toro_4_", 0.51, 0.24, 0.95, "godunov");
}

TEST(Toro_1d_Godunov, Test_5) {
  auto L = State::FromPrimitive(1.0, {-19.59745}, 1000.0, eos);
  auto R = State::FromPrimitive(1.0, {-19.59745}, 0.01, eos);
  CheckConvergenceOrder<Godunov>(0.8, 0.012, L, R, "toro_5_", 0.80, 0.44, 0.63, "godunov");
}

TEST(Toro_1d_MUSCL, Test_1) {
  auto L = State::FromPrimitive(1.0, {0.75}, 1.0, eos);
  auto R = State::FromPrimitive(0.125, {0.0}, 0.1, eos);
  CheckConvergenceOrder<MUSCL>(0.3, 0.2, L, R, "toro_1_", 0.84, 0.39, 0.91, "muscl");
}

TEST(Toro_1d_MUSCL, Test_2) {
  auto L = State::FromPrimitive(1.0, {-2.0}, 0.4, eos);
  auto R = State::FromPrimitive(1.0, {2.0}, 0.4, eos);
  CheckConvergenceOrder<MUSCL>(0.5, 0.15, L, R, "toro_2_", 0.63, 0.46, 0.96, "muscl");
}

TEST(Toro_1d_MUSCL, Test_3) {
  auto L = State::FromPrimitive(1.0, {0.0}, 1000.0, eos);
  auto R = State::FromPrimitive(1.0, {0.0}, 0.01, eos);
  CheckConvergenceOrder<MUSCL>(0.5, 0.012, L, R, "toro_3_", 0.79, 0.39, 0.88, "muscl");
}

TEST(Toro_1d_MUSCL, Test_4) {
  auto L = State::FromPrimitive(5.99924, {19.5975}, 460.894, eos);
  auto R = State::FromPrimitive(5.99242, {-6.19633}, 46.0950, eos);
  CheckConvergenceOrder<MUSCL>(0.4, 0.035, L, R, "toro_4_", 0.49, 0.29, 0.58, "muscl");
}

TEST(Toro_1d_MUSCL, Test_5) {
  auto L = State::FromPrimitive(1.0, {-19.59745}, 1000.0, eos);
  auto R = State::FromPrimitive(1.0, {-19.59745}, 0.01, eos);
  CheckConvergenceOrder<MUSCL>(0.8, 0.012, L, R, "toro_5_", 0.86, 0.40, 0.26, "muscl");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
