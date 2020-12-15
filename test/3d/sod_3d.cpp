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
#include <Kokkos_Core.hpp>

#include "../regression.hpp"
#include "../test_utils.hpp"

#include "hllc.hpp"
#include "ideal_gas_eos.hpp"
#include "slopes.hpp"

#include "uniform_mesh/godunov_3d.hpp"
#include "uniform_mesh/muscl_3d.hpp"

using Mesh = Mesh_u<3>;
using State = State_u<3>;
using StateData = Kokkos::View<State ***, Kokkos::HostSpace>;
using EOS = IdealGas<State>;
using HLLC = HLLC_u<State, EOS>;
using ExecSpace = Kokkos::OpenMP;

constexpr auto OF = BoundaryConditionType::Outflow;
constexpr auto PR = BoundaryConditionType::Periodic;

const EOS eos(1.4); // gamma = 1.4
const double cfl = 0.9;

template <typename HYDRO>
void SodPerpendicular(const size_t dir, const std::string name, const double athena_tol) {
  const size_t N = 256;
  const size_t N_other = 10;

  auto L = State::FromPrimitive(1.0, {0.0, 0.0, 0.0}, 1.0, eos);
  auto R = State::FromPrimitive(0.125, {0.0, 0.0, 0.0}, 0.1, eos);
  const double x0 = 0.5;

  Array<size_t, 3> Ns = {N_other, N_other, N_other};
  Ns[dir] = N;
  Mesh mesh({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, Ns);
  Mesh_u<1> mesh_1d({0.0}, {1.0}, {N});
  StateData init("init", Ns[0], Ns[1], Ns[2]);

  parallel_for(
      MDRangePolicy<ExecSpace, Rank<3>>({0, 0, 0}, {Ns[0], Ns[1], Ns[2]}),
      KOKKOS_LAMBDA(const size_t &i, const size_t &j, const size_t &k) {
        if (mesh.x({i, j, k})[dir] < x0)
          init(i, j, k) = L;
        else
          init(i, j, k) = R;
      });

  auto res = HYDRO::Evolve(mesh, eos, cfl, init, 0.0, 0.3);

  EXPECT_EQ(res.extent(0), Ns[0]);
  EXPECT_EQ(res.extent(1), Ns[1]);
  EXPECT_EQ(res.extent(2), Ns[2]);

  // check that the solution is the same in the planes perpendicular to the given direction
  std::vector<State> res_1d(N);
  std::vector<State_u<1>> res_1d_1vel(N);
  double max_diff = 0.0;

  for (size_t p = 0; p < N; ++p) {
    State ref;
    if (dir == 0)
      ref = res(p, 0, 0);
    else if (dir == 1)
      ref = res(0, p, 0);
    else if (dir == 2)
      ref = res(0, 0, p);
    else
      throw std::runtime_error("Invalid direction");

    State_u<1> ref_1d;
    ref_1d.rho = ref.rho;
    ref_1d.epsilon = ref.epsilon;
    ref_1d.mu[0] = ref.mu[dir];
    res_1d_1vel[p] = ref_1d;

    res_1d[p] = ref;

    for (size_t o1 = 0; o1 < N_other; ++o1) {
      for (size_t o2 = 0; o2 < N_other; ++o2) {
        State s;
        if (dir == 0)
          s = res(p, o1, o2);
        else if (dir == 1)
          s = res(o1, p, o2);
        else if (dir == 2)
          s = res(o1, o2, p);

        for (size_t c = 0; c < State::size; ++c) {
          max_diff = std::max(max_diff, fabs(ref[c] - s[c]));
        }
      }
    }
  }

  printf("max diff = %.10e\n", max_diff);
  EXPECT_EQ(max_diff, 0.0) << "Differences in perpendicular direction to shock";

  auto err = RegressionTest("sod_3d_files/" + name, mesh_1d, res_1d, false);
  EXPECT_LE(err, 1.0e-12) << "Regression error too big: " << err;

  CompareAthenaResults("sod_3d_files/sod_athena_1d", mesh_1d, res_1d_1vel, athena_tol);
}

TEST(Sod_3D, Godunov_Perpendicular) {
  SodPerpendicular<Godunov3D<HLLC, OF, OF, PR, PR, PR, PR>>(0, "godunov_sod_3d_x", 0.1);
  SodPerpendicular<Godunov3D<HLLC, PR, PR, OF, OF, PR, PR>>(1, "godunov_sod_3d_y", 0.1);
  SodPerpendicular<Godunov3D<HLLC, PR, PR, PR, PR, OF, OF>>(2, "godunov_sod_3d_z", 0.1);
}

TEST(Sod_3D, MUSCL_MC_Perpendicular) {
  SodPerpendicular<MUSCL3D<HLLC, SlopeType::MC, OF, OF, PR, PR, PR, PR>>(
      0, "muscl_MC_sod_3d_x", 0.01);
  SodPerpendicular<MUSCL3D<HLLC, SlopeType::MC, PR, PR, OF, OF, PR, PR>>(
      1, "muscl_MC_sod_3d_y", 0.01);
  SodPerpendicular<MUSCL3D<HLLC, SlopeType::MC, PR, PR, PR, PR, OF, OF>>(
      2, "muscl_MC_sod_3d_z", 0.01);
}

TEST(Sod_3D, MUSCL_SuperBee_Perpendicular) {
  SodPerpendicular<MUSCL3D<HLLC, SlopeType::SuperBee, OF, OF, PR, PR, PR, PR>>(
      0, "muscl_SB_sod_3d_x", 0.025);
  SodPerpendicular<MUSCL3D<HLLC, SlopeType::SuperBee, PR, PR, OF, OF, PR, PR>>(
      1, "muscl_SB_sod_3d_y", 0.025);
  SodPerpendicular<MUSCL3D<HLLC, SlopeType::SuperBee, PR, PR, PR, PR, OF, OF>>(
      2, "muscl_SB_sod_3d_z", 0.025);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();

  Kokkos::finalize();
  return ret;
}
