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
#include <random>
#include <sstream>

#include <gtest/gtest.h>

#include "../regression.hpp"
#include "../test_utils.hpp"

#include "hllc.hpp"
#include "ideal_gas_eos.hpp"
#include "slopes.hpp"

#include "block_mesh/amr_driver.hpp"
#include "block_mesh/driver.hpp"
#include "uniform_mesh/godunov_3d.hpp"
#include "uniform_mesh/muscl_3d.hpp"

using Mesh = Mesh_u<3>;
using State = State_u<3>;
using StateData = Kokkos::View<State ***, Kokkos::HostSpace>;
using EOS = IdealGas<State>;
using HLLC = HLLC_u<State, EOS>;

constexpr auto PR = BoundaryConditionType::Periodic;

const EOS eos(1.4); // gamma = 1.4
const double cfl = 0.9 / 3;

const int amr_max_level = 2;
const int amr_derefinement_count = 5;

std::pair<Mesh, StateData> MakeMeshInit(const size_t N) {
  Mesh mesh({-1.0, -0.5, -0.5}, {1.0, 0.5, 0.5}, {N, N / 2, N / 2});
  auto Ns = mesh.N();

  const double rho0 = 1.0;
  const double rho_ratio = 2.0; // density raio between low and high density region
  const double v_flow = 0.5;    // flow velocity in x-direction
  const double v_ran_amp = 0.2; // amplitude of random velocity perturbations
  const double p0 = 2.5;        // initial uniform pressure

  StateData init("init", Ns[0], Ns[1], Ns[2]);

  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> uniform(-0.5, 0.5);

  for (size_t i = 0; i < Ns[0]; ++i) {
    for (size_t j = 0; j < Ns[1]; ++j) {
      for (size_t k = 0; k < Ns[2]; ++k) {
        double y = mesh.x({i, j, k})[1];
        double z = mesh.x({i, j, k})[2];
        bool inner = (sqrt(y * y + z * z) < 0.25);

        double rho = rho0 * (inner ? rho_ratio : 1.0);
        double ux = (v_flow + v_ran_amp * uniform(rng)) * (inner ? -1.0 : 1.0);
        double uy = v_ran_amp * uniform(rng);
        double uz = v_ran_amp * uniform(rng);

        init(i, j, k) = State::FromPrimitive(rho, {ux, uy, uz}, p0, eos);
      }
    }
  }

  return {mesh, init};
}

template <typename HYDRO>
void RunKH_Uniform(const std::string &output_prefix) {
  size_t N = 128;
  auto mesh_init = MakeMeshInit(N);
  auto mesh = mesh_init.first;

  double t_final = 0.5;
  auto out_file = "kelvin-helmholtz_3d_files/" + output_prefix + "_" + std::to_string(N);

  auto hook = HYDRO::no_output();
  hook = [&](const size_t num_steps, const double time, const Mesh &mesh, const StateData state) {
    if ((num_steps % 100 == 0) || (time == t_final)) {
      char buffer[16];
      sprintf(buffer, "%06lu", num_steps);
      HYDRO::DumpGrid(out_file + "." + std::string(buffer) + ".xmf", time, mesh, state, eos);
    }
  };

  auto res = HYDRO::Evolve(mesh, eos, cfl, mesh_init.second, 0.0, t_final, hook);
}

template <typename RIEMANN, SlopeType SLOPE, size_t BLOCK_SIZE>
void RunKH_Block(const std::string &output_prefix) {
  size_t N = 128;
  auto mesh_init = MakeMeshInit(N);
  auto mesh = AMRMeshInfo<BLOCK_SIZE>(mesh_init.first);
  auto init = mesh_init.second;

  double t_final = 0.5;
  auto out_file = "kelvin-helmholtz_3d_files/" + output_prefix + "_" + std::to_string(N);

  BoundaryConditionType BCs[3][2] = {{PR, PR}, {PR, PR}, {PR, PR}};
  Driver<RIEMANN, SLOPE, BLOCK_SIZE> driver;

  auto hook = decltype(driver)::no_output();
  hook = [&](const size_t num_steps,
             const double time,
             const AMRMeshInfo<BLOCK_SIZE> &mesh,
             const decltype(driver) &driver) {
    if ((num_steps % 100 == 0) || (time == t_final)) {
      char buffer[16];
      sprintf(buffer, "%06lu", num_steps);
      driver.DumpGrid(out_file + "." + std::string(buffer), num_steps, time, mesh, eos);
    }
  };

  driver.Evolve(mesh, BCs, eos, cfl, init, 0.0, t_final, hook);
}

template <typename RIEMANN, SlopeType SLOPE, size_t BLOCK_SIZE>
void RunKH_Block_AMR(const std::string &output_prefix) {
  size_t N = 128;
  auto mesh_init = MakeMeshInit(N);
  auto mesh = AMRMeshInfo<BLOCK_SIZE>(mesh_init.first);
  auto init = mesh_init.second;

  double t_final = 0.5;
  auto out_file = "kelvin-helmholtz_3d_files/" + output_prefix + "_" + std::to_string(N);

  BoundaryConditionType BCs[3][2] = {{PR, PR}, {PR, PR}, {PR, PR}};
  AMRDriver<RIEMANN, SLOPE, BLOCK_SIZE> driver(out_file + ".log");

  typename decltype(driver)::GridInitialData init_data;
  init_data.init_data = init;

  auto hook = decltype(driver)::no_output();
  hook = [&](const size_t num_steps,
             const double time,
             const double /*dt*/,
             const AMRMeshInfo<BLOCK_SIZE> &mesh,
             const decltype(driver) &driver) {
    if ((num_steps % 100 == 0) || (time == t_final)) {
      char buffer[16];
      sprintf(buffer, "%06lu", num_steps);
      driver.DumpGrid(out_file + "." + std::string(buffer), num_steps, time, mesh, eos);
    }
  };

  auto refine = decltype(driver)::Block_t::no_refine();

  // don't do AMR in this test
  driver.Evolve(mesh,
      BCs,
      eos,
      cfl,
      refine,
      amr_max_level,
      amr_derefinement_count,
      &init_data,
      0.0,
      t_final,
      1.0,
      hook);
}

TEST(Kelvin_Helmholtz_3D, Uniform_MUSCL_MC) {
  RunKH_Uniform<MUSCL3D<HLLC, SlopeType::MC, PR, PR, PR, PR, PR, PR>>("KH_3d_uniform_muscl_mc");
}

TEST(Kelvin_Helmholtz_3D, Uniform_Godunov) {
  RunKH_Uniform<Godunov3D<HLLC, PR, PR, PR, PR, PR, PR>>("KH_3d_uniform_godunov");
}

TEST(Kelvin_Helmholtz_3D, Block_MC) { RunKH_Block<HLLC, SlopeType::MC, 32>("KH_3d_block_mc"); }

TEST(Kelvin_Helmholtz_3D, Block_AMR_MC) {
  RunKH_Block_AMR<HLLC, SlopeType::MC, 32>("KH_3d_block_amr_mc");
}

template <typename DATA>
double CompareResults(const DATA a, const DATA b) {
  assert(a.extent(0) == b.extent(0));
  assert(a.extent(1) == b.extent(1));
  assert(a.extent(2) == b.extent(2));

  double max_diff = 0.0;

  parallel_reduce(
      MDRangePolicy<OpenMP, Rank<3>>({0, 0, 0}, {a.extent(0), a.extent(1), a.extent(2)}),
      KOKKOS_LAMBDA(const int &i, const int &j, const int &k, double &value_to_update) {
        for (size_t s = 0; s < State::size; ++s) {
          double diff = fabs(a(i, j, k)[s] - b(i, j, k)[s]);
          value_to_update = fmax(value_to_update, diff);
        }
      },
      Max<double>(max_diff));

  EXPECT_NEAR(max_diff, 0, 1.0E-14);

  return max_diff;
}

TEST(Kelvin_Helmholtz_3D, Compare_Uniform_Block) {
  size_t N = 128;
  auto mesh_init = MakeMeshInit(N);
  auto mesh = mesh_init.first;
  auto init = mesh_init.second;

  double t_final = 0.1;

  using HYDRO = MUSCL3D<HLLC, SlopeType::MC, PR, PR, PR, PR, PR, PR>;

  auto uniform_res =
      HYDRO::Evolve(mesh, eos, cfl, init, 0.0, t_final, HYDRO::no_output());

  BoundaryConditionType BCs[3][2] = {{PR, PR}, {PR, PR}, {PR, PR}};
  Driver<HLLC, SlopeType::MC, 64> driver_64;
  Driver<HLLC, SlopeType::MC, 32> driver_32;
  Driver<HLLC, SlopeType::MC, 16> driver_16;

  printf("Block size 64 max error: %.10e\n",
      CompareResults(uniform_res, driver_64.Evolve(mesh, BCs, eos, cfl, init, 0.0, t_final)));
  printf("Block size 32 max error: %.10e\n",
      CompareResults(uniform_res, driver_32.Evolve(mesh, BCs, eos, cfl, init, 0.0, t_final)));
  printf("Block size 16 max error: %.10e\n",
      CompareResults(uniform_res, driver_16.Evolve(mesh, BCs, eos, cfl, init, 0.0, t_final)));
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();

  Kokkos::finalize();
  return ret;
}
