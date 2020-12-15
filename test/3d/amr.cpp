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

#include <chrono>
#include <cstdio>
#include <omp.h>
#include <random>
#include <sstream>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "../regression.hpp"
#include "../test_utils.hpp"

#include "hllc.hpp"
#include "ideal_gas_eos.hpp"
#include "slopes.hpp"

#include "block_mesh/amr_driver.hpp"

using Mesh = Mesh_u<3>;
using State = State_u<3>;
using StateData = Kokkos::View<State ***, Kokkos::HostSpace>;
using EOS = IdealGas<State>;
using HLLC = HLLC_u<State, EOS>;

constexpr auto PR = BoundaryConditionType::Periodic;
constexpr auto OF = BoundaryConditionType::Outflow;

const EOS eos(1.4); // gamma = 1.4
const double cfl = 0.9 / 3;

template <typename RIEMANN, SlopeType SLOPE, size_t BLOCK_SIZE>
void RunKH_Block_AMR(const std::string &output_prefix,
    size_t Nx,
    size_t Ny,
    size_t Nz,
    int num_threads,
    double t_final,
    bool is_movie = false) {
  const double amr_threshold = 0.5;
  const int amr_max_level =
      2; // FOR COMPARISON TO ATHENA++, THIS MUST BE 1 LESS THAN ATHENA++ max num levels
  const int amr_derefinement_count = 5;

  double x_limit = 1.0 * double(Nx) / 128.0;
  double y_limit = 1.0 * double(Ny) / 128.0;
  double z_limit = 1.0 * double(Nz) / 128.0;
  Mesh uni_mesh({-x_limit, -y_limit, -z_limit}, {x_limit, y_limit, z_limit}, {Nx, Ny, Nz});
  auto mesh = AMRMeshInfo<BLOCK_SIZE>(uni_mesh);

  // double t_final = 2.0;
  std::stringstream iss;
  iss << "amr_files/" << output_prefix << "_grid_" << Nx << "x" << Ny << "x" << Nz << "_bs_"
      << BLOCK_SIZE << "_threads_" << num_threads;
  auto out_file = iss.str();

  BoundaryConditionType BCs[3][2] = {{PR, PR}, {PR, PR}, {PR, PR}};
  // BoundaryConditionType BCs[3][2] = {{OF, OF}, {OF, OF}, {OF, OF}};

  if (num_threads > 0) omp_set_num_threads(num_threads);

  AMRDriver<RIEMANN, SLOPE, BLOCK_SIZE> driver;
  using Block_t = typename decltype(driver)::Block_t;

  typename decltype(driver)::GeneratorInitialData init_data;

  const double rho0 = 1.0;
  const double rho_ratio = 2.0; // density raio between low and high density region
  const double v_flow = 0.5;    // flow velocity in x-direction
  const double v_ran_amp = 0.2; // amplitude of random velocity perturbations
  const double p0 = 2.5;        // initial uniform pressure

  std::mt19937_64 rng(
      is_movie ? std::chrono::high_resolution_clock::now().time_since_epoch().count() : 420);
  std::uniform_real_distribution<double> uniform(-0.5, 0.5);

  init_data.generator = [&](Block_t *block) {
    for (size_t i = 0; i < BLOCK_SIZE; ++i) {
      for (size_t j = 0; j < BLOCK_SIZE; ++j) {
        for (size_t k = 0; k < BLOCK_SIZE; ++k) {
          double y = block->GetCellCenterCoordsFromLogicalIndices(i, j, k)[1];
          bool inner = (fabs(y) < 0.25);

          double rho = rho0 * (inner ? rho_ratio : 1.0);
          double ux = (v_flow + v_ran_amp * uniform(rng)) * (inner ? -1.0 : 1.0);
          double uy = v_ran_amp * uniform(rng);
          double uz = 0.0; // v_ran_amp * uniform(rng);

          block->SetStateLogicalIndices(i, j, k, State::FromPrimitive(rho, {ux, uy, uz}, p0, eos));
        }
      }
    }
  };

  auto hist_file = out_file + ".hist";
  auto fout = fopen(hist_file.c_str(), "w");
  fprintf(fout, "# [1] = step number\n");
  fprintf(fout, "# [2] = time [s]\n");
  fprintf(fout, "# [3] = time step [s]\n");
  fprintf(fout, "# [4] = number of blocks\n");

  double output_time = 0.0;
  double output_dt = 0.005;

  auto hook = decltype(driver)::no_output();
  hook = [&](const size_t num_steps,
             const double time,
             const double dt,
             const AMRMeshInfo<BLOCK_SIZE> &mesh,
             const decltype(driver) &driver) {
    if ((time >= output_time) || (time == t_final)) {
      char buffer[16];
      sprintf(buffer, "%06lu", num_steps);
      driver.DumpGrid(out_file + "." + std::string(buffer), num_steps, time, mesh, eos, is_movie);
      output_time += output_dt;
    }
    fprintf(fout, "%6lu  %18.10E  %18.10E  %8lu\n", num_steps, time, dt, driver.blocks().size());
  };

  auto N_ghost = Block_t::N_ghost;
  auto refine = Block_t::no_refine();
  refine = [=](const Block_t *block, const double /*time*/) {
    double max = 0.0;

    const auto &rho = block->rho();
    const auto &mu0 = block->mu0();
    const auto &mu1 = block->mu1();

    for (uint k = N_ghost; k < BLOCK_SIZE + N_ghost; ++k) {
      for (uint j = N_ghost; j < BLOCK_SIZE + N_ghost; ++j) {
        for (uint i = N_ghost; i < BLOCK_SIZE + N_ghost; ++i) {
          double vgy =
              fabs(mu1(i + 1, j, k) / rho(i + 1, j, k) - mu1(i - 1, j, k) / rho(i - 1, j, k)) * 0.5;
          double vgx =
              fabs(mu0(i, j + 1, k) / rho(i, j + 1, k) - mu0(i, j - 1, k) / rho(i, j - 1, k)) * 0.5;

          double vg = sqrt(vgx * vgx + vgy * vgy);

          max = std::max(max, vg);
        }
      }
    }

    // printf("max = %.4f\n", max);

    if (max < 0.5 * amr_threshold) {
      return -1;
    } else if (max > amr_threshold) {
      return 1;
    } else {
      return 0;
    }
  };

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

  fclose(fout);
}

TEST(AMR, KH_8) { RunKH_Block_AMR<HLLC, SlopeType::MC, 8>("amr_mc_8", 256, 128, 8, 0, 0.025); }

TEST(AMR, Blast_8) {
  const int BLOCK_SIZE = 8;
  const std::string output_prefix = "amr_blast_8";
  const double amr_threshold = 3.0;
  const int amr_max_level = 2; // ONE LESS THAN ATHENA++ VALUE
  const int amr_derefinement_count = 5;

  size_t N = 32;
  Mesh uni_mesh({-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}, {N, N, N});
  auto mesh = AMRMeshInfo<BLOCK_SIZE>(uni_mesh);

  double t_final = 1.0;
  auto out_file = "amr_files/" + output_prefix + "_" + std::to_string(N);

  BoundaryConditionType BCs[3][2] = {{OF, OF}, {OF, OF}, {OF, OF}};
  AMRDriver<HLLC, SlopeType::MC, BLOCK_SIZE> driver; // (out_file + ".log");
  using Block_t = typename decltype(driver)::Block_t;

  typename decltype(driver)::GeneratorInitialData init_data;

  const double rho0 = 1.0;
  const double r0 = 0.1;            // initial radius
  const double press_out = 0.1;     // pressure outside
  const double press_ratio = 100.0; // pressure ratio inside to outside

  init_data.generator = [&](Block_t *block) {
    for (size_t i = 0; i < BLOCK_SIZE; ++i) {
      for (size_t j = 0; j < BLOCK_SIZE; ++j) {
        for (size_t k = 0; k < BLOCK_SIZE; ++k) {
          auto x = block->GetCellCenterCoordsFromLogicalIndices(i, j, k);
          double r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
          double rho = rho0;
          double ux = 0.0;
          double uy = 0.0;
          double uz = 0.0;
          double p = press_out;

          if (r < r0) p *= press_ratio;

          block->SetStateLogicalIndices(i, j, k, State::FromPrimitive(rho, {ux, uy, uz}, p, eos));
        }
      }
    }
  };

  auto hist_file = out_file + ".hist";
  auto fout = fopen(hist_file.c_str(), "w");
  fprintf(fout, "# [1] = step number\n");
  fprintf(fout, "# [2] = time [s]\n");
  fprintf(fout, "# [3] = time step [s]\n");
  fprintf(fout, "# [4] = number of blocks\n");

  double output_time = 0.0;
  double output_dt = 0.01;

  auto hook = decltype(driver)::no_output();
  hook = [&](const size_t num_steps,
             const double time,
             const double dt,
             const AMRMeshInfo<BLOCK_SIZE> &mesh,
             const decltype(driver) &driver) {
    if ((time >= output_time) || (time == t_final)) {
      char buffer[16];
      sprintf(buffer, "%06lu", num_steps);
      driver.DumpGrid(out_file + "." + std::string(buffer), num_steps, time, mesh, eos, false);
      output_time += output_dt;
    }
    fprintf(fout, "%6lu  %18.10E  %18.10E  %8lu\n", num_steps, time, dt, driver.blocks().size());
  };

  auto N_ghost = Block_t::N_ghost;
  auto refine = Block_t::no_refine();
  refine = [=](const Block_t *block, const double /*time*/) {
    MinMaxScalar<double> minmax;
    minmax.min_val = 1.0E300;
    minmax.max_val = 0.0;

    for (size_t k = N_ghost - 1; k < BLOCK_SIZE + N_ghost + 1; ++k) {
      for (size_t j = N_ghost - 1; j < BLOCK_SIZE + N_ghost + 1; ++j) {
        for (size_t i = N_ghost - 1; i < BLOCK_SIZE + N_ghost + 1; ++i) {
          // avoid corners and edges
          int num_outside = 0;
          if ((i < N_ghost) || (i >= BLOCK_SIZE + N_ghost)) ++num_outside;
          if ((j < N_ghost) || (j >= BLOCK_SIZE + N_ghost)) ++num_outside;
          if ((k < N_ghost) || (k >= BLOCK_SIZE + N_ghost)) ++num_outside;

          // double pxm = eos.pressure(block->GetState(i - 1, j, k));
          // double pxp = eos.pressure(block->GetState(i + 1, j, k));
          // double pym = eos.pressure(block->GetState(i, j - 1, k));
          // double pyp = eos.pressure(block->GetState(i, j + 1, k));
          // double pzm = eos.pressure(block->GetState(i, j, k - 1));
          // double pzp = eos.pressure(block->GetState(i, j, k + 1));

          // double dx = 0.5 * (pxp - pxm);
          // double dy = 0.5 * (pyp - pym);
          // double dz = 0.5 * (pzp - pzm);

          // double norm = sqrt(dx * dx + dy * dy + dz * dz) / eos.pressure(block->GetState(i, j,
          // k));
          // value_to_update = std::max(value_to_update, norm);
          if (num_outside <= 1) {
            double press = eos.pressure(block->GetState(i, j, k));
            minmax.min_val = std::min(minmax.min_val, press);
            minmax.max_val = std::max(minmax.max_val, press);
          }
        }
      }
    }

    // printf("max = %.4f\n", max);
    double ratio = minmax.max_val / minmax.min_val;

    if (ratio < 0.5 * amr_threshold) {
      return -1;
    } else if (ratio > amr_threshold) {
      return 1;
    } else {
      return 0;
    }
  };

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

  fclose(fout);
}

TEST(AMR, Refine_and_derefine) {
  Mesh uni_mesh({-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}, {16, 16, 8});
  auto Ns = uni_mesh.N();

  StateData init("init", Ns[0], Ns[1], Ns[2]);

  const double rho0 = 1.0e6; // very big so sound speed is small, leading to big time step
  const double p0 = 2.5;

  for (size_t i = 0; i < Ns[0]; ++i) {
    for (size_t j = 0; j < Ns[1]; ++j) {
      for (size_t k = 0; k < Ns[2]; ++k) {
        init(i, j, k) = State::FromPrimitive(rho0, {0.0, 0.0, 0.0}, p0, eos);
      }
    }
  }

  auto mesh = AMRMeshInfo<8>(uni_mesh);

  double t_final = 2.5;
  std::string out_file = "amr_files/refine";

  BoundaryConditionType BCs[3][2] = {{PR, PR}, {PR, PR}, {PR, PR}};
  AMRDriver<HLLC, SlopeType::MC, 8> driver(out_file + ".log");

  using Block_t = decltype(driver)::Block_t;
  auto N = Block_t::N;
  auto N_ghost = Block_t::N_ghost;

  decltype(driver)::GridInitialData init_data;
  init_data.init_data = init;

  auto hook = decltype(driver)::no_output();
  hook = [&](const size_t num_steps,
             const double time,
             const double /*dt*/,
             const AMRMeshInfo<8> &mesh,
             const decltype(driver) &driver) {
    char buffer[16];
    sprintf(buffer, "%06lu", num_steps);
    driver.DumpGrid(out_file + "." + std::string(buffer), num_steps, time, mesh, eos);

    if (num_steps <= 4) {
      EXPECT_EQ(driver.blocks().size(), 2048);
    } else if (num_steps <= 7) {
      EXPECT_EQ(driver.blocks().size(), 1040);
    } else if (num_steps <= 10) {
      EXPECT_EQ(driver.blocks().size(), 1026);
    } else if (num_steps == 11) {
      EXPECT_EQ(driver.blocks().size(), 1033);
    } else if (num_steps == 12) {
      EXPECT_EQ(driver.blocks().size(), 1047);
    } else if (num_steps == 13) {
      EXPECT_EQ(driver.blocks().size(), 263);
    } else if (num_steps <= 15) {
      EXPECT_EQ(driver.blocks().size(), 193);
    } else {
      EXPECT_EQ(driver.blocks().size(), 95);
    }

    for (auto &b : driver.blocks()) {
      for (uint k = N_ghost; k < N + N_ghost; ++k) {
        for (uint j = N_ghost; j < N + N_ghost; ++j) {
          for (uint i = N_ghost; i < N + N_ghost; ++i) {
            auto state = b->GetState(i, j, k);
            EXPECT_NEAR(state.rho, rho0, 1.0e-12);
            EXPECT_NEAR(state.mu[0], 0.0, 1.0e-12);
            EXPECT_NEAR(state.mu[1], 0.0, 1.0e-12);
            EXPECT_NEAR(state.mu[2], 0.0, 1.0e-12);
            EXPECT_NEAR(eos.pressure(state), p0, 1.0e-12);
          }
        }
      }
    }
  };

  auto refine = Block_t::no_refine();
  refine = [](const Block_t *block, const double time) {
    auto lb = block->lower_bounds();
    auto bs = block->block_size();

    if (time < 0.15) {
      return 1;
    } else if (time < 0.95) {
      if ((lb[0] >= -0.5) && (lb[1] >= -0.5) && (lb[1] >= -0.5)) return -1;
    } else {
      if ((lb[0] <= 0.01) && ((lb[0] + bs[0]) > 0.01) && (lb[1] <= 0.01) &&
          ((lb[1] + bs[1]) > 0.01) && (lb[2] <= 0.01) && ((lb[2] + bs[2]) > 0.01)) {
        return 1;
      } else {
        return -1;
      }
    }

    return 0;
  };

  driver.Evolve(mesh, BCs, eos, cfl, refine, 3, 3, &init_data, 0.0, t_final, 0.1, hook);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);

  SC_CHECK_MPI(sc_MPI_Init(nullptr, nullptr));
  auto mpi_world = sc_MPI_COMM_WORLD;

  sc_init(mpi_world, 1, 1, nullptr, SC_LP_ESSENTIAL);
  p4est_init(nullptr, SC_LP_ESSENTIAL);

  auto ret = RUN_ALL_TESTS();

  SC_CHECK_MPI(sc_MPI_Finalize());
  Kokkos::finalize();

  return ret;
}
