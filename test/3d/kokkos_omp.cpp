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
#include <Kokkos_Core.hpp>

#include "../regression.hpp"
#include "../test_utils.hpp"

#include "hllc.hpp"
#include "ideal_gas_eos.hpp"
#include "slopes.hpp"

#include "kokkos_omp/amr_driver.hpp"

using Mesh = Mesh_u<3>;
using State = State_u<3>;
using StateData = Kokkos::View<State ***>;
using EOS = IdealGas<State>;
using HLLC = HLLC_u<State, EOS>;

constexpr auto PR = BoundaryConditionType::Periodic;
constexpr auto OF = BoundaryConditionType::Outflow;

const EOS eos(1.4); // gamma = 1.4
const double cfl = 0.9 / 3;

template <typename RIEMANN, SlopeType SLOPE, int BLOCK_SIZE>
void RunKH_Block_AMR(const std::string &output_prefix,
    size_t Nx,
    size_t Ny,
    size_t Nz,
    int num_threads,
    double t_final,
    bool is_movie = false) {
  const double amr_threshold = 0.5;
  const int amr_max_level =
      1; // FOR COMPARISON TO ATHENA++, THIS MUST BE 1 LESS THAN ATHENA++ max num levels
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

  AMRDriver<RIEMANN, SLOPE, BLOCK_SIZE, Kokkos::OpenMP> driver;
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
    for (uint i = 0; i < BLOCK_SIZE; ++i) {
      for (uint j = 0; j < BLOCK_SIZE; ++j) {
        for (uint k = 0; k < BLOCK_SIZE; ++k) {
          double y = block->GetCellCenterCoordsFromLogicalIndices(i, j, k)[1];
          bool inner = (fabs(y) < 0.5);

          double rho = rho0 * (inner ? rho_ratio : 1.0);
          double ux = (v_flow + v_ran_amp * uniform(rng)) * (inner ? -1.0 : 1.0);
          double uy = v_ran_amp * uniform(rng);
          double uz = 0.0; // v_ran_amp * uniform(rng);

          block->SetStateLogicalIndices(
              driver.CellData(), i, j, k, State::FromPrimitive(rho, {ux, uy, uz}, p0, eos));
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
  double output_dt = 0.1;

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

  driver.Evolve(mesh,
      BCs,
      eos,
      cfl,
      Block_t::refine_KH,
      amr_max_level,
      amr_derefinement_count,
      amr_threshold,
      &init_data,
      0.0,
      t_final,
      1.0,
      hook);

  fclose(fout);
}

TEST(AMR, Kokkos_OMP_KH_8) {
  RunKH_Block_AMR<HLLC, SlopeType::MC, 8>("amr_kokkos_omp_mc_8", 256, 128, 8, 0, 0.5);
}

void RunBlast_Block_AMR() {
  const int BLOCK_SIZE = 8;
  const std::string output_prefix = "amr_kokkos_omp_blast_8";
  const double amr_threshold = 3.0;
  const int amr_max_level = 2;
  const int amr_derefinement_count = 5;

  size_t N = 32;
  Mesh uni_mesh({-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}, {N, N, N});
  auto mesh = AMRMeshInfo<BLOCK_SIZE>(uni_mesh);

  double t_final = 1.0;
  auto out_file = "amr_files/" + output_prefix + "_" + std::to_string(N);

  BoundaryConditionType BCs[3][2] = {{OF, OF}, {OF, OF}, {OF, OF}};
  AMRDriver<HLLC, SlopeType::MC, BLOCK_SIZE, Kokkos::OpenMP> driver;
  using Block_t = typename decltype(driver)::Block_t;

  typename decltype(driver)::GeneratorInitialData init_data;

  const double rho0 = 1.0;
  const double r0 = 0.1;            // initial radius
  const double press_out = 0.1;     // pressure outside
  const double press_ratio = 100.0; // pressure ratio inside to outside

  init_data.generator = [&](Block_t *block) {
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      for (int j = 0; j < BLOCK_SIZE; ++j) {
        for (int k = 0; k < BLOCK_SIZE; ++k) {
          auto x = block->GetCellCenterCoordsFromLogicalIndices(i, j, k);
          double r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
          double rho = rho0;
          double ux = 0.0;
          double uy = 0.0;
          double uz = 0.0;
          double p = press_out;

          if (r < r0) p *= press_ratio;

          block->SetStateLogicalIndices(
              driver.CellData(), i, j, k, State::FromPrimitive(rho, {ux, uy, uz}, p, eos));
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
  double output_dt = 1e6;

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

  driver.Evolve(mesh,
      BCs,
      eos,
      cfl,
      Block_t::refine_Blast,
      amr_max_level,
      amr_derefinement_count,
      amr_threshold,
      &init_data,
      0.0,
      t_final,
      1.0,
      hook);

  fclose(fout);
}

TEST(AMR, Kokkos_OMP_Blast_8) { RunBlast_Block_AMR(); }

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
