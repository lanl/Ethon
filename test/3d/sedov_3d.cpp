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

// #include <H5Cpp.h>
#include <gtest/gtest.h>

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

constexpr auto PR = BoundaryConditionType::Periodic;

const EOS eos(1.4); // gamma = 1.4
const double cfl = 0.9 / 3;

std::pair<Mesh, StateData> MakeMeshInit(const size_t N) {
  Mesh mesh({-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}, {N, N, N});
  auto dx = mesh.cell_size();

  const double rho0 = 1.0;
  const double p0 = 1.0e-4;
  const double Eblast = 0.8;
  const double r0 = 0.1;

  StateData init("init", N, N, N);
  auto init_state = State::FromPrimitive(rho0, {0.0, 0.0, 0.0}, p0, eos);

  parallel_for(
      MDRangePolicy<ExecSpace, Rank<3>>({0, 0, 0}, {N, N, N}),
      KOKKOS_LAMBDA(const int &i, const int &j, const int &k) {
        auto x = mesh.x({size_t(i), size_t(j), size_t(k)});
        double r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        init(i, j, k) = init_state;

        if (r < r0) {
          // double p_before = eos.pressure(init(i, j, k));
          init(i, j, k).epsilon = Eblast / (dx[0] * dx[1] * dx[2]);
          // double p_after = eos.pressure(init(i, j, k));
          // printf("[%i, %i, %i] pressure ratio = %.10e/%.10e = %.10e\n",
          //     i,
          //     j,
          //     k,
          //     p_after,
          //     p_before,
          //     p_after / p_before);
        }
      });

  return {mesh, init};
}

void CheckSymmetry(const Kokkos::View<State ***, LayoutStride, Kokkos::HostSpace> res, size_t N) {
  auto diff_func = KOKKOS_LAMBDA(const State &Ul, const State &Ur, const Array<bool, 3> flip_vel) {
    double diff = 0.0;
    diff = fmax(diff, fabs(Ul.rho - Ur.rho));
    diff = fmax(diff, Ul.epsilon - Ur.epsilon);
    for (size_t d = 0; d < State::dim; ++d) {
      if (flip_vel[d])
        diff = fmax(diff, fabs(Ul.mu[d] - (-Ur.mu[d])));
      else
        diff = fmax(diff, fabs(Ul.mu[d] - Ur.mu[d]));
    }

    return diff;
  };

  // check symmetry across xy-, xz-, and yz-planes
  double max_diff_1 = 0.0;
  parallel_reduce(
      MDRangePolicy<ExecSpace, Rank<3>>({0, 0, 0}, {N / 2, N, N}),
      KOKKOS_LAMBDA(const int &p, const int &i, const int &j, double &value_to_update) {
        double xy_diff = 0.0;
        double xz_diff = 0.0;
        double yz_diff = 0.0;
        xy_diff = diff_func(res(i, j, p), res(i, j, N - 1 - p), {false, false, true});
        xz_diff = diff_func(res(i, p, j), res(i, N - 1 - p, j), {false, true, false});
        yz_diff = diff_func(res(p, i, j), res(N - 1 - p, i, j), {true, false, false});

        value_to_update = fmax(value_to_update, fmax(xy_diff, fmax(xz_diff, yz_diff)));
      },
      Max<double>(max_diff_1));

  // check symmetry across x-, y-, and z-axes
  double max_diff_2 = 0.0;
  parallel_reduce(
      MDRangePolicy<ExecSpace, Rank<3>>({0, 0, 0}, {N / 2, N / 2, N}),
      KOKKOS_LAMBDA(const int &p, const int &q, const int &i, double &value_to_update) {
        double x_diff = 0.0;
        double y_diff = 0.0;
        double z_diff = 0.0;
        x_diff = diff_func(res(i, p, q), res(i, N - 1 - p, N - 1 - q), {false, true, true});
        y_diff = diff_func(res(p, i, q), res(N - 1 - p, i, N - 1 - q), {true, false, true});
        z_diff = diff_func(res(p, q, i), res(N - 1 - p, N - 1 - q, i), {true, true, false});

        value_to_update = fmax(value_to_update, fmax(x_diff, fmax(y_diff, z_diff)));
      },
      Max<double>(max_diff_2));

  // check symmetry across origin
  double max_diff_3 = 0.0;
  parallel_reduce(
      MDRangePolicy<ExecSpace, Rank<3>>({0, 0, 0}, {N / 2, N / 2, N / 2}),
      KOKKOS_LAMBDA(const int &p, const int &q, const int &r, double &value_to_update) {
        double diff = 0.0;
        diff = diff_func(res(p, q, r), res(N - 1 - p, N - 1 - q, N - 1 - r), {true, true, true});

        value_to_update = fmax(value_to_update, diff);
      },
      Max<double>(max_diff_3));

  double max_diff = std::max({max_diff_1, max_diff_2, max_diff_3});
  printf("max symmetry diff = %.10e\n", max_diff);
  EXPECT_LE(max_diff, 1.0e-12) << "symmmetry violation in 3D sedov";
}

/* Comparison with Athena: MUSCL results seem to agree quite well, but not exactly, so let's not
   make this comparison part of the test.

namespace {
std::vector<double> ReadAthenaCoords(
    const H5::H5File &file, const std::string &dataset_name, const size_t N) {
  auto dataset = file.openDataSet(dataset_name);
  auto ds = dataset.getSpace();

  int rank = ds.getSimpleExtentNdims();
  if (rank != 2) throw std::invalid_argument("Expected H5 dataset to have rank 2");

  hsize_t dims[2];
  ds.getSimpleExtentDims(dims, nullptr);
  if (dims[0] != 1) throw std::invalid_argument("Expected H5 dataset to have first dimension of 1");
  if (dims[1] != N)
    throw std::invalid_argument(
        "Expected H5 dataset to have first dimension of " + std::to_string(N));

  std::vector<double> res(N);
  H5::DataSpace ms(1, &dims[1]);

  dataset.read(res.data(), H5::PredType::NATIVE_DOUBLE, ms, ds);

  return res;
}
} // namespace

void CompareAthena(
    const Mesh &mesh, const Kokkos::View<State ***, LayoutStride> res, const size_t N) {
  H5::H5File f("sedov_3d_files/athena.h5", H5F_ACC_RDONLY);

  auto xs = ReadAthenaCoords(f, "x1v", N);
  auto zs = ReadAthenaCoords(f, "x2v", N);
  auto ys = ReadAthenaCoords(f, "x3v", N);

  for (size_t i = 0; i < N; ++i) {
    auto coords = mesh.x({i, i, i});
    EXPECT_NEAR(xs[i], coords[0], 1.0E-12);
    EXPECT_NEAR(ys[i], coords[1], 1.0E-12);
    EXPECT_NEAR(zs[i], coords[2], 1.0E-12);
  }

  H5::DataSet dataset = f.openDataSet("cons");
  auto ds = dataset.getSpace();

  int rank = ds.getSimpleExtentNdims();
  if (rank != 5) throw std::invalid_argument("Expected cons to have rank 5");

  hsize_t dims[5];
  ds.getSimpleExtentDims(dims, nullptr);
  if ((dims[0] != 5) || (dims[1] != 1) || (dims[2] != N) || (dims[3] != N) || (dims[4] != N))
    throw std::invalid_argument("cons has wrong size");

  std::vector<double> athena_dat(5 * N * N * N);
  hsize_t mem_dims[4] = {5, N, N, N};
  H5::DataSpace ms(4, mem_dims);

  dataset.read(athena_dat.data(), H5::PredType::NATIVE_DOUBLE, ms, ds);

  size_t idx = (10 * N + 15) * N + 20;
  printf("Cell (0,0,0): %.10e, %.10e, %.10e, %.10e, %.10e\n",
      athena_dat[0 * N * N * N + idx],
      athena_dat[1 * N * N * N + idx],
      athena_dat[2 * N * N * N + idx],
      athena_dat[3 * N * N * N + idx],
      athena_dat[4 * N * N * N + idx]);
  Hydro3D<0>::Dump_cell(res(20, 15, 10));

  auto N_tot = res.size();
  std::vector<double> r(N_tot), rho(N_tot), u(N_tot), epsilon(N_tot);

  parallel_for(
      MDRangePolicy<ExecSpace, Rank<3>>({0, 0, 0}, {N, N, N}), [&](const int &i, const int &j, const int &k) {
        int gi = (k * N + j) * N + i;
        auto x = mesh.x({size_t(i), size_t(j), size_t(k)});
        r[gi] = sqrt(xs[i] * xs[i] + ys[j] * ys[j] + zs[k] * zs[k]);

        rho[gi] = athena_dat[0 * N * N * N + gi];
        double u0 = athena_dat[2 * N * N * N + gi];
        double u1 = athena_dat[3 * N * N * N + gi];
        double u2 = athena_dat[4 * N * N * N + gi];
        u[gi] = sqrt(u0 * u0 + u1 * u1 + u2 * u2);
        epsilon[gi] = athena_dat[1 * N * N * N + gi];
      });

  RegressionTest(
      "sedov_3d_files/athena_1d", {"r", "rho", "|u|", "epsilon"}, {r, rho, u, epsilon}, true);

  double max_error = 0.0;
  parallel_reduce(MDRangePolicy<ExecSpace, Rank<3>>({0, 0, 0}, {N, N, N}),
      [&](const int &i, const int &j, const int &k, double &value_to_update) {
        double diff = abs(athena_dat[((0 * N + i) * N + j) * N + k] - res(i, j, k).rho);
        value_to_update = std::max({value_to_update, diff});
      },
      Max<double>(max_error));

  printf("max error compared with Athena: %.10e\n", max_error);
}
*/

template <typename HYDRO>
void RunSedov(const std::string &output_prefix) {
  size_t N = 32;
  auto mesh_init = MakeMeshInit(N);
  auto mesh = mesh_init.first;
  auto Ns = mesh.N();

  double t_final = 0.5;
  auto out_file = "sedov_3d_files/" + output_prefix + "_" + std::to_string(N);

  auto hook = HYDRO::no_output();
  // hook = [&](const size_t num_steps, const double time, const Mesh &mesh, const StateData state)
  // {
  //   if ((num_steps % 10 == 0) || (time == t_final)) {
  //     char buffer[16];
  //     sprintf(buffer, "%06lu", num_steps);
  //     HYDRO::DumpGrid(out_file + "." + std::string(buffer) + ".xmf", time, mesh, state, eos);
  //   }
  // };

  auto res = HYDRO::Evolve(mesh, eos, cfl, mesh_init.second, 0.0, t_final, hook);

  CheckSymmetry(res, N);
  // CompareAthena(mesh, res, N);

  // make 1d output
  auto N_tot = res.size();
  std::vector<double> r(N_tot), rho(N_tot), u(N_tot), epsilon(N_tot), p(N_tot), e(N_tot);

  parallel_for(MDRangePolicy<ExecSpace, Rank<3>>({0, 0, 0}, {Ns[0], Ns[1], Ns[2]}),
      [&](const int &i, const int &j, const int &k) {
        int gi = (k * Ns[1] + j) * Ns[0] + i;
        auto x = mesh.x({size_t(i), size_t(j), size_t(k)});
        r[gi] = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);

        auto U = res(i, j, k);
        rho[gi] = U.rho;
        u[gi] = sqrt(U.mu_squared());
        epsilon[gi] = U.epsilon;
        p[gi] = eos.pressure(U);
        e[gi] = U.specific_internal_energy();
      });

  RegressionTest(out_file,
      {"r", "rho", "|u|", "epsilon", "pressure", "specific internal energy"},
      {r, rho, u, epsilon, p, e},
      true);
}

TEST(Sedov_3D, MUSCL_MC) {
  RunSedov<MUSCL3D<HLLC, SlopeType::MC, PR, PR, PR, PR, PR, PR>>("sedov_3d_muscl_mc");
}

TEST(Sedov_3D, MUSCL_SuperBee) {
  RunSedov<MUSCL3D<HLLC, SlopeType::SuperBee, PR, PR, PR, PR, PR, PR>>("sedov_3d_muscl_superbee");
}

TEST(Sedov_3D, Godunov) { RunSedov<Godunov3D<HLLC, PR, PR, PR, PR, PR, PR>>("sedov_3d_godunov"); }

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();

  Kokkos::finalize();
  return ret;
}
