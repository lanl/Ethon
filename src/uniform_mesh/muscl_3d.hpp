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

#ifndef ETHON_UNIFORM_MESH_MUSCL_3D_FIXED_HPP_
#define ETHON_UNIFORM_MESH_MUSCL_3D_FIXED_HPP_

#include "uniform_mesh/hydro_3d.hpp"

template <typename RIEMANN,
    SlopeType SLOPE,
    BoundaryConditionType BDRY_L0,
    BoundaryConditionType BDRY_U0,
    BoundaryConditionType BDRY_L1,
    BoundaryConditionType BDRY_U1,
    BoundaryConditionType BDRY_L2,
    BoundaryConditionType BDRY_U2>
class MUSCL3D : public Hydro3D<2> {
  static_assert(
      (BDRY_L0 == BoundaryConditionType::Periodic) == (BDRY_U0 == BoundaryConditionType::Periodic),
      "Either both or neither boundaries can be periodic in x-direction");
  static_assert(
      (BDRY_L1 == BoundaryConditionType::Periodic) == (BDRY_U1 == BoundaryConditionType::Periodic),
      "Either both or neither boundaries can be periodic in y-direction");
  static_assert(
      (BDRY_L2 == BoundaryConditionType::Periodic) == (BDRY_U2 == BoundaryConditionType::Periodic),
      "Either both or neither boundaries can be periodic in z-direction");

public:
  using EOS = typename RIEMANN::EOS;
  using StateDataHost = Kokkos::View<State ***, Kokkos::HostSpace>;

  MUSCL3D() = delete;

  static StateData Evolve(const Mesh &mesh,
      const EOS &eos,
      const double cfl,
      const StateData &init,
      const double t_start,
      const double t_end,
      output_func_t hook = no_output()) {
    const Array<int64_t, 3> N = {int64_t(mesh.N()[0]), int64_t(mesh.N()[1]), int64_t(mesh.N()[2])};
    const auto cell_size = mesh.cell_size();
    RIEMANN riemann(eos);

    // cell-averaged state vector in each cell
    StateDataHost state(
        "state", N[0] + 2 * num_ghost, N[1] + 2 * num_ghost, N[2] + 2 * num_ghost);

    // half-step evolved lower and upper boundary states in each direction
    using State3D = State ***; // Hack because editor is confused by ***[3]
    View<State3D[3], HostSpace> lower_boundary_state(
        "lower_boundary_state", N[0] + 2 * num_ghost, N[1] + 2 * num_ghost, N[2] + 2 * num_ghost);
    View<State3D[3], HostSpace> upper_boundary_state(
        "upper_boundary_state", N[0] + 2 * num_ghost, N[1] + 2 * num_ghost, N[2] + 2 * num_ghost);

    // inter-cell fluxes
    StateDataHost x_fluxes(
        "x_fluxes", N[0] + 2 * num_ghost, N[1] + 2 * num_ghost, N[2] + 2 * num_ghost);
    StateDataHost y_fluxes(
        "y_fluxes", N[0] + 2 * num_ghost, N[1] + 2 * num_ghost, N[2] + 2 * num_ghost);
    StateDataHost z_fluxes(
        "z_fluxes", N[0] + 2 * num_ghost, N[1] + 2 * num_ghost, N[2] + 2 * num_ghost);

    // CONVENTION: upper-case index is logical index (-num_ghost, ..., -2, -1, 0, 1, ..., N-2, N-1,
    // N, N+1, ..., N+(num_ghost-1)) and lower-case is real index. The logical index L corresponds
    // to real index l = L + num_ghost

    // initialize state and determine first time step
    double dt = std::numeric_limits<double>::max();
    parallel_reduce(
        MDRangePolicy<ExecSpace, Rank<3>>({0, 0, 0}, N),
        KOKKOS_LAMBDA(const int &I, const int &J, const int &K, double &value_to_update) {
          auto i = I + num_ghost;
          auto j = J + num_ghost;
          auto k = K + num_ghost;
          state(i, j, k) = init(I, J, K);

          double min_dt = min_dt_cell(state(i, j, k), cell_size, eos);
          value_to_update = fmin(value_to_update, min_dt);
        },
        Min<double>(dt));

    // printf("overall min_dt = %.10e\n", dt);

    // make first time step smaller because we have a discontinuity but fluid velocity is still 0
    dt *= cfl * 0.2;
    dt = std::min(dt, t_end - t_start);

    // main evolution loop
    double t = t_start; // time in s
    size_t num_steps = 0;

    hook(num_steps, t, mesh, state);
    // printf("[%06lu] t = %.10e, dt = %.10e\n", num_steps, t, dt);

    while (t < t_end) {
      Boundary3D<State, num_ghost, 0, BDRY_L0, Side::Lower>::Apply(state);
      Boundary3D<State, num_ghost, 0, BDRY_U0, Side::Upper>::Apply(state);
      Boundary3D<State, num_ghost, 1, BDRY_L1, Side::Lower>::Apply(state);
      Boundary3D<State, num_ghost, 1, BDRY_U1, Side::Upper>::Apply(state);
      Boundary3D<State, num_ghost, 2, BDRY_L2, Side::Lower>::Apply(state);
      Boundary3D<State, num_ghost, 2, BDRY_U2, Side::Upper>::Apply(state);

      // compute boundary extrapolated states
      parallel_for(
          MDRangePolicy<ExecSpace, Rank<3>>({-1, -1, -1}, {N[0] + 1, N[1] + 1, N[2] + 1}),
          KOKKOS_LAMBDA(const int &I, const int &J, const int &K) {
            auto i = I + num_ghost;
            auto j = J + num_ghost;
            auto k = K + num_ghost;

            // state of this cell
            const auto &U = state(i, j, k);

            if (state(i, j, k).specific_internal_energy() < 0.0) {
              printf("state(%li,%li,%li) has negative internal energy\n", i, j, k);
            }

            // get limited slopes using states from the lower and upper neighboring cells, if this
            // cell is at the boundary, we use this cell instead, which results in outflow boundary
            // conditions
            Array<State, 3> D;
            D[0] = Slope<SLOPE>::get(state(i - 1, j, k), U, state(i + 1, j, k));
            D[1] = Slope<SLOPE>::get(state(i, j - 1, k), U, state(i, j + 1, k));
            D[2] = Slope<SLOPE>::get(state(i, j, k - 1), U, state(i, j, k + 1));

            // boundary extrapolated states to lower and upper boundaries
            for (size_t d = 0; d < 3; ++d) {
              auto UL = U - D[d] * 0.5;
              auto UU = U + D[d] * 0.5;

              const auto FL = State::Flux(d, UL, eos.pressure(UL));
              const auto FU = State::Flux(d, UU, eos.pressure(UU));

              const auto flux_diff = (FL - FU) * 0.5 * dt / cell_size[d];

              lower_boundary_state(i, j, k, d) = UL + flux_diff;
              upper_boundary_state(i, j, k, d) = UU + flux_diff;

              // make sure intermediate states are physical
              if ((lower_boundary_state(i, j, k, d).specific_internal_energy() < 0.0) ||
                  (upper_boundary_state(i, j, k, d).specific_internal_energy() < 0.0)) {
                // printf("fixing negative energy in %i,%i,%i,%lu\n", i, j, k, d);
                lower_boundary_state(i, j, k, d) = U;
                upper_boundary_state(i, j, k, d) = U;
              }
            }
          });

      // compute fluxes
      // need a sync here (i.e. new parallel_for loop), since we are going to read
      // lower_boundary_state and upper_boundary_state from neighboring cells
      parallel_for(
          MDRangePolicy<ExecSpace, Rank<3>>({0, 0, 0}, {N[0] + 1, N[1], N[2]}),
          KOKKOS_LAMBDA(const int &I, const int &J, const int &K) {
            auto i = I + num_ghost;
            auto j = J + num_ghost;
            auto k = K + num_ghost;
            x_fluxes(i, j, k) =
                riemann(0, upper_boundary_state(i - 1, j, k, 0), lower_boundary_state(i, j, k, 0));
          });

      parallel_for(
          MDRangePolicy<ExecSpace, Rank<3>>({0, 0, 0}, {N[0], N[1] + 1, N[2]}),
          KOKKOS_LAMBDA(const int &I, const int &J, const int &K) {
            auto i = I + num_ghost;
            auto j = J + num_ghost;
            auto k = K + num_ghost;
            y_fluxes(i, j, k) =
                riemann(1, upper_boundary_state(i, j - 1, k, 1), lower_boundary_state(i, j, k, 1));
          });

      parallel_for(
          MDRangePolicy<ExecSpace, Rank<3>>({0, 0, 0}, {N[0], N[1], N[2] + 1}),
          KOKKOS_LAMBDA(const int &I, const int &J, const int &K) {
            auto i = I + num_ghost;
            auto j = J + num_ghost;
            auto k = K + num_ghost;
            z_fluxes(i, j, k) =
                riemann(2, upper_boundary_state(i, j, k - 1, 2), lower_boundary_state(i, j, k, 2));
          });

      // need to sync here so that all fluxes are available
      double new_dt = std::numeric_limits<double>::max();
      parallel_reduce(
          MDRangePolicy<ExecSpace, Rank<3>>({0, 0, 0}, N),
          KOKKOS_LAMBDA(const int &I, const int &J, const int &K, double &value_to_update) {
            auto i = I + num_ghost;
            auto j = J + num_ghost;
            auto k = K + num_ghost;
            State flux_diff;

            flux_diff += (x_fluxes(i, j, k) - x_fluxes(i + 1, j, k)) / cell_size[0];
            flux_diff += (y_fluxes(i, j, k) - y_fluxes(i, j + 1, k)) / cell_size[1];
            flux_diff += (z_fluxes(i, j, k) - z_fluxes(i, j, k + 1)) / cell_size[2];

            state(i, j, k) += flux_diff * dt;

            double min_dt = min_dt_cell(state(i, j, k), cell_size, eos);
            value_to_update = fmin(value_to_update, min_dt);
          },
          Min<double>(new_dt));

      t += dt;
      ++num_steps;
      dt = new_dt * cfl;
      if (num_steps <= 5) dt *= 0.2;
      dt = std::min(dt, t_end - t);

      hook(num_steps, t, mesh, state);

      printf("[%06lu] t = %.10e, dt = %.10e\n", num_steps, t, dt);
      // dump_cell(state(N[0] / 2, N[1] / 2, N[2] / 2));
      // dump_cell(state(1, 1, 1));
    }

    // return without ghost zones
    return subview(state,
        make_pair(num_ghost, N[0] + num_ghost),
        make_pair(num_ghost, N[1] + num_ghost),
        make_pair(num_ghost, N[2] + num_ghost));
  }
};

#endif // ETHON_UNIFORM_MESH_MUSCL_3D_FIXED_HPP_
