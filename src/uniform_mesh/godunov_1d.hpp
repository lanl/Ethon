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

#ifndef ETHON_UNIFORM_MESH_GODUNOV_1D_HPP_
#define ETHON_UNIFORM_MESH_GODUNOV_1D_HPP_

#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "hydro_1d.hpp"

template <typename RIEMANN, BoundaryConditionType BDRY_L, BoundaryConditionType BDRY_U>
class Godunov1D : public Hydro1D<1> {
  static_assert(
      (BDRY_L == BoundaryConditionType::Periodic) == (BDRY_U == BoundaryConditionType::Periodic),
      "Either both or neither boundaries can be periodic");

public:
  using EOS = typename RIEMANN::EOS;

  Godunov1D() = delete;

  static StateData Evolve(const Mesh &mesh,
      const EOS &eos,
      const double cfl,
      const StateData &init,
      const double t_start,
      const double t_end) {
    const auto N = mesh.N()[0];
    const auto dx = mesh.cell_size()[0];

    RIEMANN riemann(eos);
    StateData Us(N + 2 * num_ghost);
    StateData Fs(N + 2 * num_ghost);

    // CONVENTION: upper-case index is logical index (-num_ghost, ..., -2, -1, 0, 1, ..., N-2, N-1,
    // N, N+1, ..., N+(num_ghost-1)) and lower-case is real index. The logical index L corresponds
    // to real index l = L + num_ghost

    double Smax = 0.0;
    for (size_t I = 0; I < N; ++I) {
      auto i = I + num_ghost;
      Us[i] = init[I];
      Smax = std::max(Smax, fabs(Us[i].u(0)) + eos.sound_speed(Us[i]));
    }
    double dt = cfl * dx / Smax * 0.2;
    dt = std::min(dt, t_end - t_start);

    // double out_t = t_start;
    // int out_cnt = 0;
    double t = t_start;
    size_t num_steps = 0;
    // printf("[%06lu] t = %.10e, dt = %.10e\n", num_steps, t, dt);
    // Dump(mesh, eos, Us, t, "out_" + std::to_string(num_steps));

    while (t < t_end) {
      Boundary1D<State, num_ghost, BDRY_L, Side::Lower>::Apply(Us);
      Boundary1D<State, num_ghost, BDRY_U, Side::Upper>::Apply(Us);

      // compute fluxes
      for (size_t I = 0; I < N + 1; ++I) {
        auto i = I + num_ghost;
        Fs[i] = riemann(0, Us[i - 1], Us[i]);
      }

      // update state
      double Smax = 0.0;
      for (size_t I = 0; I < N; ++I) {
        auto i = I + num_ghost;
        Us[i] += (Fs[i] - Fs[i + 1]) * dt / dx;
        Smax = std::max(Smax, fabs(Us[i].u(0)) + eos.sound_speed(Us[i]));
      }

      // if (t >= out_t) {
      //   auto str = std::to_string(out_cnt);
      //   std::string lead = "";
      //   for (size_t i = 0; i < 5 - str.size(); ++i)
      //     lead += "0";
      //   Dump(mesh, eos, Us, t, "godunov_" + std::to_string(N) + "_out_" + lead + str);
      //   out_t += 0.005;
      //   ++out_cnt;
      // }

      t += dt;
      ++num_steps;
      dt = cfl * dx / Smax;
      if (num_steps <= 5) dt *= 0.2;
      dt = std::min(dt, t_end - t);
      // dt = std::min(dt, out_t - t);

      // printf("[%06lu] t = %.10e, dt = %.10e\n", num_steps, t, dt);
      // Dump(mesh, eos, Us, t, "godunov_" + std::to_string(N) + "_out_" + std::to_string(num_steps));
    }
    // printf("finished after %lu steps\n", num_steps);

    // return without ghost zones
    return StateData(Us.begin() + num_ghost, Us.end() - num_ghost);
  }
};

#endif // ETHON_UNIFORM_MESH_GODUNOV_1D_HPP_
