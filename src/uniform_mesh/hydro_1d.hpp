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

#ifndef ETHON_UNIFORM_MESH_HDYRO_1D_HPP_
#define ETHON_UNIFORM_MESH_HDYRO_1D_HPP_

#include "uniform_mesh/boundary.hpp"
#include "state.hpp"
#include "uniform_mesh/uniform_mesh.hpp"

template <size_t NUM_GHOST>
class Hydro1D {
public:
  using State = State_u<1>;
  using StateData = std::vector<State>;
  using Mesh = Mesh_u<1>;

  static constexpr size_t num_ghost = NUM_GHOST;

  template <typename EOS>
  static void Dump(const Mesh &mesh,
      const EOS &eos,
      const StateData &data,
      const double t,
      const std::string &fname) {
    FILE *f = fopen(fname.c_str(), "w");
    fprintf(f, "# time = %.8e\n", t);
    fprintf(f, "# [1] = x\n");
    fprintf(f, "# [2] = rho\n");
    fprintf(f, "# [3] = mu\n");
    fprintf(f, "# [4] = epsilon\n");
    fprintf(f, "# [5] = u\n");
    fprintf(f, "# [6] = specific internal energy\n");
    fprintf(f, "# [7] = pressure\n");
    fprintf(f, "# [8] = sound speed\n");

    // data must either be the same as the number of points in the mesh or with the correct number
    // of ghost zones
    size_t offset = 0;
    if (data.size() > mesh.N()[0]) {
      if (data.size() != mesh.N()[0] + 2 * NUM_GHOST)
        throw std::invalid_argument("Dump called with an unexpected data size");
      offset = NUM_GHOST;
    }

    for (size_t i = offset; i < mesh.N()[0] + offset; ++i) {
      double x = mesh.x({i - offset})[0];
      double u = data[i].u(0);
      double e = data[i].specific_internal_energy();
      double p = eos.pressure(data[i]);
      double cs = eos.sound_speed(data[i]);
      fprintf(f,
          "%16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e  %16.8e\n",
          x,
          data[i].rho,
          data[i].mu[0],
          data[i].epsilon,
          u,
          e,
          p,
          cs);
    }
    fclose(f);
  }
};

#endif // ETHON_UNIFORM_MESH_HDYRO_1D_HPP_
