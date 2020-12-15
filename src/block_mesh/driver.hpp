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

#ifndef ETHON_BLOCK_MESH_DRIVER_HPP_
#define ETHON_BLOCK_MESH_DRIVER_HPP_

#include <chrono>
#include <memory>
#include <vector>

#include "block_mesh/amr_mesh_info.hpp"
#include "block_mesh/block.hpp"
#include "block_mesh/face.hpp"
#include "block_mesh/output.hpp"

template <typename RIEMANN, SlopeType SLOPE, size_t BLOCK_SIZE>
class Driver {
public:
  using State = State_u<3>;
  using StateData = Kokkos::View<State ***, LayoutStride, Kokkos::HostSpace>;
  using Mesh = AMRMeshInfo<BLOCK_SIZE>;
  using output_func_t =
      std::function<void(const size_t, const double, const Mesh &, const Driver &)>;
  using EOS = typename RIEMANN::EOS;

  using Block_t = Block<EOS, SLOPE, BLOCK_SIZE>;

  using Clock = std::chrono::high_resolution_clock;

  static output_func_t no_output() {
    return [](const size_t, const double, const Mesh &, const Driver &) { return; };
  }

  Driver() = default;

  StateData Evolve(const Mesh &mesh,
      BoundaryConditionType boundary_conditions[3][2],
      const EOS &eos,
      const double cfl,
      const StateData &init,
      const double t_start,
      const double t_end,
      output_func_t hook = no_output()) {
    double time_setup = 0.0;
    double time_ghost_exchange = 0.0;
    double time_hydro = 0.0;
    double time_hook = 0.0;

    auto elapsed = [](const Clock::time_point start) {
      return std::chrono::duration_cast<std::chrono::duration<double>>(Clock::now() - start)
          .count();
    };

    auto beginning = Clock::now();

    const auto N_blocks = mesh.num_blocks();
    const auto cell_size = mesh.cell_size();
    RIEMANN riemann(eos);

    // flag indicating periodic boundary conditions in each direction
    auto check_periodic = [&](const size_t dir) {
      if (boundary_conditions[dir][0] == BoundaryConditionType::Periodic) {
        assert(boundary_conditions[dir][1] == BoundaryConditionType::Periodic);
        return true;
      } else {
        assert(boundary_conditions[dir][1] != BoundaryConditionType::Periodic);
        return false;
      }
    };

    bool x_periodic = check_periodic(0);
    bool y_periodic = check_periodic(1);
    bool z_periodic = check_periodic(2);

    // set sizes of block and faces vectors (if we have periodic boundary conditions in one
    // direction, we need one fewer face in that direction because we use an IntercellFace to handle
    // the periodic boundary condition)
    blocks_ = std::vector<Block_t>(N_blocks[0] * N_blocks[1] * N_blocks[2]);
    x_faces_.resize((N_blocks[0] + (x_periodic ? 0 : 1)) * N_blocks[1] * N_blocks[2]);
    y_faces_.resize(N_blocks[0] * (N_blocks[1] + (y_periodic ? 0 : 1)) * N_blocks[2]);
    z_faces_.resize(N_blocks[0] * N_blocks[1] * (N_blocks[2] + (z_periodic ? 0 : 1)));

    auto block_idx = [=](size_t i, size_t j, size_t k) {
      return (k * N_blocks[1] + j) * N_blocks[0] + i;
    };
    auto x_face_idx = [=](size_t i, size_t j, size_t k) {
      return (k * N_blocks[1] + j) * (N_blocks[0] + (x_periodic ? 0 : 1)) + i;
    };
    auto y_face_idx = [=](size_t i, size_t j, size_t k) {
      return (k * (N_blocks[1] + (y_periodic ? 0 : 1)) + j) * N_blocks[0] + i;
    };
    auto z_face_idx = [=](size_t i, size_t j, size_t k) {
      return (k * N_blocks[1] + j) * N_blocks[0] + i;
    };

    // construct x-faces
    for (size_t j = 0; j < N_blocks[1]; ++j) {
      for (size_t k = 0; k < N_blocks[2]; ++k) {
        if (x_periodic) {
          x_faces_[x_face_idx(0, j, k)] =
              std::shared_ptr<Face<Block_t>>(new IntercellFace<Block_t, 0>());
        } else {
          x_faces_[x_face_idx(0, j, k)] =
              CreateBoundaryFace<Block_t, 0, Side::Lower>(boundary_conditions[0][0]);
          x_faces_[x_face_idx(N_blocks[0], j, k)] =
              CreateBoundaryFace<Block_t, 0, Side::Upper>(boundary_conditions[0][1]);
        }

        for (size_t i = 1; i < N_blocks[0]; ++i) {
          x_faces_[x_face_idx(i, j, k)] =
              std::shared_ptr<Face<Block_t>>(new IntercellFace<Block_t, 0>());
        }
      }
    }

    // construct y-faces
    for (size_t i = 0; i < N_blocks[0]; ++i) {
      for (size_t k = 0; k < N_blocks[2]; ++k) {
        if (y_periodic) {
          y_faces_[y_face_idx(i, 0, k)] =
              std::shared_ptr<Face<Block_t>>(new IntercellFace<Block_t, 1>());
        } else {
          y_faces_[y_face_idx(i, 0, k)] =
              CreateBoundaryFace<Block_t, 1, Side::Lower>(boundary_conditions[1][0]);
          y_faces_[y_face_idx(i, N_blocks[1], k)] =
              CreateBoundaryFace<Block_t, 1, Side::Upper>(boundary_conditions[1][1]);
        }

        for (size_t j = 1; j < N_blocks[1]; ++j) {
          y_faces_[y_face_idx(i, j, k)] =
              std::shared_ptr<Face<Block_t>>(new IntercellFace<Block_t, 1>());
        }
      }
    }

    // construct z-faces
    for (size_t i = 0; i < N_blocks[0]; ++i) {
      for (size_t j = 0; j < N_blocks[1]; ++j) {
        if (z_periodic) {
          z_faces_[z_face_idx(i, j, 0)] =
              std::shared_ptr<Face<Block_t>>(new IntercellFace<Block_t, 2>());
        } else {
          z_faces_[z_face_idx(i, j, 0)] =
              CreateBoundaryFace<Block_t, 2, Side::Lower>(boundary_conditions[2][0]);
          z_faces_[z_face_idx(i, j, N_blocks[2])] =
              CreateBoundaryFace<Block_t, 2, Side::Upper>(boundary_conditions[2][1]);
        }

        for (size_t k = 1; k < N_blocks[2]; ++k) {
          z_faces_[z_face_idx(i, j, k)] =
              std::shared_ptr<Face<Block_t>>(new IntercellFace<Block_t, 2>());
        }
      }
    }

    // make sure all faces exist
    for (size_t i = 0; i < x_faces_.size(); ++i) {
      if (!x_faces_[i]) throw std::runtime_error("x_faces_ is NULL at " + std::to_string(i));
    }
    for (size_t i = 0; i < y_faces_.size(); ++i) {
      if (!y_faces_[i]) throw std::runtime_error("y_faces_ is NULL at " + std::to_string(i));
    }
    for (size_t i = 0; i < z_faces_.size(); ++i) {
      if (!z_faces_[i]) throw std::runtime_error("z_faces_ is NULL at " + std::to_string(i));
    }

    // construct blocks and connect them to the faces
    for (size_t i = 0; i < N_blocks[0]; ++i) {
      for (size_t j = 0; j < N_blocks[1]; ++j) {
        for (size_t k = 0; k < N_blocks[2]; ++k) {
          size_t i_upper = i + 1;
          if (x_periodic && (i == N_blocks[0] - 1)) {
            i_upper = 0;
          }

          size_t j_upper = j + 1;
          if (y_periodic && (j == N_blocks[1] - 1)) {
            j_upper = 0;
          }

          size_t k_upper = k + 1;
          if (z_periodic && (k == N_blocks[2] - 1)) {
            k_upper = 0;
          }

          auto x_lower = x_faces_[x_face_idx(i, j, k)];
          auto x_upper = x_faces_[x_face_idx(i_upper, j, k)];
          auto y_lower = y_faces_[y_face_idx(i, j, k)];
          auto y_upper = y_faces_[y_face_idx(i, j_upper, k)];
          auto z_lower = z_faces_[z_face_idx(i, j, k)];
          auto z_upper = z_faces_[z_face_idx(i, j, k_upper)];

          auto block = &blocks_[block_idx(i, j, k)];
          block->Init(
              mesh.CellLowerBounds({i * BLOCK_SIZE, j * BLOCK_SIZE, k * BLOCK_SIZE}), cell_size, 0);

          x_lower->ConnectBlock(Side::Upper, block);
          x_upper->ConnectBlock(Side::Lower, block);
          y_lower->ConnectBlock(Side::Upper, block);
          y_upper->ConnectBlock(Side::Lower, block);
          z_lower->ConnectBlock(Side::Upper, block);
          z_upper->ConnectBlock(Side::Lower, block);

          block->SetState(subview(init,
              make_pair(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE),
              make_pair(j * BLOCK_SIZE, (j + 1) * BLOCK_SIZE),
              make_pair(k * BLOCK_SIZE, (k + 1) * BLOCK_SIZE)));
        }
      }
    }

    // compute first time step size
    double dt = std::numeric_limits<double>::max();

    // for some reason OMP here segfaults, don't understand why
    //#pragma omp parallel for reduction(min : dt)
    for (auto &b : blocks_) {
      dt = std::min(dt, b.MinDt(eos));
    }

    // make first time step smaller because we have a discontinuity but fluid velocity is still 0
    dt *= cfl * 0.2;
    dt = std::min(dt, t_end - t_start);

    // main evolution loop
    double t = t_start; // time in s
    size_t num_steps = 0;
    time_setup += elapsed(beginning);

    {
      auto start = Clock::now();
      hook(num_steps, t, mesh, *this);
      time_hook += elapsed(start);
    }

    while (t < t_end) {
      // exchange data between blocks
      {
        auto start = Clock::now();
#pragma omp parallel for
        for (auto &f : x_faces_)
          f->ExchangeGhostData();
#pragma omp parallel for
        for (auto &f : y_faces_)
          f->ExchangeGhostData();
#pragma omp parallel for
        for (auto &f : z_faces_)
          f->ExchangeGhostData();

        time_ghost_exchange += elapsed(start);
      }

      // take time step in each block
      double new_dt = std::numeric_limits<double>::max();
      {
        auto start = Clock::now();

#pragma omp parallel for
        for (auto &b : blocks_) {
          b.TakeStepA(riemann, eos, dt);
        }

#pragma omp parallel for
        for (auto &b : blocks_) {
          b.TakeStepB(riemann, eos, dt);
        }

#pragma omp parallel for reduction(min : new_dt)
        for (auto &b : blocks_) {
          new_dt = std::min(new_dt, b.TakeStepC(riemann, eos, dt));
        }
        time_hydro += elapsed(start);
      }

      t += dt;
      ++num_steps;
      dt = new_dt * cfl;
      if (num_steps <= 5) dt *= 0.2;
      dt = std::min(dt, t_end - t);

      {
        auto start = Clock::now();
        hook(num_steps, t, mesh, *this);
        time_hook += elapsed(start);
      }
      printf("[%06lu] t = %.10e, dt = %.10e\n", num_steps, t, dt);
    }

    // assemble final state into StateData
    auto start = Clock::now();
    Kokkos::View<State ***, HostSpace> output(
        "output", mesh.num_cells()[0], mesh.num_cells()[1], mesh.num_cells()[2]);

    for (size_t i = 0; i < N_blocks[0]; ++i) {
      for (size_t j = 0; j < N_blocks[1]; ++j) {
        for (size_t k = 0; k < N_blocks[2]; ++k) {
          auto sub = subview(output,
              make_pair(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE),
              make_pair(j * BLOCK_SIZE, (j + 1) * BLOCK_SIZE),
              make_pair(k * BLOCK_SIZE, (k + 1) * BLOCK_SIZE));

          deep_copy(sub, blocks_[block_idx(i, j, k)].GetStateData());
        }
      }
    }

    time_setup += elapsed(start);
    double time_total = elapsed(beginning);
    double time_other = time_total - (time_setup + time_ghost_exchange + time_hydro + time_hook);

    printf("\n");
    printf("Setup & tear-down: %8.3f s  %6.2f%%\n", time_setup, 100.0 * time_setup / time_total);
    printf("   Ghost exchange: %8.3f s  %6.2f%%\n",
        time_ghost_exchange,
        100.0 * time_ghost_exchange / time_total);
    printf("            Hydro: %8.3f s  %6.2f%%\n", time_hydro, 100.0 * time_hydro / time_total);
    printf("             Hook: %8.3f s  %6.2f%%\n", time_hook, 100.0 * time_hook / time_total);
    printf("            Other: %8.3f s  %6.2f%%\n", time_other, 100.0 * time_other / time_total);
    printf("            Total: %8.3f s  %6.2f%%\n\n", time_total, 100.0 * time_total / time_total);

    return output;
  }

  template <typename EOS>
  void DumpGrid(const std::string &fname,
      const size_t num_steps,
      const double time,
      const Mesh &mesh,
      const EOS &eos,
      const bool z_slice = false) const {
    std::vector<const Block_t *> block_ptrs(blocks_.size());
    for (size_t i = 0; i < blocks_.size(); ++i)
      block_ptrs[i] = &blocks_[i];

    output::DumpGrid(fname, num_steps, time, mesh, eos, block_ptrs, z_slice);
  }

private:
  std::vector<Block_t> blocks_;

  std::vector<std::shared_ptr<Face<Block_t>>> x_faces_;
  std::vector<std::shared_ptr<Face<Block_t>>> y_faces_;
  std::vector<std::shared_ptr<Face<Block_t>>> z_faces_;
};

#endif // ETHON_BLOCK_MESH_DRIVER_HPP_
