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

#ifndef ETHON_BLOCK_MESH_BLOCK_HPP_
#define ETHON_BLOCK_MESH_BLOCK_HPP_

#include <Kokkos_Core.hpp>

#include "block_mesh/amr_mesh_info.hpp"
#include "block_mesh/block_array.hpp"
#include "block_mesh/face.hpp"
#include "state.hpp"

using namespace Kokkos;

template <typename EOS, SlopeType SLOPE, size_t BLOCK_SIZE>
class Block {
public:
  using Mesh = AMRMeshInfo<BLOCK_SIZE>;
  using State = State_u<3>;
  using StateData = View<State ***, LayoutStride, Kokkos::HostSpace>;
  using refinement_func_t = std::function<int(const Block *block, const double time)>;

  static refinement_func_t no_refine() {
    return [](const Block *, const double) { return 0; };
  }

  static constexpr size_t N = BLOCK_SIZE;
  static constexpr size_t N_ghost = 2;
  static constexpr size_t N_tot = N + 2 * N_ghost;

  using BlockArray = Array3D<double, N_tot, N_tot, N_tot>;

  Block() { Init({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, -1); };

  Block(const Array<double, 3> lower_bounds, const Array<double, 3> cell_size, const int level) {
    Init(lower_bounds, cell_size, level);
  }

  void Init(
      const Array<double, 3> lower_bounds, const Array<double, 3> cell_size, const int level) {
    refinement_flag_ = 0;
    derefine_count_ = 0;

    lower_bounds_ = lower_bounds;
    cell_size_ = cell_size;
    block_size_[0] = N * cell_size_[0];
    block_size_[1] = N * cell_size_[1];
    block_size_[2] = N * cell_size_[2];

    // initialize neighbor levels
    for (int i = -1; i <= 1; ++i) {
      for (int j = -1; j <= 1; ++j) {
        for (int k = -1; k <= 1; ++k) {
          neighbor_level(i, j, k) = -1;
        }
      }
    }
    neighbor_level(0, 0, 0) = level;
  }

  ~Block() {
    // printf("Block destroyed\n");
  }

  auto lower_bounds() const { return lower_bounds_; }
  auto cell_size() const { return cell_size_; }
  auto block_size() const { return block_size_; }

  inline const auto &neighbor_level(const int i, const int j, const int k) const {
    assert(i >= -1);
    assert(i <= 1);
    assert(j >= -1);
    assert(j <= 1);
    assert(k >= -1);
    assert(k <= 1);
    return neighbor_levels_[i + 1][j + 1][k + 1];
  }

  inline auto &neighbor_level(const int i, const int j, const int k) {
    assert(i >= -1);
    assert(i <= 1);
    assert(j >= -1);
    assert(j <= 1);
    assert(k >= -1);
    assert(k <= 1);
    return neighbor_levels_[i + 1][j + 1][k + 1];
  }

  inline auto level() const { return neighbor_level(0, 0, 0); }

  /**
   * @brief Return the refinement flag set by EvaluateRefinement.
   *
   * The refinement flag is cached because it will need to be accessed multiple times. By caching
   * it, we avoid unnecessary evaluations and we don't mess up the derefinement count.
   *
   * @return int
   */
  int RefinementFlag() const { return refinement_flag_; }

  auto &rho() { return rho_; }
  auto &mu0() { return mu0_; }
  auto &mu1() { return mu1_; }
  auto &mu2() { return mu2_; }
  auto &epsilon() { return epsilon_; }

  const auto &rho() const { return rho_; }
  const auto &mu0() const { return mu0_; }
  const auto &mu1() const { return mu1_; }
  const auto &mu2() const { return mu2_; }
  const auto &epsilon() const { return epsilon_; }

  void SetState(const StateData &init) {
    assert(init.extent(0) == N);
    assert(init.extent(1) == N);
    assert(init.extent(2) == N);

    for (size_t k = N_ghost; k < N + N_ghost; ++k) {
      for (size_t j = N_ghost; j < N + N_ghost; ++j) {
        for (size_t i = N_ghost; i < N + N_ghost; ++i) {
          const auto U = init(i - N_ghost, j - N_ghost, k - N_ghost);
          rho_(i, j, k) = U.rho;
          mu0_(i, j, k) = U.mu[0];
          mu1_(i, j, k) = U.mu[1];
          mu2_(i, j, k) = U.mu[2];
          epsilon_(i, j, k) = U.epsilon;
        }
      }
    }
  }

  inline void SetState(const int i, const int j, const int k, const State &state) {
    rho_(i, j, k) = state.rho;
    mu0_(i, j, k) = state.mu[0];
    mu1_(i, j, k) = state.mu[1];
    mu2_(i, j, k) = state.mu[2];
    epsilon_(i, j, k) = state.epsilon;
  }

  inline void SetStateLogicalIndices(const int i, const int j, const int k, const State &state) {
    SetState(i + N_ghost, j + N_ghost, k + N_ghost, state);
  }

  inline State GetState(const int i, const int j, const int k) const {
    State U;
    U.rho = rho_(i, j, k);
    U.mu[0] = mu0_(i, j, k);
    U.mu[1] = mu1_(i, j, k);
    U.mu[2] = mu2_(i, j, k);
    U.epsilon = epsilon_(i, j, k);
    return U;
  }

  StateData GetStateData() const {
    Kokkos::View<State ***, HostSpace> output("state", N, N, N);
    for (size_t k = 0; k < N; ++k) {
      for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
          output(i, j, k) = GetState(i + N_ghost, j + N_ghost, k + N_ghost);
        }
      }
    }

    return output;
  }

  inline Array<double, 3> GetCellCenterCoordsFromLogicalIndices(
      const int i, const int j, const int k) const {
    Array<double, 3> coords = lower_bounds_;

    coords[0] += cell_size_[0] * (0.5 + double(i));
    coords[1] += cell_size_[1] * (0.5 + double(j));
    coords[2] += cell_size_[2] * (0.5 + double(k));

    return coords;
  }

  double MinDt(const EOS &eos) const {
    double dt = std::numeric_limits<double>::max();

    for (size_t k = N_ghost; k < N + N_ghost; ++k) {
      for (size_t j = N_ghost; j < N + N_ghost; ++j) {
        for (size_t i = N_ghost; i < N + N_ghost; ++i) {
          auto U = GetState(i, j, k);
          double cs = eos.sound_speed(U);

          double min_dt = cell_size_[0] / (fabs(U.u(0)) + cs);
          min_dt = std::min(min_dt, cell_size_[1] / (fabs(U.u(1)) + cs));
          min_dt = std::min(min_dt, cell_size_[2] / (fabs(U.u(2)) + cs));

          dt = std::min(dt, min_dt);
        }
      }
    }

    return dt;
  }

  template <typename RIEMANN>
  void TakeStepA(const RIEMANN &/*riemann*/, const EOS &eos, const double dt) {
    // compute boundary extrapolated states
    for (size_t k = 1; k < N_tot - 1; ++k) {
      for (size_t j = 1; j < N_tot - 1; ++j) {
        for (size_t i = 1; i < N_tot - 1; ++i) {
          // state of this cell
          const auto U = GetState(i, j, k);

          // FIXME get rid of this for production
          // {
          //   // we don't care about edges or corners
          //   bool check = true;
          //   if ((i == 1) || (i == N_tot - 2)) {
          //     // we are in the i ghost zone, only check if j and k are interior
          //     if ((j == 1) || (j == N_tot - 2) || (k == 1) || (k == N_tot - 2)) check = false;
          //   }
          //   if ((j == 1) || (j == N_tot - 2)) {
          //     // we are in the j ghost zone, only check if i and k are interior
          //     if ((i == 1) || (i == N_tot - 2) || (k == 1) || (k == N_tot - 2)) check = false;
          //   }
          //   if ((k == 1) || (k == N_tot - 2)) {
          //     // we are in the i ghost zone, only check if j and k are interior
          //     if ((i == 1) || (i == N_tot - 2) || (j == 1) || (j == N_tot - 2)) check = false;
          //   }
          //   if (check && (U.specific_internal_energy() < 0.0)) {
          //     printf("state(%i,%i,%i) has negative internal energy\n", i, j, k);
          //   }
          // }

          // get limited slopes using states from the lower and upper neighboring cells, if this
          // cell is at the boundary, we use this cell instead, which results in outflow boundary
          // conditions
          Array<State, 3> D;
          D[0] = Slope<SLOPE>::get(GetState(i - 1, j, k), U, GetState(i + 1, j, k));
          D[1] = Slope<SLOPE>::get(GetState(i, j - 1, k), U, GetState(i, j + 1, k));
          D[2] = Slope<SLOPE>::get(GetState(i, j, k - 1), U, GetState(i, j, k + 1));

          // boundary extrapolated states to lower and upper boundaries
          for (size_t d = 0; d < 3; ++d) {
            auto UL = U - D[d] * 0.5;
            auto UU = U + D[d] * 0.5;

            const auto FL = State::Flux(d, UL, eos.pressure(UL));
            const auto FU = State::Flux(d, UU, eos.pressure(UU));

            const auto flux_diff = (FL - FU) * 0.5 * dt / cell_size_[d];

            const auto this_lower_bdry_state = UL + flux_diff;
            const auto this_upper_bdry_state = UU + flux_diff;

            // make sure intermediate states are physical
            if ((this_lower_bdry_state.specific_internal_energy() < 0.0) ||
                (this_upper_bdry_state.specific_internal_energy() < 0.0)) {
              // printf("fixing negative energy in %i,%i,%i,%lu\n", i, j, k, d);
              lower_boundary_state_(i, j, k, d) = U;
              upper_boundary_state_(i, j, k, d) = U;
            } else {
              lower_boundary_state_(i, j, k, d) = this_lower_bdry_state;
              upper_boundary_state_(i, j, k, d) = this_upper_bdry_state;
            }
          }
        }
      }
    }
  }

  template <typename RIEMANN>
  void TakeStepB(const RIEMANN &riemann, const EOS &/*eos*/, const double /*dt*/) {
    // TODO combine into one loop

    // compute fluxes
    // need a sync here (i.e. new parallel_for loop), since we are going to read
    // lower_boundary_state_ and upper_boundary_state_ from neighboring cells
    for (size_t k = N_ghost; k < N + N_ghost; ++k) {
      for (size_t j = N_ghost; j < N + N_ghost; ++j) {
        for (size_t i = N_ghost; i < N + N_ghost + 1; ++i) {
          x_fluxes_(i, j, k) =
              riemann(0, upper_boundary_state_(i - 1, j, k, 0), lower_boundary_state_(i, j, k, 0));
        }
      }
    }

    for (size_t k = N_ghost; k < N + N_ghost; ++k) {
      for (size_t j = N_ghost; j < N + N_ghost + 1; ++j) {
        for (size_t i = N_ghost; i < N + N_ghost; ++i) {
          y_fluxes_(i, j, k) =
              riemann(1, upper_boundary_state_(i, j - 1, k, 1), lower_boundary_state_(i, j, k, 1));
        }
      }
    }

    for (size_t k = N_ghost; k < N + N_ghost + 1; ++k) {
      for (size_t j = N_ghost; j < N + N_ghost; ++j) {
        for (size_t i = N_ghost; i < N + N_ghost; ++i) {
          z_fluxes_(i, j, k) =
              riemann(2, upper_boundary_state_(i, j, k - 1, 2), lower_boundary_state_(i, j, k, 2));
        }
      }
    }
  }

  template <typename RIEMANN>
  double TakeStepC(const RIEMANN &/*riemann*/, const EOS &eos, const double dt) {

    // need to sync here so that all fluxes are available
    double new_dt = std::numeric_limits<double>::max();
    for (size_t k = N_ghost; k < N + N_ghost; ++k) {
      for (size_t j = N_ghost; j < N + N_ghost; ++j) {
        for (size_t i = N_ghost; i < N + N_ghost; ++i) {
          State flux_diff;
          flux_diff += (x_fluxes_(i, j, k) - x_fluxes_(i + 1, j, k)) / cell_size_[0];
          flux_diff += (y_fluxes_(i, j, k) - y_fluxes_(i, j + 1, k)) / cell_size_[1];
          flux_diff += (z_fluxes_(i, j, k) - z_fluxes_(i, j, k + 1)) / cell_size_[2];

          auto U = GetState(i, j, k);
          U += flux_diff * dt;

          double cs = eos.sound_speed(U);

          double min_dt = cell_size_[0] / (fabs(U.u(0)) + cs);
          min_dt = std::min(min_dt, cell_size_[1] / (fabs(U.u(1)) + cs));
          min_dt = std::min(min_dt, cell_size_[2] / (fabs(U.u(2)) + cs));

          new_dt = std::min(new_dt, min_dt);

          rho_(i, j, k) = U.rho;
          mu0_(i, j, k) = U.mu[0];
          mu1_(i, j, k) = U.mu[1];
          mu2_(i, j, k) = U.mu[2];
          epsilon_(i, j, k) = U.epsilon;
        }
      }
    }

    return new_dt;
  }

  // Set the refinement_flag_ to 1 if block should be refined, to 0 if neither refined or
  // coarsened, or to -1 if block should be coarsened
  void EvaluateRefinement(
      refinement_func_t refinement_func, const double time, const int amr_derefinement_count) {
    int ref = refinement_func(this, time);
    if (ref == -1) {
      // this block is marked for derefinement, but only derefine if it's been marked for
      // derefinement for at least amr_derefinement_count times in a row
      ++derefine_count_;
      if (derefine_count_ >= amr_derefinement_count) {
        refinement_flag_ = -1;
      } else {
        refinement_flag_ = 0;
      }
    } else {
      // this block is not marked for derefinement, reset the counter
      derefine_count_ = 0;

      if (ref == 1)
        refinement_flag_ = 1;
      else
        refinement_flag_ = 0;
    }
  }

  /**
   * @brief Prolongate the data of this block to an immediate child block including the ghost
   * zones.
   *
   * ASSUMPTION: We can only fill in the child's ghost zones if the number of ghost zones is 2.
   *
   * @param child     The child block to prolongate on
   * @param child_i   0 or 1 indicating whether the child is in the upper or lower half in x
   * @param child_j   0 or 1 indicating whether the child is in the upper or lower half in y
   * @param child_k   0 or 1 indicating whether the child is in the upper or lower half in z
   */
  static_assert(N_ghost == 2, "ProlongateOntoChild only works with N_ghost == 2");
  void ProlongateOntoChild(Block *child, const int child_i, const int child_j, const int child_k) {
    ProlongateFieldOntoChild(&child->rho(), &rho_, child_i, child_j, child_k);
    ProlongateFieldOntoChild(&child->mu0(), &mu0_, child_i, child_j, child_k);
    ProlongateFieldOntoChild(&child->mu1(), &mu1_, child_i, child_j, child_k);
    ProlongateFieldOntoChild(&child->mu2(), &mu2_, child_i, child_j, child_k);
    ProlongateFieldOntoChild(&child->epsilon(), &epsilon_, child_i, child_j, child_k);
  }

  /**
   * @brief Restrict the data of this block to an immidate parent block. The ghost zones will NOT
   * be filled in, because the finer child block does not have the data for two layers of ghost
   * zones on the coarser parent. So we'll need to do a ghost exchange after this.
   *
   * @param parent    The parent block to restrict on
   * @param child_i   0 or 1 indicating whether the child is in the upper or lower half in x
   * @param child_j   0 or 1 indicating whether the child is in the upper or lower half in y
   * @param child_k   0 or 1 indicating whether the child is in the upper or lower half in z
   */
  void RestrictOntoParent(Block *parent, const int child_i, const int child_j, const int child_k) {
    RestrictFieldOntoParent(&parent->rho(), &rho_, child_i, child_j, child_k);
    RestrictFieldOntoParent(&parent->mu0(), &mu0_, child_i, child_j, child_k);
    RestrictFieldOntoParent(&parent->mu1(), &mu1_, child_i, child_j, child_k);
    RestrictFieldOntoParent(&parent->mu2(), &mu2_, child_i, child_j, child_k);
    RestrictFieldOntoParent(&parent->epsilon(), &epsilon_, child_i, child_j, child_k);
  }

private:
  void ProlongateFieldOntoChild(BlockArray *dat_child,
      const BlockArray *dat_parent,
      const int child_i,
      const int child_j,
      const int child_k) {
    // loop over 1/8 of this block with one layer of ghost zones (which will be prolongated to
    // 2 layers of ghost zones on the child)
    for (int r = -1; r < int(N) / 2 + 1; ++r) {
      for (int q = -1; q < int(N) / 2 + 1; ++q) {
        for (int p = -1; p < int(N) / 2 + 1; ++p) {
          // coarse indices
          int pc = N_ghost + child_i * N / 2 + p;
          int qc = N_ghost + child_j * N / 2 + q;
          int rc = N_ghost + child_k * N / 2 + r;

          // fine indices
          int pf = N_ghost + 2 * p;
          int qf = N_ghost + 2 * q;
          int rf = N_ghost + 2 * r;

          DoProlongation<BlockArray, SLOPE>(dat_child, dat_parent, pc, qc, rc, pf, qf, rf);
        }
      }
    }
  }

  void RestrictFieldOntoParent(BlockArray *dat_parent,
      const BlockArray *dat_child,
      const int child_i,
      const int child_j,
      const int child_k) {
    // loop over interior points in this block and restrict them to the parent block. We
    // don't have enough data to fill in two layers of ghost zones on the parent
    for (int r = 0; r < int(N) / 2; ++r) {
      for (int q = 0; q < int(N) / 2; ++q) {
        for (int p = 0; p < int(N) / 2; ++p) {
          // coarse indices
          int pc = N_ghost + child_i * N / 2 + p;
          int qc = N_ghost + child_j * N / 2 + q;
          int rc = N_ghost + child_k * N / 2 + r;

          // fine indices
          int pf = N_ghost + 2 * p;
          int qf = N_ghost + 2 * q;
          int rf = N_ghost + 2 * r;

          DoRestriction<BlockArray>(dat_parent, dat_child, pc, qc, rc, pf, qf, rf);
        }
      }
    }
  }

  // x, y, z coordinates of the lower bounds of this block
  Array<double, 3> lower_bounds_;

  // the size of the cells and size of the entire block in the x-, y-, and z-direction
  Array<double, 3> cell_size_, block_size_;

  // Store the level of the 26 neighbor blocks (6 face connections, 12 edge connections, and 8
  // corner connections). Note that a block may have 4 neighbors across a face if they are on a
  // finer level, but we only need to store one number for those, because these 4 neighbors all
  // have the same level (level of this block + 1). Same goes for edges, where there could be 2
  // finer blocks. In each dimension, the neighbor index can be n = -1, 0, or +1, which
  // corresponds to array index n + 1. Neighbor index (0,0,0) refers to this block and stores the
  // level of this block
  Array<Array<Array<int, 3>, 3>, 3> neighbor_levels_;

  // mass density in g cm^{-3}
  BlockArray rho_;

  // momentum density (mu_i = rho * u_i, where u_i is fluid velocity) in g cm^{-2} s^{-1}
  BlockArray mu0_, mu1_, mu2_;

  // total energy density (rho * e + 1/2 * rho * u_i u^i, where e is the internal specific energy)
  // in erg cm^{-3} = g cm^{-1} s^{-2}
  BlockArray epsilon_;

  // temporary storage
  Array4D<State, N_tot, N_tot, N_tot, 3> lower_boundary_state_, upper_boundary_state_;
  Array3D<State, N_tot, N_tot, N_tot> x_fluxes_, y_fluxes_, z_fluxes_;

  // Flag indicating whether the block should be refined (1), derefined (-1), or no action is
  // necessary (0). The flag is set by the EvaluateRefinement function.
  int refinement_flag_ = 0;

  // The number of times in a row this block has been marked for derefinement in
  // EvaluateRefinement. We only actually set the flag indicating that this block should be
  // derefined if this counter reaches 5. If the block is not marked for derefinement in
  // EvaluateRefinement, this counter is reset to 0.
  int derefine_count_ = 0;
};

#endif // ETHON_BLOCK_MESH_BLOCK_HPP_
