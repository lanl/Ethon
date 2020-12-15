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

#ifndef ETHON_KOKKOS_OMP_BLOCK_HPP_
#define ETHON_KOKKOS_OMP_BLOCK_HPP_

#include <Kokkos_Core.hpp>

#include "kokkos_omp/amr_mesh_info.hpp"
#include "kokkos_omp/face.hpp"
#include "state.hpp"

using namespace Kokkos;

template <typename EOS, SlopeType SLOPE, int BLOCK_SIZE, typename ExecSpace>
class Block {
public:
  constexpr static int N = BLOCK_SIZE;
  constexpr static int N_ghost = 2;
  constexpr static int N_tot = N + 2 * N_ghost;
  constexpr static int N_vars = 5;
  constexpr static int N_blocks = 10000;

  using CellData_t = Kokkos::
      View<double[N_vars][N_tot][N_tot][N_tot][N_blocks], Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using FluxData_t = Kokkos::
      View<double[N_vars][N_tot][N_tot][N_tot][3][N_blocks], Kokkos::LayoutLeft, Kokkos::HostSpace>;
  using BlockIDs_t = Kokkos::View<int *, Kokkos::HostSpace>;
  using CellSizes_t = Kokkos::View<double *[3], Kokkos::HostSpace>;
  using ExecSpace_t = ExecSpace;

  using Mesh = AMRMeshInfo<BLOCK_SIZE>;
  using State = State_u<3>;
  using StateData = View<State ***, LayoutStride, Kokkos::HostSpace>;
  using refinement_func_t = int (*)(const CellData_t &dat,
      int bid,
      double time,
      Array<double, 3> cell_size,
      double amr_threshold,
      const EOS &);

  KOKKOS_FUNCTION static int no_refine(
      const CellData_t &, int, double, Array<double, 3>, double, const EOS &) {
    return 0;
  }

  KOKKOS_FUNCTION static int refine_KH(const CellData_t &dat,
      int bid,
      double /*time*/,
      Kokkos::Array<double, 3> /*cell_size*/,
      double amr_threshold,
      const EOS & /*eos*/) {
    double max_val = 0.0;

    for (int k = N_ghost; k < BLOCK_SIZE + N_ghost; ++k) {
      for (int j = N_ghost; j < BLOCK_SIZE + N_ghost; ++j) {
        for (int i = N_ghost; i < BLOCK_SIZE + N_ghost; ++i) {
          double vgy = fabs(dat(2, i + 1, j, k, bid) / dat(0, i + 1, j, k, bid) -
                            dat(2, i - 1, j, k, bid) / dat(0, i - 1, j, k, bid)) *
                       0.5;
          double vgx = fabs(dat(1, i, j + 1, k, bid) / dat(0, i, j + 1, k, bid) -
                            dat(1, i, j - 1, k, bid) / dat(0, i, j - 1, k, bid)) *
                       0.5;

          double vg = sqrt(vgx * vgx + vgy * vgy);

          max_val = fmax(max_val, vg);
        }
      }
    }

    // printf("max = %.4f\n", max);

    if (max_val < 0.5 * amr_threshold) {
      return -1;
    } else if (max_val > amr_threshold) {
      return 1;
    } else {
      return 0;
    }
  }

  KOKKOS_FUNCTION static int refine_Blast(const CellData_t &dat,
      int bid,
      double /*time*/,
      Kokkos::Array<double, 3> /*cell_size*/,
      double amr_threshold,
      const EOS &eos) {
    MinMaxScalar<double> minmax;
    minmax.min_val = 1.0E300;
    minmax.max_val = 0.0;

    for (int k = N_ghost - 1; k < BLOCK_SIZE + N_ghost + 1; ++k) {
      for (int j = N_ghost - 1; j < BLOCK_SIZE + N_ghost + 1; ++j) {
        for (int i = N_ghost - 1; i < BLOCK_SIZE + N_ghost + 1; ++i) {
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

          // double norm = sqrt(dx * dx + dy * dy + dz * dz) / eos.pressure(block->GetState(i,
          // j, k)); value_to_update = std::max(value_to_update, norm);
          if (num_outside <= 1) {
            double press = eos.pressure(GetState(dat, bid, i, j, k));
            minmax.min_val = fmin(minmax.min_val, press);
            minmax.max_val = fmax(minmax.max_val, press);
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
  }

  Block() {
    // printf("Block created\n");
    Init(-1, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, -1);
  };

  void Init(int id, Array<double, 3> lower_bounds, Array<double, 3> cell_size, int level) {
    id_ = id;
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

  auto id() const { return id_; }
  auto lower_bounds() const { return lower_bounds_; }
  auto cell_size() const { return cell_size_; }
  auto block_size() const { return block_size_; }

  inline const auto &neighbor_level(int i, int j, int k) const {
    assert(i >= -1);
    assert(i <= 1);
    assert(j >= -1);
    assert(j <= 1);
    assert(k >= -1);
    assert(k <= 1);
    return neighbor_levels_[i + 1][j + 1][k + 1];
  }

  inline auto &neighbor_level(int i, int j, const int k) {
    assert(i >= -1);
    assert(i <= 1);
    assert(j >= -1);
    assert(j <= 1);
    assert(k >= -1);
    assert(k <= 1);
    return neighbor_levels_[i + 1][j + 1][k + 1];
  }

  inline auto level() const { return neighbor_level(0, 0, 0); }

  void SetState(const CellData_t &dat, const StateData &init) {
    assert(init.extent(0) == N);
    assert(init.extent(1) == N);
    assert(init.extent(2) == N);

    for (int k = N_ghost; k < N + N_ghost; ++k) {
      for (int j = N_ghost; j < N + N_ghost; ++j) {
        for (int i = N_ghost; i < N + N_ghost; ++i) {
          const auto U = init(i - N_ghost, j - N_ghost, k - N_ghost);
          SetState(dat, id_, i, j, k, U);
        }
      }
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION static void SetState(
      const CellData_t &dat, int bid, int i, int j, int k, const State &state) {
    dat(0, i, j, k, bid) = state.rho;
    dat(1, i, j, k, bid) = state.mu[0];
    dat(2, i, j, k, bid) = state.mu[1];
    dat(3, i, j, k, bid) = state.mu[2];
    dat(4, i, j, k, bid) = state.epsilon;
  }

  KOKKOS_FORCEINLINE_FUNCTION static void SetFlux(
      const FluxData_t &dat, int bid, int i, int j, int k, int d, const State &state) {
    dat(0, i, j, k, d, bid) = state.rho;
    dat(1, i, j, k, d, bid) = state.mu[0];
    dat(2, i, j, k, d, bid) = state.mu[1];
    dat(3, i, j, k, d, bid) = state.mu[2];
    dat(4, i, j, k, d, bid) = state.epsilon;
  }

  inline void SetStateLogicalIndices(
      const CellData_t &dat, int i, int j, int k, const State &state) {
    SetState(dat, id_, i + N_ghost, j + N_ghost, k + N_ghost, state);
  }

  KOKKOS_FORCEINLINE_FUNCTION static State GetState(
      const CellData_t &dat, int bid, int i, int j, int k) {
    State U;
    U.rho = dat(0, i, j, k, bid);
    U.mu[0] = dat(1, i, j, k, bid);
    U.mu[1] = dat(2, i, j, k, bid);
    U.mu[2] = dat(3, i, j, k, bid);
    U.epsilon = dat(4, i, j, k, bid);
    return U;
  }

  KOKKOS_FORCEINLINE_FUNCTION static State GetFlux(
      const FluxData_t &dat, int bid, int i, int j, int k, int d) {
    State U;
    U.rho = dat(0, i, j, k, d, bid);
    U.mu[0] = dat(1, i, j, k, d, bid);
    U.mu[1] = dat(2, i, j, k, d, bid);
    U.mu[2] = dat(3, i, j, k, d, bid);
    U.epsilon = dat(4, i, j, k, d, bid);
    return U;
  }

  StateData GetStateData() const {
    Kokkos::View<State ***, Kokkos::HostSpace> output("state", N, N, N);
    for (int k = 0; k < N; ++k) {
      for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
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

  static double MinDt(const CellData_t &dat,
      const BlockIDs_t &block_ids,
      const CellSizes_t &cell_sizes,
      const EOS &eos) {
    double dt = std::numeric_limits<double>::max();
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecSpace>(0, block_ids.extent(0)),
        KOKKOS_LAMBDA(int b, double &this_dt) {
          for (int k = N_ghost; k < N + N_ghost; ++k) {
            for (int j = N_ghost; j < N + N_ghost; ++j) {
              for (int i = N_ghost; i < N + N_ghost; ++i) {
                auto U = GetState(dat, block_ids(b), i, j, k);
                double cs = eos.sound_speed(U);

                double min_dt = cell_sizes(b, 0) / (fabs(U.u(0)) + cs);
                min_dt = fmin(min_dt, cell_sizes(b, 1) / (fabs(U.u(1)) + cs));
                min_dt = fmin(min_dt, cell_sizes(b, 2) / (fabs(U.u(2)) + cs));

                this_dt = fmin(this_dt, min_dt);
              }
            }
          }
        },
        Kokkos::Min<double>(dt));

    return dt;
  }

  template <typename RIEMANN>
  static void TakeStepA(const CellData_t &dat,
      const FluxData_t &lower_bdry_flxs,
      const FluxData_t &upper_bdry_flxs,
      const BlockIDs_t &block_ids,
      const CellSizes_t &cell_sizes,
      const RIEMANN & /*riemann*/,
      const EOS &eos,
      double dt) {
    // compute boundary extrapolated states
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>(0, block_ids.extent(0)), KOKKOS_LAMBDA(int b) {
          int bid = block_ids(b);
          for (int k = 1; k < N_tot - 1; ++k) {
            for (int j = 1; j < N_tot - 1; ++j) {
              for (int i = 1; i < N_tot - 1; ++i) {
                // state of this cell
                const auto U = GetState(dat, bid, i, j, k);

                // FIXME get rid of this for production
                // {
                //   // we don't care about edges or corners
                //   bool check = true;
                //   if ((i == 1) || (i == N_tot - 2)) {
                //     // we are in the i ghost zone, only check if j and k are interior
                //     if ((j == 1) || (j == N_tot - 2) || (k == 1) || (k == N_tot - 2)) check =
                //     false;
                //   }
                //   if ((j == 1) || (j == N_tot - 2)) {
                //     // we are in the j ghost zone, only check if i and k are interior
                //     if ((i == 1) || (i == N_tot - 2) || (k == 1) || (k == N_tot - 2)) check =
                //     false;
                //   }
                //   if ((k == 1) || (k == N_tot - 2)) {
                //     // we are in the i ghost zone, only check if j and k are interior
                //     if ((i == 1) || (i == N_tot - 2) || (j == 1) || (j == N_tot - 2)) check =
                //     false;
                //   }
                //   if (check && (U.specific_internal_energy() < 0.0)) {
                //     printf("state(%i,%i,%i) has negative internal energy\n", i, j, k);
                //   }
                // }

                // get limited slopes using states from the lower and upper neighboring cells, if
                // this cell is at the boundary, we use this cell instead, which results in outflow
                // boundary conditions
                Array<State, 3> D;
                D[0] = Slope<SLOPE>::get(
                    GetState(dat, bid, i - 1, j, k), U, GetState(dat, bid, i + 1, j, k));
                D[1] = Slope<SLOPE>::get(
                    GetState(dat, bid, i, j - 1, k), U, GetState(dat, bid, i, j + 1, k));
                D[2] = Slope<SLOPE>::get(
                    GetState(dat, bid, i, j, k - 1), U, GetState(dat, bid, i, j, k + 1));

                // boundary extrapolated states to lower and upper boundaries
                for (int d = 0; d < 3; ++d) {
                  auto UL = U - D[d] * 0.5;
                  auto UU = U + D[d] * 0.5;

                  const auto FL = State::Flux(d, UL, eos.pressure(UL));
                  const auto FU = State::Flux(d, UU, eos.pressure(UU));

                  const auto flux_diff = (FL - FU) * 0.5 * dt / cell_sizes(b, d);

                  const auto this_lower_bdry_state = UL + flux_diff;
                  const auto this_upper_bdry_state = UU + flux_diff;

                  // make sure intermediate states are physical
                  if ((this_lower_bdry_state.specific_internal_energy() < 0.0) ||
                      (this_upper_bdry_state.specific_internal_energy() < 0.0)) {
                    // printf("fixing negative energy in %i,%i,%i,%lu\n", i, j, k, d);
                    SetFlux(lower_bdry_flxs, bid, i, j, k, d, U);
                    SetFlux(upper_bdry_flxs, bid, i, j, k, d, U);
                  } else {
                    SetFlux(lower_bdry_flxs, bid, i, j, k, d, this_lower_bdry_state);
                    SetFlux(upper_bdry_flxs, bid, i, j, k, d, this_upper_bdry_state);
                  }
                }
              }
            }
          }
        });
  }

  template <typename RIEMANN>
  static void TakeStepB(const FluxData_t &lower_bdry_flxs,
      const FluxData_t &upper_bdry_flxs,
      const CellData_t &x_fluxes,
      const CellData_t &y_fluxes,
      const CellData_t &z_fluxes,
      const BlockIDs_t &block_ids,
      const RIEMANN &riemann,
      const EOS & /*eos*/,
      double /*dt*/) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>(0, block_ids.extent(0)), KOKKOS_LAMBDA(int b) {
          // TODO combine into one loop
          int bid = block_ids(b);

          // compute fluxes
          // need a sync here (i.e. new parallel_for loop), since we are going to read
          // lower_boundary_state_ and upper_boundary_state_ from neighboring cells
          for (int k = N_ghost; k < N + N_ghost; ++k) {
            for (int j = N_ghost; j < N + N_ghost; ++j) {
              for (int i = N_ghost; i < N + N_ghost + 1; ++i) {
                auto lower = GetFlux(upper_bdry_flxs, bid, i - 1, j, k, 0);
                auto upper = GetFlux(lower_bdry_flxs, bid, i, j, k, 0);
                auto state = riemann(0, lower, upper);
                SetState(x_fluxes, bid, i, j, k, state);
              }
            }
          }

          for (int k = N_ghost; k < N + N_ghost; ++k) {
            for (int j = N_ghost; j < N + N_ghost + 1; ++j) {
              for (int i = N_ghost; i < N + N_ghost; ++i) {
                auto lower = GetFlux(upper_bdry_flxs, bid, i, j - 1, k, 1);
                auto upper = GetFlux(lower_bdry_flxs, bid, i, j, k, 1);
                auto state = riemann(1, lower, upper);
                SetState(y_fluxes, bid, i, j, k, state);
              }
            }
          }

          for (int k = N_ghost; k < N + N_ghost + 1; ++k) {
            for (int j = N_ghost; j < N + N_ghost; ++j) {
              for (int i = N_ghost; i < N + N_ghost; ++i) {
                auto lower = GetFlux(upper_bdry_flxs, bid, i, j, k - 1, 2);
                auto upper = GetFlux(lower_bdry_flxs, bid, i, j, k, 2);
                auto state = riemann(2, lower, upper);
                SetState(z_fluxes, bid, i, j, k, state);
              }
            }
          }
        });
  }

  template <typename RIEMANN>
  static double TakeStepC(const CellData_t &x_fluxes,
      const CellData_t &y_fluxes,
      const CellData_t &z_fluxes,
      const CellData_t &cell_dat,
      const BlockIDs_t &block_ids,
      const CellSizes_t &cell_sizes,
      const RIEMANN & /*riemann*/,
      const EOS &eos,
      double dt) {
    double new_dt = std::numeric_limits<double>::max();
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<ExecSpace>(0, block_ids.extent(0)),
        KOKKOS_LAMBDA(int b, double &this_dt) {
          int bid = block_ids(b);

          // need to sync here so that all fluxes are available
          for (int k = N_ghost; k < N + N_ghost; ++k) {
            for (int j = N_ghost; j < N + N_ghost; ++j) {
              for (int i = N_ghost; i < N + N_ghost; ++i) {
                auto U = GetState(cell_dat, bid, i, j, k);

                for (int v = 0; v < N_vars; ++v) {
                  double flux_diff = 0.0;
                  flux_diff += (x_fluxes(v, i, j, k, bid) - x_fluxes(v, i + 1, j, k, bid)) /
                               cell_sizes(b, 0);
                  flux_diff += (y_fluxes(v, i, j, k, bid) - y_fluxes(v, i, j + 1, k, bid)) /
                               cell_sizes(b, 1);
                  flux_diff += (z_fluxes(v, i, j, k, bid) - z_fluxes(v, i, j, k + 1, bid)) /
                               cell_sizes(b, 2);

                  U[v] += flux_diff * dt;
                }

                double cs = eos.sound_speed(U);

                double min_dt = cell_sizes(b, 0) / (fabs(U.u(0)) + cs);
                min_dt = fmin(min_dt, cell_sizes(b, 1) / (fabs(U.u(1)) + cs));
                min_dt = fmin(min_dt, cell_sizes(b, 2) / (fabs(U.u(2)) + cs));

                this_dt = fmin(this_dt, min_dt);
                SetState(cell_dat, bid, i, j, k, U);
              }
            }
          }
        },
        Kokkos::Min<double>(new_dt));

    return new_dt;
  }

  // Set the refinement_flag_ to 1 if block should be refined, to 0 if neither refined or
  // coarsened, or to -1 if block should be coarsened
  static void EvaluateRefinement(const CellData_t &dat,
      const BlockIDs_t &block_ids,
      const CellSizes_t &cell_sizes,
      refinement_func_t refinement_func,
      double time,
      double amr_threshold,
      const EOS &eos,
      Kokkos::View<int *, Kokkos::HostSpace> refinement_flags) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>(0, block_ids.extent(0)), KOKKOS_LAMBDA(int b) {
          int bid = block_ids(b);
          Kokkos::Array<double, 3> cell_size;
          cell_size[0] = cell_sizes(b, 0);
          cell_size[1] = cell_sizes(b, 1);
          cell_size[2] = cell_sizes(b, 2);

          int ref = refinement_func(dat, bid, time, cell_size, amr_threshold, eos);
          int flag = refinement_flags(bid);
          if (ref == -1) {
            if (flag >= 0) {
              // this block was not flagged for refinement in the last check
              flag = -1;
            } else {
              // this block was flagged for the refinement in the last check, increase the counter
              --flag;
            }
          } else {
            flag = ref;
          }
          refinement_flags(bid) = flag;
        });
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
  KOKKOS_FUNCTION static void ProlongateFieldOntoChild(
      const CellData_t &dat, int bid_child, int bid_parent, int child_i, int child_j, int child_k) {
    for (int v = 0; v < N_vars; ++v) {
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

            DoProlongation<CellData_t, SLOPE>(
                dat, bid_child, bid_parent, v, pc, qc, rc, pf, qf, rf);
          }
        }
      }
    }
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
  KOKKOS_FUNCTION static void RestrictFieldOntoParent(
      const CellData_t &dat, int bid_parent, int bid_child, int child_i, int child_j, int child_k) {
    for (int v = 0; v < N_vars; ++v) {
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

            DoRestriction(dat, bid_parent, bid_child, v, pc, qc, rc, pf, qf, rf);
          }
        }
      }
    }
  }

private:
  int id_;

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
};

#endif // ETHON_KOKKOS_OMP_BLOCK_HPP_
