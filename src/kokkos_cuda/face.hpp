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

#ifndef ETHON_KOKKOS_CUDA_FACE_HPP_
#define ETHON_KOKKOS_CUDA_FACE_HPP_

#include <array>
#include <memory>

#include <Kokkos_Core.hpp>
#include <Kokkos_Array.hpp>

#include "slopes.hpp"
#include "state.hpp"
#include "uniform_mesh/boundary.hpp"

using namespace Kokkos;

template <typename Block_t>
struct Face {
  static constexpr int N = Block_t::N;
  static constexpr int N_ghost = Block_t::N_ghost;
  static constexpr int N_tot = Block_t::N_tot;
  static constexpr int N_vars = Block_t::N_vars;

  using CellData_t = typename Block_t::CellData_t;
  using ExecSpace_t = typename Block_t::ExecSpace_t;
  using team_member_t = typename TeamPolicy<ExecSpace_t>::member_type;

  int8_t dir; // direction of the face, 0, 1, 2

  // we need a default constructor so we can make a Kokkos::View out of Faces
  KOKKOS_FUNCTION Face() : dir(-1) {}

  KOKKOS_FUNCTION Face(int8_t dir) : dir(dir) {
    // assert((dir >= 0) && (dir <= 2));
    // if ((dir < 0) || (dir > 2))
    //   throw std::invalid_argument("Face direction can only be 0, 1, 2");
  }

  /**
   * @brief Return true if this face is fully initialized (has all the required blocks connected)
   */
  virtual bool IsInitialized() const = 0;
  virtual void ConnectBlock(Side side, const Block_t *block) = 0;

protected:
  template <bool INVERT = false>
  KOKKOS_FORCEINLINE_FUNCTION static void CopyPlanes(const team_member_t &team_member,
      const CellData_t &dat,
      int bid_to,
      int bid_from,
      int dir,
      Kokkos::Array<int, N_ghost> idxs_to,
      Kokkos::Array<int, N_ghost> idxs_from) {
    constexpr int loop_count = N * N * N_ghost * N_vars;
    int Ni = dir == 0 ? N_ghost : N;
    int Nj = dir == 1 ? N_ghost : N;

    Kokkos::parallel_for(TeamThreadRange<>(team_member, loop_count), [=](int idx) {
      // idx = v + i * N_v + j * N_v * N_i + k * N_v * N_i * N_j
      // Rely on compiler optimization of modulus since divisor is compile-time constant
      int v = idx % N_vars;
      idx /= N_vars;

      int i = idx % Ni;
      idx /= Ni;

      int j = idx % Nj;
      int k = idx / Nj;

      int x = dir == 0 ? idxs_from[i] : i + N_ghost;
      int y = dir == 1 ? idxs_from[j] : j + N_ghost;
      int z = dir == 2 ? idxs_from[k] : k + N_ghost;
      double val = dat(v, x, y, z, bid_from);

      x = dir == 0 ? idxs_to[i] : i + N_ghost;
      y = dir == 1 ? idxs_to[j] : j + N_ghost;
      z = dir == 2 ? idxs_to[k] : k + N_ghost;
      dat(v, x, y, z, bid_to) = (INVERT && (v == (dir + 1))) ? -val : val;
    });
  }
};

/**
 * @brief A face between two blocks of the same refinement level
 */
template <typename Block_t>
struct IntercellFace : public Face<Block_t> {
  using Face<Block_t>::N;
  using Face<Block_t>::N_ghost;
  using Face<Block_t>::N_vars;
  using typename Face<Block_t>::CellData_t;
  using typename Face<Block_t>::ExecSpace_t;
  using typename Face<Block_t>::team_member_t;

  int lower_bid, upper_bid;

  // we need a default constructor so we can make a Kokkos::View out of BoundaryFaces
  KOKKOS_FUNCTION IntercellFace() {}

  KOKKOS_FUNCTION IntercellFace(int8_t dir) : Face<Block_t>(dir), lower_bid(-1), upper_bid(-1) {}

  bool IsInitialized() const final override {
    if (lower_bid < 0) return false;
    if (upper_bid < 0) return false;
    return true;
  }

  void ConnectBlock(Side side, const Block_t *block) final override {
    if (side == Side::Lower) {
      if (lower_bid >= 0) {
        throw std::runtime_error(
            "Trying to connect block to a face that's already connected to a block");
      }
      lower_bid = block->id();
    } else if (side == Side::Upper) {
      if (upper_bid >= 0) {
        throw std::runtime_error(
            "Trying to connect block to a face that's already connected to a block");
      }
      upper_bid = block->id();
    } else {
      throw std::runtime_error("Unknown side");
    }
  }

  static void ExchangeGhostData(
      const CellData_t &dat, const Kokkos::View<IntercellFace *, Kokkos::CudaSpace> &faces) {

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace_t>(faces.extent(0), Kokkos::AUTO()),
        KOKKOS_LAMBDA(team_member_t team_member) {
          int fid = team_member.league_rank();
          int bid_lower = faces(fid).lower_bid;
          int bid_upper = faces(fid).upper_bid;
          int dir = faces(fid).dir;

          Kokkos::Array<int, N_ghost> idxs_to, idxs_from;

          for (int i = 0; i < N_ghost; ++i) {
            idxs_to[i] = i;
            idxs_from[i] = i + N;
          }
          Face<Block_t>::CopyPlanes(
              team_member, dat, bid_upper, bid_lower, dir, idxs_to, idxs_from);

          for (int i = 0; i < N_ghost; ++i) {
            idxs_to[i] = i + N + N_ghost;
            idxs_from[i] = i + N_ghost;
          }
          Face<Block_t>::CopyPlanes(
              team_member, dat, bid_lower, bid_upper, dir, idxs_to, idxs_from);
        });
  }
};

// ci, cj, ck are coarse indices, fi, fj, fk are fine indices
template <typename CellData_t>
KOKKOS_FORCEINLINE_FUNCTION void DoRestriction(const CellData_t &dat,
    int bid_coarse,
    int bid_fine,
    int v,
    int ci,
    int cj,
    int ck,
    int fi,
    int fj,
    int fk) {
  dat(v, ci, cj, ck, bid_coarse) =
      (dat(v, fi, fj, fk, bid_fine) + dat(v, fi, fj, fk + 1, bid_fine) +
          dat(v, fi, fj + 1, fk, bid_fine) + dat(v, fi, fj + 1, fk + 1, bid_fine) +
          dat(v, fi + 1, fj, fk, bid_fine) + dat(v, fi + 1, fj, fk + 1, bid_fine) +
          dat(v, fi + 1, fj + 1, fk, bid_fine) + dat(v, fi + 1, fj + 1, fk + 1, bid_fine)) /
      8.0;
}

// ci, cj, ck are coarse indices, fi, fj, fk are fine indices
template <typename CellData_t, SlopeType SLOPE>
KOKKOS_FORCEINLINE_FUNCTION static void DoProlongation(const CellData_t &dat,
    int bid_fine,
    int bid_coarse,
    int v,
    int ci,
    int cj,
    int ck,
    int fi,
    int fj,
    int fk) {
  auto this_coarse = dat(v, ci, cj, ck, bid_coarse);
  double slope_x = 0.25 * Slope<SLOPE>::get(dat(v, ci - 1, cj, ck, bid_coarse),
                              this_coarse,
                              dat(v, ci + 1, cj, ck, bid_coarse));
  double slope_y = 0.25 * Slope<SLOPE>::get(dat(v, ci, cj - 1, ck, bid_coarse),
                              this_coarse,
                              dat(v, ci, cj + 1, ck, bid_coarse));
  double slope_z = 0.25 * Slope<SLOPE>::get(dat(v, ci, cj, ck - 1, bid_coarse),
                              this_coarse,
                              dat(v, ci, cj, ck + 1, bid_coarse));

  dat(v, fi, fj, fk, bid_fine) = this_coarse - slope_x - slope_y - slope_z;
  dat(v, fi, fj, fk + 1, bid_fine) = this_coarse - slope_x - slope_y + slope_z;
  dat(v, fi, fj + 1, fk, bid_fine) = this_coarse - slope_x + slope_y - slope_z;
  dat(v, fi, fj + 1, fk + 1, bid_fine) = this_coarse - slope_x + slope_y + slope_z;
  dat(v, fi + 1, fj, fk, bid_fine) = this_coarse + slope_x - slope_y - slope_z;
  dat(v, fi + 1, fj, fk + 1, bid_fine) = this_coarse + slope_x - slope_y + slope_z;
  dat(v, fi + 1, fj + 1, fk, bid_fine) = this_coarse + slope_x + slope_y - slope_z;
  dat(v, fi + 1, fj + 1, fk + 1, bid_fine) = this_coarse + slope_x + slope_y + slope_z;
}

/**
 * @brief Face between two blocks of different refinement levels.
 */
template <typename Block_t, SlopeType SLOPE>
struct InterLevelFace : public Face<Block_t> {
  using Face<Block_t>::N;
  using Face<Block_t>::N_ghost;
  using Face<Block_t>::N_vars;
  using typename Face<Block_t>::CellData_t;
  using typename Face<Block_t>::ExecSpace_t;
  using typename Face<Block_t>::team_member_t;

  static_assert((N_ghost % 2) == 0,
      "InterLevelFace is currently only implemented for even number of ghost cells");

  bool lower_coarse;
  int coarse_bid;
  Kokkos::Array<Kokkos::Array<int, 2>, 2> fine_bid;

  // we need a default constructor so we can make a Kokkos::View out of BoundaryFaces
  KOKKOS_FUNCTION InterLevelFace() {}

  KOKKOS_FUNCTION InterLevelFace(int8_t dir, Side coarse_side)
      : Face<Block_t>(dir), lower_coarse(coarse_side == Side::Lower), coarse_bid(-1) {
    fine_bid[0][0] = -1;
    fine_bid[0][1] = -1;
    fine_bid[1][0] = -1;
    fine_bid[1][1] = -1;
  }

  bool IsInitialized() const final override {
    if (coarse_bid < 0) return false;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        if (fine_bid[i][j] < 0) return false;
      }
    }
    return true;
  }

  // coarse block needs to be connected first
  void ConnectBlock(Side side, const Block_t *block) final override {
    if (lower_coarse == (side == Side::Lower)) {
      if (coarse_bid >= 0) {
        throw std::runtime_error("Trying to connect coarse block to a face that's already "
                                 "connected to a coarse block");
      }
      coarse_bid = block->id();
    } else {
      throw std::runtime_error("Trying to connect fine block with the wrong ConnectBlock method");
    }
  }

  void ConnectFineBlock(Side side, const Block_t *fine_block, const Block_t *coarse_block) {
    if (coarse_bid < 0)
      throw std::runtime_error("Trying to connect fine block before coarse block was connected");

    if (coarse_bid != coarse_block->id())
      throw std::runtime_error("Trying to connect fine block with a different coarse block");

    if (lower_coarse == (side == Side::Lower))
      throw std::runtime_error(
          "Trying to connect coarse block with the wrong ConnectFineBlock method");

    // this check fails for faces that connect blocks for opposite sides of the domain to
    // implement periodic boundary conditions

    // check coordinates match in the direction perpendicular to the face
    // double coarse_perp = coarse_block_->lower_bounds()[DIR] +
    //                      (COARSE_SIDE == Side::Lower ? coarse_block_->block_size()[DIR] : 0.0);
    // double fine_perp = block->lower_bounds()[DIR] +
    //                    (COARSE_SIDE == Side::Lower ? 0.0 : block->block_size()[DIR]);

    // printf("Coarse block %f, %f, %f, block size = %f, %f, %f\n",
    //     coarse_block_->lower_bounds()[0],
    //     coarse_block_->lower_bounds()[1],
    //     coarse_block_->lower_bounds()[2],
    //     coarse_block_->block_size()[0],
    //     coarse_block_->block_size()[1],
    //     coarse_block_->block_size()[2]);

    // printf("Fine block %f, %f, %f, block size = %f, %f, %f\n",
    //     block->lower_bounds()[0],
    //     block->lower_bounds()[1],
    //     block->lower_bounds()[2],
    //     block->block_size()[0],
    //     block->block_size()[1],
    //     block->block_size()[2]);

    // printf("Face dir %i, coarse_perp = %f, fine_perp = %f\n", DIR, coarse_perp, fine_perp);

    // if (fabs(coarse_perp - fine_perp) > 1.0e-12)
    //   throw std::runtime_error("Coarse and fine blocks coordinates don't match");

    // get logical coordinates of this block
    // get parallel directions
    const int PARA0 = (this->dir == 0) ? 1 : 0;
    const int PARA1 = (this->dir == 2) ? 1 : 2;

    Kokkos::Array<int, 2> idxs;
    for (int i = 0; i < 2; ++i) {
      int parallel_idx = (i == 0 ? PARA0 : PARA1);
      double diff = fabs(
          fine_block->lower_bounds()[parallel_idx] - coarse_block->lower_bounds()[parallel_idx]);

      if (diff < 1.0e-12) {
        idxs[i] = 0;
      } else if (fabs(diff - fine_block->block_size()[parallel_idx]) < 1.0e-12) {
        idxs[i] = 1;
      } else {
        std::string first_second = (i == 0 ? "first" : "second");
        throw std::runtime_error("Cannot determine " + first_second + " index of fine block");
      }
    }

    if (fine_bid[idxs[0]][idxs[1]] >= 0) {
      throw std::runtime_error("Fine block already set");
    }

    fine_bid[idxs[0]][idxs[1]] = fine_block->id();
  }

  template <bool DO_RESTRICT>
  KOKKOS_IMPL_FORCEINLINE static void RestrictOrProlongateGhostData(
      const CellData_t &dat, const Kokkos::View<InterLevelFace *, Kokkos::CudaSpace> &faces) {
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace_t>(faces.extent(0), Kokkos::AUTO()),
        KOKKOS_LAMBDA(team_member_t team_member) {
          int fid = team_member.league_rank();
          auto dir = faces(fid).dir;
          auto coarse_bid = faces(fid).coarse_bid;
          auto lower_coarse = faces(fid).lower_coarse;
          const auto &fine_bids = faces(fid).fine_bid;

          DoRestrictOrProlongate<DO_RESTRICT>(
              team_member, dat, 0, 0, coarse_bid, fine_bids[0][0], lower_coarse, dir);
          DoRestrictOrProlongate<DO_RESTRICT>(
              team_member, dat, 0, 1, coarse_bid, fine_bids[0][1], lower_coarse, dir);
          DoRestrictOrProlongate<DO_RESTRICT>(
              team_member, dat, 1, 0, coarse_bid, fine_bids[1][0], lower_coarse, dir);
          DoRestrictOrProlongate<DO_RESTRICT>(
              team_member, dat, 1, 1, coarse_bid, fine_bids[1][1], lower_coarse, dir);
        });
  }

  KOKKOS_IMPL_FORCEINLINE static void RestrictGhostData(
      const CellData_t &dat, const Kokkos::View<InterLevelFace *, Kokkos::CudaSpace> &faces) {
    RestrictOrProlongateGhostData<true>(dat, faces);
  }

  KOKKOS_IMPL_FORCEINLINE static void ProlongateGhostData(
      const CellData_t &dat, const Kokkos::View<InterLevelFace *, Kokkos::CudaSpace> &faces) {
    RestrictOrProlongateGhostData<false>(dat, faces);
  }

private:
  template <bool DO_RESTRICT>
  KOKKOS_FORCEINLINE_FUNCTION static void DoRestrictOrProlongate(const team_member_t &team_member,
      const CellData_t &dat,
      int i,
      int j,
      int bid_coarse,
      int bid_fine,
      bool lower_coarse,
      int dir) {
    constexpr int loop_count = N_vars * (N / 2) * (N / 2) * (DO_RESTRICT ? N_ghost : (N_ghost / 2));

    Kokkos::parallel_for(TeamThreadRange<>(team_member, loop_count), [=](int idx) {
      // idx = v + i * N_v + j * N_v * N_i + k * N_v * N_i * N_j
      // Rely on compiler optimization of modulus since divisor is compile-time constant
      int v = idx % N_vars;
      idx /= N_vars;

      int p = idx % (N / 2);
      idx /= (N / 2);

      int q = idx % (N / 2);
      int g = idx / (N / 2);

      int coarse_idx = DO_RESTRICT ? (lower_coarse ? N + N_ghost + g : g)
                                   : (lower_coarse ? N + N_ghost / 2 + g : N_ghost + g);
      int fine_idx = DO_RESTRICT ? (lower_coarse ? N_ghost + 2 * g : N - N_ghost + 2 * g)
                                 : (lower_coarse ? 2 * g : N + N_ghost + 2 * g);

      // coarse indices
      int pc = N_ghost + i * N / 2 + p;
      int qc = N_ghost + j * N / 2 + q;

      // fine indices
      int pf = N_ghost + 2 * p;
      int qf = N_ghost + 2 * q;

      int xc = dir == 0 ? coarse_idx : pc;
      int yc = dir == 1 ? coarse_idx : (dir == 0 ? pc : qc);
      int zc = dir == 2 ? coarse_idx : qc;

      int xf = dir == 0 ? fine_idx : pf;
      int yf = dir == 1 ? fine_idx : (dir == 0 ? pf : qf);
      int zf = dir == 2 ? fine_idx : qf;

      if (DO_RESTRICT)
        DoRestriction<CellData_t>(dat, bid_coarse, bid_fine, v, xc, yc, zc, xf, yf, zf);
      else
        DoProlongation<CellData_t, SLOPE>(dat, bid_fine, bid_coarse, v, xc, yc, zc, xf, yf, zf);
    });
  }
};

template <typename Block_t>
struct BoundaryFace : public Face<Block_t> {
  using Face<Block_t>::N;
  using Face<Block_t>::N_ghost;
  using Face<Block_t>::N_tot;
  using Face<Block_t>::N_vars;
  using typename Face<Block_t>::CellData_t;
  using typename Face<Block_t>::ExecSpace_t;
  using typename Face<Block_t>::team_member_t;

  bool lower_side;
  BoundaryConditionType bc_type;
  int block_id;

  // we need a default constructor so we can make a Kokkos::View out of BoundaryFaces
  KOKKOS_FUNCTION BoundaryFace() {}

  KOKKOS_FUNCTION BoundaryFace(int8_t dir, Side side, BoundaryConditionType bc_type)
      : Face<Block_t>(dir), lower_side(side == Side::Lower), bc_type(bc_type), block_id(-1) {}

  bool IsInitialized() const final override {
    if (block_id < 0) return false;
    return true;
  }

  void ConnectBlock(Side side, const Block_t *block) final override {
    // if this is a LOWER face, the neighboring block is on the UPPER side and vice versa
    if (lower_side == (side == Side::Lower)) {
      throw std::invalid_argument("Side mismatch in Boundary face");
    }

    if (block_id >= 0) {
      throw std::runtime_error(
          "Trying to connect block to a face that's already connected to a block");
    }

    block_id = block->id();
  }

  static void ExchangeGhostData(
      const CellData_t &dat, const Kokkos::View<BoundaryFace *, Kokkos::CudaSpace> &faces) {
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace_t>(faces.extent(0), Kokkos::AUTO()),
        KOKKOS_LAMBDA(team_member_t team_member) {
          int fid = team_member.league_rank();
          int bid = faces(fid).block_id;
          int dir = faces(fid).dir;
          bool lower_side = faces(fid).lower_side;
          auto bc_type = faces(fid).bc_type;

          Kokkos::Array<int, N_ghost> idxs_to, idxs_from;

          if (lower_side) {
            for (int i = 0; i < N_ghost; ++i) {
              idxs_to[i] = i;

              if (bc_type == BoundaryConditionType::Outflow) {
                // just copy logical cell 0
                idxs_from[i] = 0 + N_ghost;
              } else if (bc_type == BoundaryConditionType::Reflecting) {
                // logical index of ghost cell: -1, -2, -3, ..., -N_ghost
                int G = i - N_ghost;
                // copy logical cell -(G + 1) and invert if it's velocity component in this
                // direction
                idxs_from[i] = -(G + 1) + N_ghost;
              }
            }
          } else {
            // replace i with ip so the loop can be unrolled
            for (int ip = 0; ip < N_ghost; ++ip) {
              int i = ip + N_tot - N_ghost;
              idxs_to[ip] = i;

              if (bc_type == BoundaryConditionType::Outflow) {
                // just copy logical cell N - 1
                idxs_from[ip] = (N - 1) + N_ghost;
              } else if (bc_type == BoundaryConditionType::Reflecting) {
                // logical index of ghost cell: N, N+1, N+2, ..., N+N_ghost-1
                int G = i - N_ghost;
                // copy logical cell N - (G - N + 1) and invert if it's velocity component in this
                // direction
                idxs_from[ip] = (N - (G - N + 1)) + N_ghost;
              }
            }
          }

          if (bc_type == BoundaryConditionType::Reflecting)
            Face<Block_t>::template CopyPlanes<true>(
                team_member, dat, bid, bid, dir, idxs_to, idxs_from);
          else
            Face<Block_t>::template CopyPlanes<false>(
                team_member, dat, bid, bid, dir, idxs_to, idxs_from);
        });
  }
};

#endif // ETHON_KOKKOS_CUDA_FACE_HPP_
