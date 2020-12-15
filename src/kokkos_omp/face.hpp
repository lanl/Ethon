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

#ifndef ETHON_KOKKOS_OMP_FACE_HPP_
#define ETHON_KOKKOS_OMP_FACE_HPP_

#include <array>
#include <memory>

#include <Kokkos_Core.hpp>

#include "state.hpp"
#include "slopes.hpp"
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
  KOKKOS_FORCEINLINE_FUNCTION static void CopyPlane(
      const CellData_t &dat, int bid_to, int bid_from, int dir, int idx_to, int idx_from, int v) {
    for (int q = N_ghost; q < N + N_ghost; ++q) {
      for (int p = N_ghost; p < N + N_ghost; ++p) {
        if (dir == 0) {
          dat(v, idx_to, p, q, bid_to) = (INVERT ? -1.0 : 1.0) * dat(v, idx_from, p, q, bid_from);
        } else if (dir == 1) {
          dat(v, p, idx_to, q, bid_to) = (INVERT ? -1.0 : 1.0) * dat(v, p, idx_from, q, bid_from);
        } else if (dir == 2) {
          dat(v, p, q, idx_to, bid_to) = (INVERT ? -1.0 : 1.0) * dat(v, p, q, idx_from, bid_from);
        }
      }
    }
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
      const CellData_t &dat, const Kokkos::View<IntercellFace *, Kokkos::HostSpace> &faces) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace_t>(0, faces.extent(0)), KOKKOS_LAMBDA(int fid) {
          for (int v = 0; v < N_vars; ++v)
            CopyData(dat, faces(fid).lower_bid, faces(fid).upper_bid, faces(fid).dir, v);
        });
  }

private:
  KOKKOS_FORCEINLINE_FUNCTION static void CopyData(
      const CellData_t &dat, int bid_lower, int bid_upper, int dir, int v) {
    for (int i = 0; i < N_ghost; ++i) {
      Face<Block_t>::CopyPlane(dat, bid_upper, bid_lower, dir, i, N + i, v);
      Face<Block_t>::CopyPlane(dat, bid_lower, bid_upper, dir, N + N_ghost + i, N_ghost + i, v);
    }
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
        throw std::runtime_error(
            "Trying to connect coarse block to a face that's already connected to a coarse block");
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
      const CellData_t &dat, const Kokkos::View<InterLevelFace *, Kokkos::HostSpace> &faces) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace_t>(0, faces.extent(0)), KOKKOS_LAMBDA(int fid) {
          auto dir = faces(fid).dir;
          auto coarse_bid = faces(fid).coarse_bid;
          auto lower_coarse = faces(fid).lower_coarse;
          const auto &fine_bids = faces(fid).fine_bid;
          for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
              for (int v = 0; v < N_vars; ++v) {
                DoRestrictOrProlongate<DO_RESTRICT>(
                    dat, i, j, coarse_bid, fine_bids[i][j], lower_coarse, dir, v);
              }
            }
          }
        });
  }

  KOKKOS_IMPL_FORCEINLINE static void RestrictGhostData(
      const CellData_t &dat, const Kokkos::View<InterLevelFace *, Kokkos::HostSpace> &faces) {
    RestrictOrProlongateGhostData<true>(dat, faces);
  }

  KOKKOS_IMPL_FORCEINLINE static void ProlongateGhostData(
      const CellData_t &dat, const Kokkos::View<InterLevelFace *, Kokkos::HostSpace> &faces) {
    RestrictOrProlongateGhostData<false>(dat, faces);
  }

private:
  template <bool DO_RESTRICT>
  KOKKOS_FORCEINLINE_FUNCTION static void DoRestrictOrProlongate(const CellData_t &dat,
      int i,
      int j,
      int bid_coarse,
      int bid_fine,
      bool lower_coarse,
      int dir,
      int v) {
    if (DO_RESTRICT) {
      // restrict fine cells to coarse ghost cells
      for (int g = 0; g < N_ghost; ++g) {
        int coarse_idx = (lower_coarse ? N + N_ghost + g : g);
        int fine_idx = (lower_coarse ? N_ghost + 2 * g : N - N_ghost + 2 * g);

        for (int q = 0; q < N / 2; ++q) {
          for (int p = 0; p < N / 2; ++p) {
            // coarse indices
            int pc = N_ghost + i * N / 2 + p;
            int qc = N_ghost + j * N / 2 + q;

            // fine indices
            int pf = N_ghost + 2 * p;
            int qf = N_ghost + 2 * q;

            if (dir == 0) {
              DoRestriction<CellData_t>(
                  dat, bid_coarse, bid_fine, v, coarse_idx, pc, qc, fine_idx, pf, qf);
            } else if (dir == 1) {
              DoRestriction<CellData_t>(
                  dat, bid_coarse, bid_fine, v, pc, coarse_idx, qc, pf, fine_idx, qf);
            } else if (dir == 2) {
              DoRestriction<CellData_t>(
                  dat, bid_coarse, bid_fine, v, pc, qc, coarse_idx, pf, qf, fine_idx);
            }
          }
        }
      }
    } else {
      // prolongate coarse cells to fine ghost cells
      for (int g = 0; g < N_ghost / 2; ++g) {
        int coarse_idx = (lower_coarse ? N + N_ghost / 2 + g : N_ghost + g);
        int fine_idx = (lower_coarse ? 2 * g : N + N_ghost + 2 * g);

        for (int q = 0; q < N / 2; ++q) {
          for (int p = 0; p < N / 2; ++p) {
            // coarse indices
            int pc = N_ghost + i * N / 2 + p;
            int qc = N_ghost + j * N / 2 + q;

            // fine indices
            int pf = N_ghost + 2 * p;
            int qf = N_ghost + 2 * q;

            if (dir == 0) {
              DoProlongation<CellData_t, SLOPE>(
                  dat, bid_fine, bid_coarse, v, coarse_idx, pc, qc, fine_idx, pf, qf);
            } else if (dir == 1) {
              DoProlongation<CellData_t, SLOPE>(
                  dat, bid_fine, bid_coarse, v, pc, coarse_idx, qc, pf, fine_idx, qf);
            } else if (dir == 2) {
              DoProlongation<CellData_t, SLOPE>(
                  dat, bid_fine, bid_coarse, v, pc, qc, coarse_idx, pf, qf, fine_idx);
            }
          }
        }
      }
    }
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
      const CellData_t &dat, const Kokkos::View<BoundaryFace *, Kokkos::HostSpace> &faces) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace_t>(0, faces.extent(0)), KOKKOS_LAMBDA(int fid) {
          auto dir = faces(fid).dir;
          auto lower_side = faces(fid).lower_side;
          auto bc_type = faces(fid).bc_type;
          auto bid = faces(fid).block_id;

          for (int v = 0; v < N_vars; ++v) {
            if (v == (dir + 1))
              ApplyBC<true>(dat, lower_side, bc_type, dir, bid, v);
            else
              ApplyBC<false>(dat, lower_side, bc_type, dir, bid, v);
          }
        });
  }

private:
  template <bool INVERT>
  KOKKOS_FORCEINLINE_FUNCTION static void ApplyBC(const CellData_t &dat,
      bool lower_side,
      BoundaryConditionType bc_type,
      int dir,
      int bid,
      int v) {
    if (lower_side) {
      for (int i = 0; i < N_ghost; ++i) {
        int64_t G = i - N_ghost; // logical index of ghost cell: -1, -2, -3, ..., -N_ghost
        if (bc_type == BoundaryConditionType::Outflow) {
          // just copy logical cell 0
          Face<Block_t>::CopyPlane(dat, bid, bid, dir, i, 0 + N_ghost, v);
        } else if (bc_type == BoundaryConditionType::Reflecting) {
          // copy logical cell -(G + 1) and invert if it's velocity component in this direction
          Face<Block_t>::template CopyPlane<INVERT>(dat, bid, bid, dir, i, -(G + 1) + N_ghost, v);
        }
      }
    } else {
      // replace i with ip so the loop can be unrolled
      for (int ip = 0; ip < N_ghost; ++ip) {
        int i = ip + N_tot - N_ghost;
        int64_t G = i - N_ghost; // logical index of ghost cell: N, N+1, N+2, ..., N+N_ghost-1
        if (bc_type == BoundaryConditionType::Outflow) {
          // just copy logical cell N - 1
          Face<Block_t>::CopyPlane(dat, bid, bid, dir, i, (N - 1) + N_ghost, v);
        } else if (bc_type == BoundaryConditionType::Reflecting) {
          // copy logical cell N - (G - N + 1) and invert if it's velocity component in this
          // direction
          Face<Block_t>::template CopyPlane<INVERT>(
              dat, bid, bid, dir, i, (N - (G - N + 1)) + N_ghost, v);
        }
      }
    }
  }
};

#endif // ETHON_KOKKOS_OMP_FACE_HPP_
