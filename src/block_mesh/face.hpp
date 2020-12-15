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

#ifndef ETHON_BLOCK_MESH_FACE_HPP_
#define ETHON_BLOCK_MESH_FACE_HPP_

#include <array>
#include <memory>

#include <Kokkos_Core.hpp>

#include "block_mesh/block_array.hpp"
#include "slopes.hpp"
#include "uniform_mesh/boundary.hpp"

using namespace Kokkos;

template <typename BLOCK>
class Face {
public:
  static constexpr size_t N = BLOCK::N;
  static constexpr size_t N_ghost = BLOCK::N_ghost;
  static constexpr size_t N_tot = BLOCK::N_tot;

  using BlockArray = typename BLOCK::BlockArray;

  /**
   * @brief Return true if this face is fully initialized (has all the required blocks connected)
   */
  virtual bool IsInitialized() const = 0;

  virtual void ConnectBlock(const Side side, BLOCK *block) = 0;

  virtual void ExchangeGhostData() const = 0;
  virtual void RestrictGhostData() const = 0;
  virtual void ProlongateGhostData() const = 0;

protected:
  template <size_t DIR, bool INVERT = false>
  KOKKOS_IMPL_FORCEINLINE static void CopyPlane(
      BlockArray *dat_to, const BlockArray *dat_from, const size_t idx_to, const size_t idx_from) {
    for (size_t q = N_ghost; q < N + N_ghost; ++q) {
      for (size_t p = N_ghost; p < N + N_ghost; ++p) {
        if (DIR == 0) {
          (*dat_to)(idx_to, p, q) = (INVERT ? -1.0 : 1.0) * (*dat_from)(idx_from, p, q);
        } else if (DIR == 1) {
          (*dat_to)(p, idx_to, q) = (INVERT ? -1.0 : 1.0) * (*dat_from)(p, idx_from, q);
        } else if (DIR == 2) {
          (*dat_to)(p, q, idx_to) = (INVERT ? -1.0 : 1.0) * (*dat_from)(p, q, idx_from);
        }
      }
    }
  }
};

/**
 * @brief A face between two blocks of the same refinement level
 */
template <typename BLOCK, size_t DIR>
class IntercellFace : public Face<BLOCK> {
  static_assert((DIR >= 0) && (DIR <= 2), "Face direction can only be 0, 1, 2");
  using Face<BLOCK>::N;
  using Face<BLOCK>::N_ghost;
  using typename Face<BLOCK>::BlockArray;

public:
  bool IsInitialized() const final override {
    if (!lower_block_) return false;
    if (!upper_block_) return false;
    return true;
  }

  void ConnectBlock(const Side side, BLOCK *block) final override {
    if (side == Side::Lower) {
      if (lower_block_) {
        throw std::runtime_error(
            "Trying to connect block to a face that's already connected to a block");
      }
      lower_block_ = block;
    } else if (side == Side::Upper) {
      if (upper_block_) {
        throw std::runtime_error(
            "Trying to connect block to a face that's already connected to a block");
      }
      upper_block_ = block;
    } else {
      throw std::runtime_error("Unknown side");
    }
  }

  KOKKOS_IMPL_FORCEINLINE void ExchangeGhostData() const final override {
    CopyData(&lower_block_->rho(), &upper_block_->rho());
    CopyData(&lower_block_->mu0(), &upper_block_->mu0());
    CopyData(&lower_block_->mu1(), &upper_block_->mu1());
    CopyData(&lower_block_->mu2(), &upper_block_->mu2());
    CopyData(&lower_block_->epsilon(), &upper_block_->epsilon());
  }

  KOKKOS_IMPL_FORCEINLINE void RestrictGhostData() const final override {
    throw std::runtime_error("Cannot restrict ghost data with an IntercellFace");
  }

  KOKKOS_IMPL_FORCEINLINE void ProlongateGhostData() const final override {
    throw std::runtime_error("Cannot prolongate ghost data with an IntercellFace");
  }

private:
  KOKKOS_IMPL_FORCEINLINE void CopyData(BlockArray *dat_lower, BlockArray *dat_upper) const {
    for (size_t i = 0; i < N_ghost; ++i) {
      this->template CopyPlane<DIR>(dat_upper, dat_lower, i, N + i);
      this->template CopyPlane<DIR>(dat_lower, dat_upper, N + N_ghost + i, N_ghost + i);
    }
  }

  BLOCK *lower_block_, *upper_block_;
};

template <typename BLOCK>
std::shared_ptr<Face<BLOCK>> CreateIntercellFace(const size_t dir) {
  if (dir == 0) {
    return std::shared_ptr<Face<BLOCK>>(new IntercellFace<BLOCK, 0>());
  } else if (dir == 1) {
    return std::shared_ptr<Face<BLOCK>>(new IntercellFace<BLOCK, 1>());
  } else if (dir == 2) {
    return std::shared_ptr<Face<BLOCK>>(new IntercellFace<BLOCK, 2>());
  } else {
    throw std::invalid_argument("Unknown face direction " + std::to_string(dir));
  }
}

// ci, cj, ck are coarse indices, fi, fj, fk are fine indices
template <typename BlockArray>
KOKKOS_IMPL_FORCEINLINE void DoRestriction(BlockArray *dat_coarse,
    const BlockArray *dat_fine,
    int ci,
    int cj,
    int ck,
    int fi,
    int fj,
    int fk) {
  const BlockArray &fine = *dat_fine;
  (*dat_coarse)(ci, cj, ck) =
      (fine(fi, fj, fk) + fine(fi, fj, fk + 1) + fine(fi, fj + 1, fk) + fine(fi, fj + 1, fk + 1) +
          fine(fi + 1, fj, fk) + fine(fi + 1, fj, fk + 1) + fine(fi + 1, fj + 1, fk) +
          fine(fi + 1, fj + 1, fk + 1)) /
      8.0;
}

// ci, cj, ck are coarse indices, fi, fj, fk are fine indices
template <typename BlockArray, SlopeType SLOPE>
KOKKOS_IMPL_FORCEINLINE void DoProlongation(BlockArray *dat_fine,
    const BlockArray *dat_coarse,
    int ci,
    int cj,
    int ck,
    int fi,
    int fj,
    int fk) {
  const BlockArray &coarse = *dat_coarse;

  auto this_coarse = coarse(ci, cj, ck);
  double slope_x =
      0.25 * Slope<SLOPE>::get(coarse(ci - 1, cj, ck), this_coarse, coarse(ci + 1, cj, ck));
  double slope_y =
      0.25 * Slope<SLOPE>::get(coarse(ci, cj - 1, ck), this_coarse, coarse(ci, cj + 1, ck));
  double slope_z =
      0.25 * Slope<SLOPE>::get(coarse(ci, cj, ck - 1), this_coarse, coarse(ci, cj, ck + 1));

  (*dat_fine)(fi, fj, fk) = this_coarse - slope_x - slope_y - slope_z;
  (*dat_fine)(fi, fj, fk + 1) = this_coarse - slope_x - slope_y + slope_z;
  (*dat_fine)(fi, fj + 1, fk) = this_coarse - slope_x + slope_y - slope_z;
  (*dat_fine)(fi, fj + 1, fk + 1) = this_coarse - slope_x + slope_y + slope_z;
  (*dat_fine)(fi + 1, fj, fk) = this_coarse + slope_x - slope_y - slope_z;
  (*dat_fine)(fi + 1, fj, fk + 1) = this_coarse + slope_x - slope_y + slope_z;
  (*dat_fine)(fi + 1, fj + 1, fk) = this_coarse + slope_x + slope_y - slope_z;
  (*dat_fine)(fi + 1, fj + 1, fk + 1) = this_coarse + slope_x + slope_y + slope_z;
}

/**
 * @brief Face between two blocks of different refinement levels.
 */
template <typename BLOCK, size_t DIR, Side COARSE_SIDE, SlopeType SLOPE>
class InterLevelFace : public Face<BLOCK> {
  static_assert((DIR >= 0) && (DIR <= 2), "Face direction can only be 0, 1, 2");
  using Face<BLOCK>::N;
  using Face<BLOCK>::N_ghost;
  using typename Face<BLOCK>::BlockArray;

  static_assert((N_ghost % 2) == 0,
      "InterLevelFace is currently only implemented for even number of ghost cells");

  // get parallel directions
  static constexpr int PARA0 = (DIR == 0) ? 1 : 0;
  static constexpr int PARA1 = (DIR == 2) ? 1 : 2;

public:
  bool IsInitialized() const final override {
    if (!coarse_block_) return false;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        if (!fine_blocks_[i][j]) return false;
      }
    }
    return true;
  }

  // coarse block needs to be connected first
  void ConnectBlock(const Side side, BLOCK *block) final override {
    if (side == COARSE_SIDE) {
      if (coarse_block_) {
        throw std::runtime_error(
            "Trying to connect coarse block to a face that's already connected to a coarse block");
      }
      coarse_block_ = block;
    } else {
      if (!coarse_block_) {
        throw std::runtime_error(
            "Trying to connect fine block to a face before coarse block was connected");
      }

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
      std::array<int, 2> idxs;
      for (int i = 0; i < 2; ++i) {
        int parallel_idx = (i == 0 ? PARA0 : PARA1);
        double diff =
            fabs(block->lower_bounds()[parallel_idx] - coarse_block_->lower_bounds()[parallel_idx]);

        if (diff < 1.0e-12) {
          idxs[i] = 0;
        } else if (fabs(diff - block->block_size()[parallel_idx]) < 1.0e-12) {
          idxs[i] = 1;
        } else {
          std::string first_second = (i == 0 ? "first" : "second");
          throw std::runtime_error("Cannot determine " + first_second + " index of fine block");
        }
      }

      if (fine_blocks_[idxs[0]][idxs[1]]) {
        throw std::runtime_error("Fine block already set");
      }

      fine_blocks_[idxs[0]][idxs[1]] = block;
    }
  }

  KOKKOS_IMPL_FORCEINLINE void ExchangeGhostData() const final override {
    throw std::runtime_error("ExchangeGhostData cannot be called on InterLevelFace. Need to call "
                             "RestrictGhostData and ProlongateGhostDate");
  }

  template <bool DO_RESTRICT>
  KOKKOS_IMPL_FORCEINLINE void RestrictOrProlongateGhostData() const {
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        DoRestrictOrProlongate<DO_RESTRICT>(
            i, j, &coarse_block_->rho(), &fine_blocks_[i][j]->rho());
        DoRestrictOrProlongate<DO_RESTRICT>(
            i, j, &coarse_block_->mu0(), &fine_blocks_[i][j]->mu0());
        DoRestrictOrProlongate<DO_RESTRICT>(
            i, j, &coarse_block_->mu1(), &fine_blocks_[i][j]->mu1());
        DoRestrictOrProlongate<DO_RESTRICT>(
            i, j, &coarse_block_->mu2(), &fine_blocks_[i][j]->mu2());
        DoRestrictOrProlongate<DO_RESTRICT>(
            i, j, &coarse_block_->epsilon(), &fine_blocks_[i][j]->epsilon());
      }
    }
  }

  KOKKOS_IMPL_FORCEINLINE void RestrictGhostData() const final override {
    RestrictOrProlongateGhostData<true>();
  }

  KOKKOS_IMPL_FORCEINLINE void ProlongateGhostData() const final override {
    RestrictOrProlongateGhostData<false>();
  }

private:
  template <bool DO_RESTRICT>
  KOKKOS_IMPL_FORCEINLINE void DoRestrictOrProlongate(
      const int i, const int j, BlockArray *dat_coarse, BlockArray *dat_fine) const {
    if (DO_RESTRICT) {
      // restrict fine cells to coarse ghost cells
      for (size_t g = 0; g < N_ghost; ++g) {
        int coarse_idx = (COARSE_SIDE == Side::Lower ? N + N_ghost + g : g);
        int fine_idx = (COARSE_SIDE == Side::Lower ? N_ghost + 2 * g : N - N_ghost + 2 * g);

        for (size_t q = 0; q < N / 2; ++q) {
          for (size_t p = 0; p < N / 2; ++p) {
            // coarse indices
            int pc = N_ghost + i * N / 2 + p;
            int qc = N_ghost + j * N / 2 + q;

            // fine indices
            int pf = N_ghost + 2 * p;
            int qf = N_ghost + 2 * q;

            if (DIR == 0) {
              DoRestriction<BlockArray>(dat_coarse, dat_fine, coarse_idx, pc, qc, fine_idx, pf, qf);
            } else if (DIR == 1) {
              DoRestriction<BlockArray>(dat_coarse, dat_fine, pc, coarse_idx, qc, pf, fine_idx, qf);
            } else if (DIR == 2) {
              DoRestriction<BlockArray>(dat_coarse, dat_fine, pc, qc, coarse_idx, pf, qf, fine_idx);
            }
          }
        }
      }
    } else {
      // prolongate coarse cells to fine ghost cells
      for (size_t g = 0; g < N_ghost / 2; ++g) {
        int coarse_idx = (COARSE_SIDE == Side::Lower ? N + N_ghost / 2 + g : N_ghost + g);
        int fine_idx = (COARSE_SIDE == Side::Lower ? 2 * g : N + N_ghost + 2 * g);

        for (size_t q = 0; q < N / 2; ++q) {
          for (size_t p = 0; p < N / 2; ++p) {
            // coarse indices
            int pc = N_ghost + i * N / 2 + p;
            int qc = N_ghost + j * N / 2 + q;

            // fine indices
            int pf = N_ghost + 2 * p;
            int qf = N_ghost + 2 * q;

            if (DIR == 0) {
              DoProlongation<BlockArray, SLOPE>(
                  dat_fine, dat_coarse, coarse_idx, pc, qc, fine_idx, pf, qf);
            } else if (DIR == 1) {
              DoProlongation<BlockArray, SLOPE>(
                  dat_fine, dat_coarse, pc, coarse_idx, qc, pf, fine_idx, qf);
            } else if (DIR == 2) {
              DoProlongation<BlockArray, SLOPE>(
                  dat_fine, dat_coarse, pc, qc, coarse_idx, pf, qf, fine_idx);
            }
          }
        }
      }
    }
  }

  BLOCK *coarse_block_;

  std::array<std::array<BLOCK *, 2>, 2> fine_blocks_;
};

template <typename BLOCK, Side COARSE_SIDE, SlopeType SLOPE>
std::shared_ptr<Face<BLOCK>> CreateInterLevelFace(const size_t dir) {
  if (dir == 0) {
    return std::shared_ptr<Face<BLOCK>>(new InterLevelFace<BLOCK, 0, COARSE_SIDE, SLOPE>());
  } else if (dir == 1) {
    return std::shared_ptr<Face<BLOCK>>(new InterLevelFace<BLOCK, 1, COARSE_SIDE, SLOPE>());
  } else if (dir == 2) {
    return std::shared_ptr<Face<BLOCK>>(new InterLevelFace<BLOCK, 2, COARSE_SIDE, SLOPE>());
  } else {
    throw std::invalid_argument("Unknown face direction " + std::to_string(dir));
  }
}

template <typename BLOCK, SlopeType SLOPE>
std::shared_ptr<Face<BLOCK>> CreateInterLevelFace(const size_t dir, const Side coarse_side) {
  if (coarse_side == Side::Lower) {
    return CreateInterLevelFace<BLOCK, Side::Lower, SLOPE>(dir);
  } else if (coarse_side == Side::Upper) {
    return CreateInterLevelFace<BLOCK, Side::Upper, SLOPE>(dir);
  } else {
    throw std::invalid_argument("Coarse side must be lower or upper");
  }
}

template <typename BLOCK, size_t DIR, BoundaryConditionType BC_TYPE, Side SIDE>
class BoundaryFace : public Face<BLOCK> {
  static_assert((DIR >= 0) && (DIR <= 2), "Face direction can only be 0, 1, 2");
  using Face<BLOCK>::N;
  using Face<BLOCK>::N_ghost;
  using Face<BLOCK>::N_tot;
  using typename Face<BLOCK>::BlockArray;

public:
  bool IsInitialized() const final override {
    if (!block_) return false;
    return true;
  }

  void ConnectBlock(const Side side, BLOCK *block) final override {
    // if this is a LOWER face, the neighboring block is on the UPPER side and vice versa
    if (side == SIDE) {
      throw std::invalid_argument("Side mismatch in Boundary face");
    }

    if (block_) {
      throw std::runtime_error(
          "Trying to connect block to a face that's already connected to a block");
    }

    block_ = block;
  }

  void ExchangeGhostData() const final override {
    ApplyBC<false>(&block_->rho());

    if (DIR == 0) {
      ApplyBC<true>(&block_->mu0());
    } else {
      ApplyBC<false>(&block_->mu0());
    }

    if (DIR == 1) {
      ApplyBC<true>(&block_->mu1());
    } else {
      ApplyBC<false>(&block_->mu1());
    }

    if (DIR == 2) {
      ApplyBC<true>(&block_->mu2());
    } else {
      ApplyBC<false>(&block_->mu2());
    }

    ApplyBC<false>(&block_->epsilon());
  }

  KOKKOS_IMPL_FORCEINLINE void RestrictGhostData() const final override {
    throw std::runtime_error("Cannot restrict ghost data with a BoundaryFace");
  }

  KOKKOS_IMPL_FORCEINLINE void ProlongateGhostData() const final override {
    throw std::runtime_error("Cannot prolongate ghost data with a BoundaryFace");
  }

private:
  template <bool INVERT>
  KOKKOS_IMPL_FORCEINLINE void ApplyBC(BlockArray *dat) const {
    if (SIDE == Side::Lower) {
      for (size_t i = 0; i < N_ghost; ++i) {
        int64_t G = i - N_ghost; // logical index of ghost cell: -1, -2, -3, ..., -N_ghost
        if (BC_TYPE == BoundaryConditionType::Outflow) {
          // just copy logical cell 0
          this->template CopyPlane<DIR>(dat, dat, i, 0 + N_ghost);
        } else if (BC_TYPE == BoundaryConditionType::Reflecting) {
          // copy logical cell -(G + 1) and invert if it's velocity component in this direction
          this->template CopyPlane<DIR, INVERT>(dat, dat, i, -(G + 1) + N_ghost);
        }
      }
    } else if (SIDE == Side::Upper) {
      // replace i with ip so the loop can be unrolled
      for (size_t ip = 0; ip < N_ghost; ++ip) {
        size_t i = ip + N_tot - N_ghost;
        int64_t G = i - N_ghost; // logical index of ghost cell: N, N+1, N+2, ..., N+N_ghost-1
        if (BC_TYPE == BoundaryConditionType::Outflow) {
          // just copy logical cell N - 1
          this->template CopyPlane<DIR>(dat, dat, i, (N - 1) + N_ghost);
        } else if (BC_TYPE == BoundaryConditionType::Reflecting) {
          // copy logical cell N - (G - N + 1) and invert if it's velocity component in this
          // direction
          this->template CopyPlane<DIR, INVERT>(dat, dat, i, (N - (G - N + 1)) + N_ghost);
        }
      }
    }
  }

  BLOCK *block_;
};

template <typename BLOCK, size_t DIR, Side SIDE>
std::shared_ptr<Face<BLOCK>> CreateBoundaryFace(const BoundaryConditionType bc_type) {
  if (bc_type == BoundaryConditionType::Outflow) {
    return std::shared_ptr<Face<BLOCK>>(
        new BoundaryFace<BLOCK, DIR, BoundaryConditionType::Outflow, SIDE>());
  } else if (bc_type == BoundaryConditionType::Reflecting) {
    return std::shared_ptr<Face<BLOCK>>(
        new BoundaryFace<BLOCK, DIR, BoundaryConditionType::Reflecting, SIDE>());
  } else if (bc_type == BoundaryConditionType::Periodic) {
    throw std::invalid_argument("Periodic boundary condition is handled by IntercellFace");
  } else {
    throw std::invalid_argument("Unknown boundary condition");
  }
}

template <typename BLOCK, size_t DIR>
std::shared_ptr<Face<BLOCK>> CreateBoundaryFace(
    const Side side, const BoundaryConditionType bc_type) {
  if (side == Side::Lower) {
    return CreateBoundaryFace<BLOCK, DIR, Side::Lower>(bc_type);
  } else if (side == Side::Upper) {
    return CreateBoundaryFace<BLOCK, DIR, Side::Upper>(bc_type);
  } else {
    throw std::invalid_argument("Unknown face side");
  }
}

template <typename BLOCK>
std::shared_ptr<Face<BLOCK>> CreateBoundaryFace(
    const size_t dir, const Side side, const BoundaryConditionType bc_type) {
  if (dir == 0) {
    return CreateBoundaryFace<BLOCK, 0>(side, bc_type);
  } else if (dir == 1) {
    return CreateBoundaryFace<BLOCK, 1>(side, bc_type);
  } else if (dir == 2) {
    return CreateBoundaryFace<BLOCK, 2>(side, bc_type);
  } else {
    throw std::invalid_argument("Unknown face direction " + std::to_string(dir));
  }
}

#endif // ETHON_BLOCK_MESH_FACE_HPP_
