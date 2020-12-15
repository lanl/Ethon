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
//========================================================================================kokkos_cuda/

#ifndef ETHON_KOKKOS_CUDA_AMR_HPP_
#define ETHON_KOKKOS_CUDA_AMR_HPP_

#include <bitset>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include <p8est_connectivity.h>
#include <p8est_extended.h>
#include <p8est_iterate.h>

#include "kokkos_cuda/face.hpp"

void my_assert(bool condition, std::string message) {
  if (!condition) throw std::runtime_error(message);
}

namespace AMR {

/**
 * @brief Create a new Block for the given quadrant (without setting any data on it)
 */
template <typename DRIVER>
static void CreateBlock(p8est_t *p8est, p4est_topidx_t which_tree, p8est_quadrant_t *quadrant) {
  double vertex[3];
  p8est_qcoord_to_vertex(
      p8est->connectivity, which_tree, quadrant->x, quadrant->y, quadrant->z, vertex);

  auto driver = static_cast<DRIVER *>(p8est->user_pointer);

  auto lower_bounds = driver->mesh()->TransformP4estVertex(vertex);
  auto cell_size = driver->mesh()->cell_size(quadrant->level);

  auto block = driver->AllocateBlock(lower_bounds, cell_size, quadrant->level);
  quadrant->p.user_data = block;
}

/**
 * @brief Create a new Block for the given quadrant and set its initial data
 */
template <typename DRIVER>
static void CreateBlockWithGridInitialData(
    p8est_t *p8est, p4est_topidx_t which_tree, p8est_quadrant_t *quadrant) {
  CreateBlock<DRIVER>(p8est, which_tree, quadrant);

  double vertex[3];
  p8est_qcoord_to_vertex(
      p8est->connectivity, which_tree, quadrant->x, quadrant->y, quadrant->z, vertex);
  auto driver = static_cast<DRIVER *>(p8est->user_pointer);
  auto block = static_cast<typename DRIVER::Block_t *>(quadrant->p.user_data);

  // set initial data, we need the i, j, k indices of the block, which we get from the vertex.
  // Unfortunately, vertex is a double, so we round to integers, avoiding truncation, which could
  // give the wrong result.

  // we can only set initial data on level 0 quadrants
  my_assert(
      quadrant->level == 0, "Tried to set initial data on a quadrant with a level other than 0");

  auto i = lround(vertex[0]);
  auto j = lround(vertex[1]);
  auto k = lround(vertex[2]);

  size_t block_size = DRIVER::Block_t::N;

  block->SetState(driver->HostCellData(),
      subview(driver->init_data(),
          Kokkos::make_pair(i * block_size, (i + 1) * block_size),
          Kokkos::make_pair(j * block_size, (j + 1) * block_size),
          Kokkos::make_pair(k * block_size, (k + 1) * block_size)));
}

/**
 * @brief Return 1 if the block corresponding to the given quadrant should be refined, return 0
 * otherwise.
 */
template <typename DRIVER>
static int RefineBlock(p8est_t *p8est, p4est_topidx_t /*which_tree*/, p8est_quadrant_t *quadrant) {
  auto driver = static_cast<DRIVER *>(p8est->user_pointer);
  auto block = static_cast<typename DRIVER::Block_t *>(quadrant->p.user_data);
  if (driver->refinement_flag(block) == 1)
    return 1;
  else
    return 0;
}

/**
 * @brief Return 1 if the family of blocks corresponding to the given quadrants should be refined,
 * return 0 otherwise.
 */
template <typename DRIVER>
static int CoarsenBlocks(
    p8est_t *p8est, p4est_topidx_t /*which_tree*/, p8est_quadrant_t *quadrants[]) {
  auto driver = static_cast<DRIVER *>(p8est->user_pointer);
  // only coarsen if each block is flagged for coarsening and it would not violate the 2:1 balance
  for (int i = 0; i < P8EST_CHILDREN; ++i) {
    auto block = static_cast<typename DRIVER::Block_t *>(quadrants[i]->p.user_data);
    if (driver->refinement_flag(block) > -driver->amr_derefinement_count()) return 0;

    // check the neighboring levels of this block. If any neighbor level is higher than this
    // level, coarsening this block would break the 2:1 balance, so we won't allow it
    for (int ni = -1; ni <= 1; ++ni) {
      for (int nj = -1; nj <= 1; ++nj) {
        for (int nk = -1; nk <= 1; ++nk) {
          if (block->neighbor_level(ni, nj, nk) > block->level()) return 0;
        }
      }
    }
  }

  // if we get here, each block is flagged for coarsening
  return 1;
}

template <typename DRIVER>
struct ReplaceTaskArgs {
  using Block_t = typename DRIVER::Block_t;
  bool restrict;

  int coarse_bid;
  Kokkos::Array<int, P8EST_CHILDREN> fine_bids;
};

/**
 * @brief Add ReplaceBlocksTaskArgs to the argument queue to replace the blocks corresponding to
 * the outgoing quadrants with the blocks corresponding ot the incoming quadrants.
 */
template <typename DRIVER>
static void ReplaceBlocks(p8est_t *p8est,
    p4est_topidx_t /*which_tree*/,
    int num_outgoing,
    p8est_quadrant_t *outgoing[],
    int num_incoming,
    p8est_quadrant_t *incoming[]) {
  auto driver = static_cast<DRIVER *>(p8est->user_pointer);
  ReplaceTaskArgs<DRIVER> args;

  if (!((num_incoming == 1) && (num_outgoing == P8EST_CHILDREN)) &&
      !((num_incoming == P8EST_CHILDREN) && (num_outgoing == 1)))
    throw std::runtime_error("Unexpected number of incoming/outgoing blocks in ReplaceBlocks");

  if (num_incoming == 1) {
    args.restrict = true;
    args.coarse_bid = static_cast<typename DRIVER::Block_t *>(incoming[0]->p.user_data)->id();
    for (int i = 0; i < P8EST_CHILDREN; ++i)
      args.fine_bids[i] = static_cast<typename DRIVER::Block_t *>(outgoing[i]->p.user_data)->id();
  } else {
    args.restrict = false;
    args.coarse_bid = static_cast<typename DRIVER::Block_t *>(outgoing[0]->p.user_data)->id();
    for (int i = 0; i < P8EST_CHILDREN; ++i)
      args.fine_bids[i] = static_cast<typename DRIVER::Block_t *>(incoming[i]->p.user_data)->id();
  }

  driver->AddReplaceBlocksTaskArgs(std::move(args));
}

/**
 * @brief Simple struct holding a list of pointers to the blocks and faces. We divide the list of
 * faces into faces between blocks of the same level and between blocks of different levels,
 * because for faces between blocks of different levels we can only perform prolongations after
 * all restrictions and data exchange between same-level blocks have been done.
 */
template <typename DRIVER>
struct BlocksFaces {
  using Block_t = typename DRIVER::Block_t;

  std::vector<Block_t *> blocks;
  typename Block_t::BlockIDs_t block_ids;
  typename Block_t::CellSizes_t cell_sizes;

  Kokkos::View<IntercellFace<Block_t> *, Kokkos::CudaSpace> same_level_faces;
  Kokkos::View<BoundaryFace<Block_t> *, Kokkos::CudaSpace> boundary_faces;
  Kokkos::View<InterLevelFace<Block_t, DRIVER::Slope> *, Kokkos::CudaSpace> inter_level_faces;
};

template <typename DRIVER>
static void GetBlocksAndFaces(p8est_t *p8est, BlocksFaces<DRIVER> *blocks_faces) {
  using Block_t = typename DRIVER::Block_t;

  struct Vecs {
    std::vector<Block_t *> blocks;
    std::vector<IntercellFace<Block_t>> same_level_faces;
    std::vector<BoundaryFace<Block_t>> boundary_faces;
    std::vector<InterLevelFace<Block_t, DRIVER::Slope>> inter_level_faces;
  };

  Vecs vecs;

  p8est_iterate(
      p8est,
      nullptr,
      &vecs,
      [](p8est_iter_volume_info_t *info, void *user_data) {
        auto &blocks = static_cast<Vecs *>(user_data)->blocks;
        blocks.push_back(static_cast<Block_t *>(info->quad->p.user_data));
      },
      [](p8est_iter_face_info_t *info, void *user_data) {
        auto driver = static_cast<DRIVER *>(info->p4est->user_pointer);
        auto &same_level_faces = static_cast<Vecs *>(user_data)->same_level_faces;
        auto &boundary_faces = static_cast<Vecs *>(user_data)->boundary_faces;
        auto &inter_level_faces = static_cast<Vecs *>(user_data)->inter_level_faces;

        if (info->sides.elem_count == 1) {
          // this is an outer boundary where we need to apply a boundary condition
          auto side = p8est_iter_fside_array_index_int(&(info->sides), 0);
          my_assert(side->is_hanging == 0, "Got a hanging face at an outer boundary");

          auto block = static_cast<Block_t *>(side->is.full.quad->p.user_data);

          my_assert((side->face >= 0) && (side->face <= 5),
              "Unknown face type " + std::to_string(side->face));

          size_t direction = side->face / 2;
          Side face_side = (side->face % 2) == 0 ? Side::Lower : Side::Upper;

          BoundaryFace<Block_t> face(direction,
              face_side,
              driver->boundary_conditions()[direction][face_side == Side::Lower ? 0 : 1]);
          face.ConnectBlock(face_side == Side::Lower ? Side::Upper : Side::Lower, block);
          boundary_faces.push_back(face);
        } else if (info->sides.elem_count == 2) {
          // this is a face between blocks
          auto side0 = p8est_iter_fside_array_index_int(&(info->sides), 0);
          auto side1 = p8est_iter_fside_array_index_int(&(info->sides), 1);

          auto face_dir = side0->face / 2;
          my_assert(
              face_dir == (side1->face / 2), "Got face with sides having different directions");
          // these should differ by 1 because one is lower and the other upper side
          my_assert(
              side0->face != side1->face, "Got face with exactly the same face type on both sides");

          // the side of the block is the opposite side of the face (if the face is on the lower
          // side of the block, the block is on the upper side of the face)
          Side block0_side = (side0->face < side1->face) ? Side::Upper : Side::Lower;
          Side block1_side = (side0->face < side1->face) ? Side::Lower : Side::Upper;

          // used only if we have an inter-level face
          p8est_iter_face_side *coarse_side = nullptr;
          p8est_iter_face_side *fine_side = nullptr;
          Side *coarse_side_type = nullptr;
          Side *fine_side_type = nullptr;

          if (side0->is_hanging == 0) {
            // side0 is not hanging
            if (side1->is_hanging == 0) {
              // this is a non-hanging face between two blocks of the same level
              IntercellFace<Block_t> face(face_dir);

              auto block0 = static_cast<Block_t *>(side0->is.full.quad->p.user_data);
              auto block1 = static_cast<Block_t *>(side1->is.full.quad->p.user_data);

              face.ConnectBlock(block0_side, block0);
              face.ConnectBlock(block1_side, block1);
              same_level_faces.push_back(face);
            } else {
              // this is a hanging face with side1 finer than side0
              coarse_side = side0;
              coarse_side_type = &block0_side;
              fine_side = side1;
              fine_side_type = &block1_side;
            }
          } else {
            // side0 is hanging
            my_assert(side1->is_hanging == 0, "Got a face with both sides hanging");

            // this is a hanging face with side0 finer than side1
            coarse_side = side1;
            coarse_side_type = &block1_side;
            fine_side = side0;
            fine_side_type = &block0_side;
          }

          // create face between blocks of different levels
          if (coarse_side != nullptr) {
            auto coarse_block = static_cast<Block_t *>(coarse_side->is.full.quad->p.user_data);

            InterLevelFace<Block_t, DRIVER::Slope> face(face_dir, *coarse_side_type);
            face.ConnectBlock(*coarse_side_type, coarse_block);
            for (int i = 0; i < 4; ++i) {
              auto fine_block = static_cast<Block_t *>(fine_side->is.hanging.quad[i]->p.user_data);
              face.ConnectFineBlock(*fine_side_type, fine_block, coarse_block);
            }
            inter_level_faces.push_back(face);
          }
        } else {
          throw std::runtime_error(
              "Got a face with " + std::to_string(info->sides.elem_count) + " sides");
        }
      },
      nullptr,
      [](p8est_iter_corner_info_t *info, void * /*user_data*/) {
        // this callback is called for all corners that are a corner for each quadrant they
        // touch, i.e. this callback is NOT called for coners on hanging edges or faces

        // for interior corners there are 8 sides, for corners sitting on an outer boundary face
        // there are 4, for corners sitting on outer boundary edge there are 2, and for the
        // outer boundary corners there is 1 side
        auto num_sides = info->sides.elem_count;
        my_assert((num_sides == 1) || (num_sides == 2) || (num_sides == 4) || (num_sides == 8),
            "Expected 1, 2, 4, or 8 sides for a corner");

        for (size_t i = 0; i < num_sides; ++i) {
          auto this_side = p8est_iter_cside_array_index(&(info->sides), i);
          auto this_level = this_side->quad->level;
          auto this_block = static_cast<Block_t *>(this_side->quad->p.user_data);

          // corner is a number between 0 and 7, indicating which corner of the quadrant this
          // corner is. Let the corner index (0,1) in the x-direction be ci, in the y-direction
          // cj, and in the z-direction ck. Then corner = 4 * ck + 2 * cj + ci
          my_assert(
              (this_side->corner >= 0) && (this_side->corner <= 7), "unexpected corner number");
          std::bitset<3> this_corner_bits(this_side->corner);
          Array<int, 3> this_corner_idxs{
              this_corner_bits[0], this_corner_bits[1], this_corner_bits[2]};

          for (size_t j = i + 1; j < num_sides; ++j) {
            // loop over all pairs of blocks connected by this corner
            auto other_side = p8est_iter_cside_array_index(&(info->sides), j);
            auto other_level = other_side->quad->level;
            auto other_block = static_cast<Block_t *>(other_side->quad->p.user_data);
            my_assert(
                (other_side->corner >= 0) && (other_side->corner <= 7), "unexpected corner number");
            std::bitset<3> other_corner_bits(other_side->corner);
            Array<int, 3> other_corner_idxs{
                other_corner_bits[0], other_corner_bits[1], other_corner_bits[2]};

            // the neighbor indices of other_block relative to this_block are simply
            // other_corner_idxs - this_corner_idxs and vice versa
            this_block->neighbor_level(other_corner_idxs[0] - this_corner_idxs[0],
                other_corner_idxs[1] - this_corner_idxs[1],
                other_corner_idxs[2] - this_corner_idxs[2]) = other_level;
            other_block->neighbor_level(this_corner_idxs[0] - other_corner_idxs[0],
                this_corner_idxs[1] - other_corner_idxs[1],
                this_corner_idxs[2] - other_corner_idxs[2]) = this_level;
          }
        }
      });

  blocks_faces->blocks = vecs.blocks;

  Kokkos::resize(blocks_faces->block_ids, vecs.blocks.size());
  Kokkos::resize(blocks_faces->cell_sizes, vecs.blocks.size());
  Kokkos::resize(blocks_faces->same_level_faces, vecs.same_level_faces.size());
  Kokkos::resize(blocks_faces->boundary_faces, vecs.boundary_faces.size());
  Kokkos::resize(blocks_faces->inter_level_faces, vecs.inter_level_faces.size());

  auto host_bids = Kokkos::create_mirror_view(blocks_faces->block_ids);
  auto host_csizes = Kokkos::create_mirror_view(blocks_faces->cell_sizes);
  auto host_same_faces = Kokkos::create_mirror_view(blocks_faces->same_level_faces);
  auto host_bdry_faces = Kokkos::create_mirror_view(blocks_faces->boundary_faces);
  auto host_inter_faces = Kokkos::create_mirror_view(blocks_faces->inter_level_faces);

  for (size_t i = 0; i < vecs.blocks.size(); ++i) {
    host_bids(i) = vecs.blocks[i]->id();
    host_csizes(i, 0) = vecs.blocks[i]->cell_size()[0];
    host_csizes(i, 1) = vecs.blocks[i]->cell_size()[1];
    host_csizes(i, 2) = vecs.blocks[i]->cell_size()[2];
  }

  for (size_t i = 0; i < vecs.same_level_faces.size(); ++i)
    host_same_faces(i) = vecs.same_level_faces[i];

  for (size_t i = 0; i < vecs.boundary_faces.size(); ++i)
    host_bdry_faces(i) = vecs.boundary_faces[i];

  for (size_t i = 0; i < vecs.inter_level_faces.size(); ++i)
    host_inter_faces(i) = vecs.inter_level_faces[i];

  Kokkos::deep_copy(blocks_faces->block_ids, host_bids);
  Kokkos::deep_copy(blocks_faces->cell_sizes, host_csizes);
  Kokkos::deep_copy(blocks_faces->same_level_faces, host_same_faces);
  Kokkos::deep_copy(blocks_faces->boundary_faces, host_bdry_faces);
  Kokkos::deep_copy(blocks_faces->inter_level_faces, host_inter_faces);
}

} // namespace AMR

#endif // ETHON_KOKKOS_CUDA_AMR_HPP_
