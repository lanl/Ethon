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

#ifndef ETHON_KOKKOS_CUDA_AMR_DRIVER_HPP_
#define ETHON_KOKKOS_CUDA_AMR_DRIVER_HPP_

#include <chrono>
#include <memory>
#include <vector>

#include "kokkos_cuda/amr.hpp"
#include "kokkos_cuda/amr_mesh_info.hpp"
#include "kokkos_cuda/block.hpp"
#include "kokkos_cuda/block_pool.hpp"
#include "kokkos_cuda/face.hpp"
#include "kokkos_cuda/output.hpp"

template <typename RIEMANN, SlopeType SLOPE, size_t BLOCK_SIZE, RefineCond COND, typename ExecSpace>
class AMRDriver {
public:
  using State = State_u<3>;
  using StateData = Kokkos::View<State ***, LayoutStride, Kokkos::HostSpace>;
  using Mesh = AMRMeshInfo<BLOCK_SIZE>;
  using output_func_t = std::function<void(
      const size_t, const double, const double, const Mesh &, const AMRDriver &)>;
  using EOS = typename RIEMANN::EOS;

  using Block_t = Block<EOS, SLOPE, BLOCK_SIZE, ExecSpace>;
  // using Face_t = Face<Block_t>;

  using Clock = std::chrono::high_resolution_clock;

  static output_func_t no_output() {
    return
        [](const size_t, const double, const double, const Mesh &, const AMRDriver &) { return; };
  }

  static constexpr SlopeType Slope = SLOPE;

  const auto &blocks() const { return blocks_faces_.blocks; }
  auto &CellData() { return block_pool_.cell_data; }
  auto &HostCellData() { return block_pool_.host_cell_data; }

  // these properties need to be available to the static AMR functions via a pointer to this class
  // instance, so we need accessors for them
  auto mesh() const { return mesh_; }
  auto init_data() const { return init_data_; }
  auto boundary_conditions() const { return boundary_conditions_; }
  auto amr_derefinement_count() const { return amr_derefinement_count_; }

  auto refinement_flag(const Block_t *block) const {
    return block_pool_.host_refinement_flags(block->id());
  }

  Block_t *AllocateBlock(const Kokkos::Array<double, 3> lower_bounds,
      const Kokkos::Array<double, 3> cell_size,
      const int level) {
    return block_pool_.Allocate(lower_bounds, cell_size, level);
  }

  /*************************************************************************************************
   * TIMING FUNCTIONS
   ************************************************************************************************/

  static auto Elapsed(const Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(Clock::now() - start).count();
  };

  void PrintTimers() const {
    double time_total = Elapsed(beginning_);
    double time_other = time_total - (time_setup_ + time_ghost_exchange_ + time_hydro_ +
                                         time_hook_ + time_amr_total_ + time_p4est_iter_);
    double time_amr_other =
        time_amr_total_ - (time_amr_refine_ + time_amr_coarsen_ + time_amr_balance_);

    printf("\n");
    printf("Setup & tear-down: %8.3f s  %6.2f%%\n", time_setup_, 100.0 * time_setup_ / time_total);
    printf("   Ghost exchange: %8.3f s  %6.2f%%\n",
        time_ghost_exchange_,
        100.0 * time_ghost_exchange_ / time_total);
    printf("            Hydro: %8.3f s  %6.2f%%\n", time_hydro_, 100.0 * time_hydro_ / time_total);
    printf("          Hydro A: %8.3f s  %6.2f%%\n", time_hydroA, 100.0 * time_hydroA / time_hydro_);
    printf("          Hydro B: %8.3f s  %6.2f%%\n", time_hydroB, 100.0 * time_hydroB / time_hydro_);
    printf("          Hydro C: %8.3f s  %6.2f%%\n", time_hydroC, 100.0 * time_hydroC / time_hydro_);
    printf("             Hook: %8.3f s  %6.2f%%\n", time_hook_, 100.0 * time_hook_ / time_total);
    printf("              AMR: %8.3f s  %6.2f%%\n",
        time_amr_total_,
        100.0 * time_amr_total_ / time_total);
    printf("    p4est iterate: %8.3f s  %6.2f%%\n",
        time_p4est_iter_,
        100.0 * time_p4est_iter_ / time_total);
    printf("            Other: %8.3f s  %6.2f%%\n", time_other, 100.0 * time_other / time_total);
    printf("            Total: %8.3f s  %6.2f%%\n\n", time_total, 100.0 * time_total / time_total);

    printf("       AMR Refine: %8.3f s  %6.2f%%\n",
        time_amr_refine_,
        100.0 * time_amr_refine_ / time_amr_total_);
    printf("      AMR Coarsen: %8.3f s  %6.2f%%\n",
        time_amr_coarsen_,
        100.0 * time_amr_coarsen_ / time_amr_total_);
    printf("      AMR Balance: %8.3f s  %6.2f%%\n",
        time_amr_balance_,
        100.0 * time_amr_balance_ / time_amr_total_);
    printf("        AMR Other: %8.3f s  %6.2f%%\n",
        time_amr_other,
        100.0 * time_amr_other / time_amr_total_);
    printf("        Total AMR: %8.3f s  %6.2f%%\n\n",
        time_amr_total_,
        100.0 * time_amr_total_ / time_amr_total_);
    printf("      Zone cycles: %lu\n", zone_cycles_ * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE);
    printf("Zone cycles / sec: %.6e\n\n",
        double(zone_cycles_ * BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE) / time_total);
  }

  /*************************************************************************************************
   * AMR FUNCTIONS
   ************************************************************************************************/

  void AddReplaceBlocksTaskArgs(AMR::ReplaceTaskArgs<AMRDriver> &&args) {
    replace_blocks_task_args_.emplace_back(args);
  }

  /**
   * @brief Replace a block to be refined with its refined children, or a family of blocks to be
   * coarsened with their coarser parent.
   */
  void DoReplaceBlocksTask(const typename Block_t::CellData_t &cell_data,
      const std::vector<AMR::ReplaceTaskArgs<AMRDriver>> &args) {
    if (args.size() > 0)
      mesh_changed_ = true;
    else
      return;

    Kokkos::View<AMR::ReplaceTaskArgs<AMRDriver> *, Kokkos::CudaSpace> args_view(
        "Replace args", args.size());
    auto host_args = Kokkos::create_mirror_view(args_view);

#pragma omp parallel for
    for (size_t i = 0; i < args.size(); ++i)
      host_args(i) = args[i];

    Kokkos::deep_copy(args_view, host_args);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<ExecSpace>(args_view.extent(0), Kokkos::AUTO()),
        KOKKOS_LAMBDA(typename Block_t::team_member_t team_member) {
          const auto &arg = args_view(team_member.league_rank());
          for (int c = 0; c < P8EST_CHILDREN; ++c) {
            // The children should be in Morton ordering (i.e. x moves faster than y, which moves
            // faster than z, so the order for xyz is 000, 100, 010, 110, 001, 101, 011, 111). Get
            // child indices (0 or 1) in x, y, and z directions
            int ci = (c & (1 << 0)) == 0 ? 0 : 1;
            int cj = (c & (1 << 1)) == 0 ? 0 : 1;
            int ck = (c & (1 << 2)) == 0 ? 0 : 1;

            if (arg.restrict)
              Block_t::template RestrictProlongateField<true>(
                  team_member, cell_data, arg.coarse_bid, arg.fine_bids[c], ci, cj, ck);
            else
              Block_t::template RestrictProlongateField<false>(
                  team_member, cell_data, arg.coarse_bid, arg.fine_bids[c], ci, cj, ck);
          }

          // if (arg.restrict) {
          //   // this is a coarsening of 8 blocks into 1
          //   for (int c = 0; c < P8EST_CHILDREN; ++c) {

          //     // restrict
          //     Block_t::RestrictFieldOntoParent(
          //         team_member, cell_data, arg.coarse_bid, arg.fine_bids[c], ci, cj, ck);
          //   }
          // } else {
          //   // this is a refinement, one block gets split into 8
          //   for (int c = 0; c < P8EST_CHILDREN; ++c) {
          //     // The children should be in Morton ordering (i.e. x moves faster than y, which moves
          //     // faster than z, so the order for xyz is 000, 100, 010, 110, 001, 101, 011, 111). Get
          //     // child indices (0 or 1) in x, y, and z directions
          //     int ci = (c & (1 << 0)) == 0 ? 0 : 1;
          //     int cj = (c & (1 << 1)) == 0 ? 0 : 1;
          //     int ck = (c & (1 << 2)) == 0 ? 0 : 1;

          //     // prolongate interior data and ghost zones onto the child, the child will have all
          //     // ghost zones filled
          //     Block_t::ProlongateFieldOntoChild(
          //         team_member, cell_data, arg.fine_bids[c], arg.coarse_bid, ci, cj, ck);
          //   }
          // }
        });

    // free replaced blocks on the host
#pragma omp parallel for
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i].restrict) {
        for (int c = 0; c < P8EST_CHILDREN; ++c) {
          block_pool_.Free(args[i].fine_bids[c]);
        }
      } else {
        block_pool_.Free(args[i].coarse_bid);
      }
    }
  }

  /**
   * @brief If the mesh has changed since the last time this function was called, iterate over the
   * p4est tree and make a list of all blocks and faces.
   */
  void UpdateBlocksFaces() {
    auto start = Clock::now();

    if (mesh_changed_) {
      AMR::GetBlocksAndFaces<AMRDriver>(p8est_, &blocks_faces_);

      printf("%lu blocks, %lu same-level faces, %lu boundary faces, %lu inter-level faces\n",
          blocks_faces_.block_ids.extent(0),
          blocks_faces_.same_level_faces.extent(0),
          blocks_faces_.boundary_faces.extent(0),
          blocks_faces_.inter_level_faces.extent(0));
    }

    time_p4est_iter_ += Elapsed(start);
  }

  void DoAMR(
      const double time, const double amr_threshold, const EOS &eos, const int amr_max_level) {
    auto start_amr = Clock::now();

    // evaluate refinement condition
    Block_t::template EvaluateRefinement<COND>(block_pool_.cell_data,
        blocks_faces_.block_ids,
        blocks_faces_.cell_sizes,
        time,
        amr_threshold,
        eos,
        block_pool_.refinement_flags);

    // copy refinement flags to device
    Kokkos::deep_copy(block_pool_.host_refinement_flags, block_pool_.refinement_flags);

    // reset replace blocks task args
    replace_blocks_task_args_.clear();
    mesh_changed_ = false;

    // refine and coarsen blocks based on flags
    {
      auto start = Clock::now();
      p8est_refine_ext(p8est_,
          0,
          amr_max_level,
          AMR::RefineBlock<AMRDriver>,
          AMR::CreateBlock<AMRDriver>,
          AMR::ReplaceBlocks<AMRDriver>);
      time_amr_refine_ += Elapsed(start);
    }

    {
      auto start = Clock::now();
      p8est_coarsen_ext(p8est_,
          0,
          0,
          AMR::CoarsenBlocks<AMRDriver>,
          AMR::CreateBlock<AMRDriver>,
          AMR::ReplaceBlocks<AMRDriver>);
      time_amr_coarsen_ += Elapsed(start);
    }

    if (replace_blocks_task_args_.size() == 0) {
      // the mesh didn't change, so we don't need to run the balance function because the mesh is
      // already balanced
      time_amr_total_ += Elapsed(start_amr);
      return;
    }

    // save the refine and coarsen task list, so we can execute them in parallel later
    auto refine_coarsen_task_list = replace_blocks_task_args_;
    replace_blocks_task_args_.clear();

    // enforce 2:1 balance
    {
      auto start = Clock::now();
      p8est_balance_ext(
          p8est_, P8EST_CONNECT_FULL, AMR::CreateBlock<AMRDriver>, AMR::ReplaceBlocks<AMRDriver>);
      time_amr_balance_ += Elapsed(start);
    }

    // save the balance task list
    auto balance_task_list = replace_blocks_task_args_;

    // Execute replace blocks tasks in parallel. We have to do the refine_coarsen_task_list first,
    // because the balance_task_list may refer to newly created blocks in refine_coarsen_task_list.
    DoReplaceBlocksTask(block_pool_.cell_data, refine_coarsen_task_list);
    DoReplaceBlocksTask(block_pool_.cell_data, balance_task_list);

    time_amr_total_ += Elapsed(start_amr);

    UpdateBlocksFaces();

    // copy init refinement flags of new blocks to device
    Kokkos::deep_copy(block_pool_.refinement_flags, block_pool_.host_refinement_flags);
  }

  /*************************************************************************************************
   * HYDRO FUNCTIONS
   ************************************************************************************************/

  void DoGhostExchange() {
    auto start = Clock::now();

    // first do same-level exchange (includes physical boundaries)
    IntercellFace<Block_t>::ExchangeGhostData(
        block_pool_.cell_data, blocks_faces_.same_level_faces);
    BoundaryFace<Block_t>::ExchangeGhostData(block_pool_.cell_data, blocks_faces_.boundary_faces);

    // now do restrictions (fine to coarse, this can be done in parallel with the above)
    InterLevelFace<Block_t, SLOPE>::RestrictGhostData(
        block_pool_.cell_data, blocks_faces_.inter_level_faces);

    // finally do prolongation (this HAS to happen AFTER same-level exchange and restriction)
    InterLevelFace<Block_t, SLOPE>::ProlongateGhostData(
        block_pool_.cell_data, blocks_faces_.inter_level_faces);

    Kokkos::Cuda().fence();
    time_ghost_exchange_ += Elapsed(start);
  }

  // Initial data can be specified either as grid data on the initial grid (without AMR), or a
  // generator that is applied to each block, including newly refined blocks
  struct InitialData {
    virtual bool is_grid_data() const = 0;
  };

  struct GridInitialData : public InitialData {
    bool is_grid_data() const final override { return true; }

    StateData init_data;
  };

  struct GeneratorInitialData : public InitialData {
    bool is_grid_data() const final override { return false; }

    std::function<void(Block_t *block)> generator;
  };

  void ApplyInitialDataGenerator(const GeneratorInitialData *init) {
    auto gen = init->generator;
    for (auto &b : blocks_faces_.blocks) {
      gen(b);
    }
  }

  void Evolve(const Mesh &mesh,
      BoundaryConditionType boundary_conditions[3][2],
      const EOS &eos,
      const double cfl,
      const int amr_max_level,
      const int amr_derefinement_count,
      const double amr_threshold,
      const InitialData *initial_data,
      const double t_start,
      const double t_end,
      const double max_dt,
      output_func_t hook = no_output()) {
    beginning_ = Clock::now();

    // copy some parameters
    mesh_ = std::make_shared<const Mesh>(mesh);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 2; ++j) {
        boundary_conditions_[i][j] = boundary_conditions[i][j];
      }
    }
    amr_derefinement_count_ = amr_derefinement_count;

    const auto N_blocks = mesh_->num_blocks();
    RIEMANN riemann(eos);

    // flag indicating periodic boundary conditions in each direction
    auto check_periodic = [&](const size_t dir) {
      if (boundary_conditions_[dir][0] == BoundaryConditionType::Periodic) {
        assert(boundary_conditions_[dir][1] == BoundaryConditionType::Periodic);
        return true;
      } else {
        assert(boundary_conditions_[dir][1] != BoundaryConditionType::Periodic);
        return false;
      }
    };

    bool x_periodic = check_periodic(0);
    bool y_periodic = check_periodic(1);
    bool z_periodic = check_periodic(2);

    auto mpi_world = sc_MPI_COMM_WORLD;

    auto conn = p8est_connectivity_new_brick(
        N_blocks[0], N_blocks[1], N_blocks[2], x_periodic, y_periodic, z_periodic);

    if (initial_data->is_grid_data()) {
      init_data_ = static_cast<const GridInitialData *>(initial_data)->init_data;
      p8est_ = p8est_new(mpi_world, conn, 0, AMR::CreateBlockWithGridInitialData<AMRDriver>, this);

      // copy initial data to device
      Kokkos::deep_copy(block_pool_.cell_data, block_pool_.host_cell_data);
    } else {
      p8est_ = p8est_new(mpi_world, conn, 0, AMR::CreateBlock<AMRDriver>, this);
    }

    time_setup_ += Elapsed(beginning_);

    // construct lists of blocks and faces and exchange ghost data (we need ghost data to evaluate
    // the refinement condition)
    mesh_changed_ = true;
    UpdateBlocksFaces();

    auto start = Clock::now();
    if (!initial_data->is_grid_data()) {
      ApplyInitialDataGenerator(static_cast<const GeneratorInitialData *>(initial_data));

      // copy initial data to device
      Kokkos::deep_copy(block_pool_.cell_data, block_pool_.host_cell_data);
    }

    time_setup_ += Elapsed(start);

    DoGhostExchange();

    // do an initial recursive AMR until mesh is not chaning anymore to satisfy the refinement
    // condition with the initial data
    while (mesh_changed_) {
      DoAMR(t_start, amr_threshold, eos, amr_max_level);

      if (mesh_changed_) {
        if (!initial_data->is_grid_data()) {
          ApplyInitialDataGenerator(static_cast<const GeneratorInitialData *>(initial_data));

          // copy initial data to device
          Kokkos::deep_copy(block_pool_.cell_data, block_pool_.host_cell_data);
        }

        DoGhostExchange();
      }
    }

    start = Clock::now();

    // compute first time step size
    double dt = Block_t::MinDt(
        block_pool_.cell_data, blocks_faces_.block_ids, blocks_faces_.cell_sizes, eos);

    // make first time step smaller because we have a discontinuity but fluid velocity is still 0
    dt *= cfl * 0.2;
    dt = std::min(dt, t_end - t_start);
    dt = std::min(dt, max_dt);

    // main evolution loop
    double t = t_start; // time in s
    size_t num_steps = 0;
    time_setup_ += Elapsed(start);

    DoGhostExchange();
    start = Clock::now();
    hook(num_steps, t, dt, *mesh_, *this);
    time_hook_ += Elapsed(start);

    while (t < t_end) {
      auto start_hydro = Clock::now();

      Block_t::TakeStepA(block_pool_.cell_data,
          block_pool_.lower_boundary,
          block_pool_.upper_boundary,
          blocks_faces_.block_ids,
          blocks_faces_.cell_sizes,
          riemann,
          eos,
          dt);

      time_hydroA += Elapsed(start_hydro);
      start = Clock::now();

      Block_t::TakeStepB(block_pool_.lower_boundary,
          block_pool_.upper_boundary,
          block_pool_.x_fluxes,
          block_pool_.y_fluxes,
          block_pool_.z_fluxes,
          blocks_faces_.block_ids,
          riemann,
          eos,
          dt);

      time_hydroB += Elapsed(start);
      start = Clock::now();

      // take time step in each block
      double new_dt = Block_t::TakeStepC(block_pool_.x_fluxes,
          block_pool_.y_fluxes,
          block_pool_.z_fluxes,
          block_pool_.cell_data,
          blocks_faces_.block_ids,
          blocks_faces_.cell_sizes,
          riemann,
          eos,
          dt);

      time_hydroC += Elapsed(start);

      t += dt;
      ++num_steps;
      zone_cycles_ += blocks_faces_.blocks.size();

      dt = new_dt * cfl;
      if (num_steps <= 5) dt *= 0.2;
      dt = std::min(dt, t_end - t);
      dt = std::min(dt, max_dt);

      time_hydro_ += Elapsed(start_hydro);

      // exchange data between blocks
      DoGhostExchange();

      start = Clock::now();
      hook(num_steps, t, dt, *mesh_, *this);
      time_hook_ += Elapsed(start);

      printf("[%06lu] t = %.10e, dt = %.10e\n", num_steps, t, dt);

      // if we're done, we don't need to do AMR
      if (t >= t_end) break;

      // Do AMR
      DoAMR(t, amr_threshold, eos, amr_max_level);
      if (mesh_changed_) DoGhostExchange();

      if (num_steps % 100 == 0) PrintTimers();
    }

    // clean up
    start = Clock::now();

    p8est_destroy(p8est_);
    p8est_connectivity_destroy(conn);
    // this fails, need to look into it at some point
    // sc_finalize();
    // SC_CHECK_MPI(sc_MPI_Finalize());

    time_setup_ += Elapsed(start);
    PrintTimers();
  }

  template <typename EOS>
  void DumpGrid(const std::string &fname,
      const size_t num_steps,
      const double time,
      const Mesh &mesh,
      const EOS &eos,
      const bool z_slice = false) const {
    // copy data to Host for output
    Kokkos::deep_copy(block_pool_.host_cell_data, block_pool_.cell_data);
    output::DumpGrid(fname,
        num_steps,
        time,
        mesh,
        eos,
        block_pool_.host_cell_data,
        blocks_faces_.blocks,
        z_slice);
  }

private:
  // these properties need to be available to the static AMR functions via a pointer to this class
  // instance, so we need to save them
  std::shared_ptr<const Mesh> mesh_;
  StateData init_data_;
  BoundaryConditionType boundary_conditions_[3][2];
  int amr_derefinement_count_;

  // p8est handle
  p8est_t *p8est_ = nullptr;

  // memory pool for blocks
  BlockPool<Block_t> block_pool_;

  // lists of blocks and faces
  AMR::BlocksFaces<AMRDriver> blocks_faces_;

  // collect arguments for ReplaceBlocksTask
  std::vector<AMR::ReplaceTaskArgs<AMRDriver>> replace_blocks_task_args_;

  // flag indicating that AMR changed the mesh, which means we need to do ghost exchange again
  bool mesh_changed_ = true;

  // timers
  Clock::time_point beginning_;
  double time_setup_ = 0.0;
  double time_ghost_exchange_ = 0.0;
  double time_hydro_ = 0.0;
  double time_hook_ = 0.0;
  double time_amr_total_ = 0.0;
  double time_amr_refine_ = 0.0;
  double time_amr_coarsen_ = 0.0;
  double time_amr_balance_ = 0.0;
  double time_p4est_iter_ = 0.0;
  size_t zone_cycles_ = 0;

  double time_hydroA = 0.0;
  double time_hydroB = 0.0;
  double time_hydroC = 0.0;
};

#endif // ETHON_KOKKOS_CUDA_AMR_DRIVER_HPP_
