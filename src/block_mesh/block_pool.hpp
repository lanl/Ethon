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

#ifndef ETHON_BLOCK_MESH_BLOCK_POOL_HPP_
#define ETHON_BLOCK_MESH_BLOCK_POOL_HPP_

#include <mutex>
#include <stack>
#include <vector>

#include <Kokkos_Core.hpp>

template <typename Block_t>
class BlockPool {
public:
  BlockPool(const size_t chunk_size) : chunk_size_(chunk_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    AddChunk();
  }

  ~BlockPool() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto chunk : chunks_)
      if (chunk != nullptr) free(chunk);
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return chunk_size_ * chunks_.size() - free_blocks_.size();
  }

  Block_t *Allocate(const Kokkos::Array<double, 3> lower_bounds,
      const Kokkos::Array<double, 3> cell_size,
      const int level) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_blocks_.size() == 0) AddChunk();

    auto block = free_blocks_.top();
    free_blocks_.pop();

    block->Init(lower_bounds, cell_size, level);
    return block;
  }

  void Free(Block_t *block) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.push(block);
  }

private:
  // needs to be called inside a locked scope
  void AddChunk() {
    Block_t *chunk = (Block_t *)malloc(chunk_size_ * sizeof(Block_t));
    chunks_.push_back(chunk);

    // push from back to front, so that we'll use the chunk from beginning to end (not that it
    // matters...)
    for (size_t i = 0; i < chunk_size_; ++i)
      free_blocks_.push(chunk + chunk_size_ - 1 - i);
  }

  size_t chunk_size_;

  std::vector<Block_t *> chunks_;

  // stack of pointers to free blocks
  std::stack<Block_t *> free_blocks_;

  // mutex to serialize access to the pool
  mutable std::mutex mutex_;
};

#endif // ETHON_BLOCK_MESH_BLOCK_POOL_HPP_
