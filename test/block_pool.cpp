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

#include <random>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "block_mesh/block.hpp"
#include "block_mesh/block_pool.hpp"

#include "ideal_gas_eos.hpp"
#include "slopes.hpp"

using State = State_u<3>;
using EOS = IdealGas<State>;
using Block_t = Block<EOS, SlopeType::MC, 8>;

TEST(Block_Pool, Create) {
  printf("size of block: %lu\n", sizeof(Block_t));

  for (size_t size = 1; size <= 100000; size *= 10) {
    printf("Allocating %lu\n", size);
    BlockPool<Block_t> pool(512);

    std::vector<Block_t *> blocks;
    for (size_t i = 0; i < size; ++i) {
      blocks.push_back(pool.Allocate({double(i), double(i), double(i)}, {1, 1, 1}, i));
    }

    EXPECT_EQ(pool.size(), size);

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, size - 1);
    // do some random tests
    for (size_t i = 0; i < size / 10; ++i) {
      int idx = distribution(generator);
      EXPECT_EQ(blocks[idx]->level(), idx);
    }

    for (size_t i = 0; i < size; ++i) {
      pool.Free(blocks[i]);
    }

    EXPECT_EQ(pool.size(), 0);
  }
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();

  Kokkos::finalize();
  return ret;
}
