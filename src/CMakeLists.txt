#========================================================================================
# Copyright (c) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#
# This program is open source under the BSD-3 License. See LICENSE file for details.
#========================================================================================

add_library(ethon INTERFACE)

target_include_directories(ethon INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(ethon INTERFACE
  Kokkos::kokkos
  OpenMP::OpenMP_CXX
  p4est
  ${HDF5_CXX_LIBRARIES}
)
