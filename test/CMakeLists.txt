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

include(GoogleTest)

add_library(test_utils
  convergence.cpp
  test_utils.cpp
)
target_link_libraries(test_utils PUBLIC ethon)

# copy 1d/toro_files
add_custom_target(toro_1d_input ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/1d/toro_files
    ${CMAKE_CURRENT_BINARY_DIR}/toro_1d_files)

# copy 1d/sedov_files
add_custom_target(sedov_1d_input ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/1d/sedov_files
    ${CMAKE_CURRENT_BINARY_DIR}/sedov_1d_files)

# copy 3d/sod_files
add_custom_target(sod_3d_input ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/3d/sod_files
    ${CMAKE_CURRENT_BINARY_DIR}/sod_3d_files)

# copy 3d/sod_files
add_custom_target(sedov_3d_input ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/3d/sedov_files
    ${CMAKE_CURRENT_BINARY_DIR}/sedov_3d_files)

# copy 3d/kelvin-helmholtz_files
add_custom_target(kelvin-helmholtz_3d_input ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/3d/kelvin-helmholtz_files
    ${CMAKE_CURRENT_BINARY_DIR}/kelvin-helmholtz_3d_files)

# copy 3d/amr_files
add_custom_target(amr_input ALL
COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/3d/amr_files
  ${CMAKE_CURRENT_BINARY_DIR}/amr_files)

add_executable(toro_1d 1d/toro_1d.cpp)
target_link_libraries(toro_1d GTest::GTest test_utils)
gtest_discover_tests(toro_1d)

add_executable(sedov_1d 1d/sedov_1d.cpp)
target_link_libraries(sedov_1d GTest::GTest test_utils)
gtest_discover_tests(sedov_1d)

add_executable(sod_3d 3d/sod_3d.cpp)
target_link_libraries(sod_3d GTest::GTest ethon test_utils)
gtest_discover_tests(sod_3d)

add_executable(sedov_3d 3d/sedov_3d.cpp)
target_link_libraries(sedov_3d GTest::GTest ethon)
gtest_discover_tests(sedov_3d)

add_executable(kelvin_helmholtz 3d/kelvin-helmholtz.cpp)
target_link_libraries(kelvin_helmholtz GTest::GTest ethon)
gtest_discover_tests(kelvin_helmholtz)

add_executable(amr 3d/amr.cpp)
target_link_libraries(amr GTest::GTest ethon)
gtest_discover_tests(amr)

add_executable(block_pool block_pool.cpp)
target_link_libraries(block_pool GTest::GTest ethon)
gtest_discover_tests(block_pool)

add_executable(kokkos_omp_tests 3d/kokkos_omp.cpp)
target_link_libraries(kokkos_omp_tests GTest::GTest ethon)
gtest_discover_tests(kokkos_omp_tests)

add_executable(kokkos_cuda_tests 3d/kokkos_cuda.cpp)
target_link_libraries(kokkos_cuda_tests GTest::GTest ethon)
gtest_discover_tests(kokkos_cuda_tests)

set_tests_properties(
  ${sod_3d_TESTS}
  ${sedov_3d_TESTS}
  ${kelvin_helmholtz_TESTS}
  ${amr_TESTS}
  ${kokkos_omp_tests_TESTS}
  ${kokkos_cuda_tests_TESTS}
  ${block_pool_TESTS}
  PROPERTIES ENVIRONMENT OMP_PROC_BIND=false
)
