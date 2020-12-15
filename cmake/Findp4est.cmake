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

include(FindPackageHandleStandardArgs)

find_library(p4est_LIBRARY p4est)
find_library(sc_LIBRARY sc)
find_path(p4est_INCLUDE_DIR p4est.h)

find_package_handle_standard_args(
    p4est
    FOUND_VAR p4est_FOUND
    REQUIRED_VARS
        p4est_LIBRARY
        sc_LIBRARY
        p4est_INCLUDE_DIR)

if (p4est_FOUND)
    add_library(sc UNKNOWN IMPORTED)
    set_target_properties(
      sc
      PROPERTIES
        IMPORTED_LOCATION "${sc_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${p4est_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "sc"
    )

    add_library(p4est UNKNOWN IMPORTED)
    set_target_properties(
      p4est
      PROPERTIES
        IMPORTED_LOCATION "${p4est_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${p4est_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "p4est"
    )

    target_link_libraries(p4est INTERFACE sc)
endif()
