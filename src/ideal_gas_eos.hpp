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

#ifndef ETHON_IDEAL_GAS_EOS_HPP_
#define ETHON_IDEAL_GAS_EOS_HPP_

#include <Kokkos_Core.hpp>

template <typename State>
class IdealGas {
public:
  IdealGas(const double gamma) : gamma_(gamma) {}

  KOKKOS_FORCEINLINE_FUNCTION double gamma() const { return gamma_; }

  KOKKOS_FORCEINLINE_FUNCTION double sound_speed(const State &U) const {
    // if (U.specific_internal_energy() < 0.0)
    //   printf("NEGATIVE SPECIFIC INTERNAL ENERGY: %.10e\n", U.specific_internal_energy());
    return sqrt(gamma_ * (gamma_ - 1.0) * U.specific_internal_energy());
  }

  KOKKOS_FORCEINLINE_FUNCTION double pressure(const State &U) const {
    return (gamma_ - 1.0) * U.rho * U.specific_internal_energy();
  }

private:
  double gamma_; // adiabatic index of the equation of state
};

#endif // ETHON_IDEAL_GAS_EOS_HPP_
