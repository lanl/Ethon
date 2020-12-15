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

#ifndef ETHON_SLOPES_HPP_
#define ETHON_SLOPES_HPP_

#include <cmath>
#include <limits>

#include <Kokkos_Core.hpp>


namespace {

KOKKOS_FORCEINLINE_FUNCTION double minmod(double a, double b) {
  if ((a * b) <= 0.0) return 0.0;
  if (fabs(a) <= fabs(b)) return a;
  return b;
}

KOKKOS_FORCEINLINE_FUNCTION double maxmod(double a, double b) {
  if ((a * b) <= 0.0) return 0.0;
  if (fabs(a) >= fabs(b)) return a;
  return b;
}

} // namespace

enum class SlopeType { MC, SuperBee };

template <SlopeType TYPE>
KOKKOS_FORCEINLINE_FUNCTION double ComputeSlope(
    const double upwind, const double central, const double downwind);

template <>
KOKKOS_FORCEINLINE_FUNCTION double ComputeSlope<SlopeType::MC>(
    const double upwind, const double central, const double downwind) {
  if (upwind * downwind <= 0.0)
    return 0.0;
  else
    return copysign(fmin(fmin(2.0 * fabs(upwind), 2.0 * fabs(downwind)), fabs(central)), central);
}

template <>
KOKKOS_FORCEINLINE_FUNCTION double ComputeSlope<SlopeType::SuperBee>(
    const double upwind, const double /*central*/, const double downwind) {
  return maxmod(minmod(upwind, 2.0 * downwind), minmod(2.0 * upwind, downwind));
}

template <SlopeType TYPE>
class Slope {
public:
  KOKKOS_FORCEINLINE_FUNCTION static double get(
      const double lower, const double middle, const double upper) {
    return ComputeSlope<TYPE>(middle - lower, (upper - lower) * 0.5, upper - middle);
  }

  /**
   * @brief Compute the limited slope
   *
   * @param UL    State in the lower neighbor cell
   * @param U     State in this cell
   * @param UU    State in the upper neighbor cell
   * @return get  Limited slope vector
   */
  template <typename State>
  KOKKOS_FORCEINLINE_FUNCTION static State get(const State &UL, const State &U, const State &UU) {
    State res;
    res[0] = get(UL[0], U[0], UU[0]);
    res[1] = get(UL[1], U[1], UU[1]);
    res[2] = get(UL[2], U[2], UU[2]);
    if (State::size > 3)
      res[3] = get(UL[3], U[3], UU[3]);
    if (State::size > 4)
      res[4] = get(UL[4], U[4], UU[4]);
    return res;
  }
};

#endif // ETHON_SLOPES_HPP_
