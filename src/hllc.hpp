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

#ifndef ETHON_HLLC_HPP_
#define ETHON_HLLC_HPP_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>

/**
 * @brief HLLC Approximate Riemann Solver
 *
 * This is the HLLC Approximate Riemann Solver from Toro's book, section 10.6. We derive wave speed
 * estimates from the pressure in the star region. The star pressure could be determined using a
 * hybrid scheme where the pressure is derived differently depending on what the solution does in
 * the cell, see section 9.5.2 in Toro.
 *
 * @tparam State  State vector type
 * @tparam EOS    EOS type
 */
template <typename State, typename EOS_t>
class HLLC_u {
public:
  using EOS = EOS_t;

  HLLC_u(const EOS &eos) : eos_(eos) {}

  /**
   * @brief Compute the approximate flux in the i-direction given a lower and upper state.
   *
   * @param i         The index of the normal direction (this is a 1D Riemann Solver)
   * @param UL        The state on the lower (left) side of the face
   * @param UU        The state on the upper (right) side of the face
   * @return State    The flux in the i-direction
   */
  KOKKOS_FUNCTION State operator()(const size_t i, const State &UL, const State &UU) const {
    assert(i < State::dim);

    // convenience variables
    double aL = eos_.sound_speed(UL);
    double aU = eos_.sound_speed(UU);
    double pL = eos_.pressure(UL);
    double pU = eos_.pressure(UU);
    double uL = UL.mu[i] / UL.rho;
    double uU = UU.mu[i] / UU.rho;

    // get pressure in star region p_star
    double avg_rho = 0.5 * (UL.rho + UU.rho);
    double avg_sound_speed = 0.5 * (aL + aU);
    double p_star = fmax(0.0, 0.5 * (pL + pU) - 0.5 * (uU - uL) * avg_rho * avg_sound_speed);

    // get wave speed estimates
    double qL = 1.0;
    double qU = 1.0;
    if (p_star > pL)
      qL = sqrt(1.0 + (eos_.gamma() + 1.0) / (2.0 * eos_.gamma()) * (p_star / pL - 1.0));
    if (p_star > pU)
      qU = sqrt(1.0 + (eos_.gamma() + 1.0) / (2.0 * eos_.gamma()) * (p_star / pU - 1.0));

    double SL = uL - aL * qL;
    double SU = uU + aU * qU;

    double S_star = (pU - pL + UL.rho * uL * (SL - uL) - UU.rho * uU * (SU - uU)) /
                    (UL.rho * (SL - uL) - UU.rho * (SU - uU));

    // calculate flux
    if (0.0 <= SL) {
      return State::Flux(i, UL, pL);
    } else if (0.0 <= S_star) {
      auto U_starL = U_star(i, UL, SL, S_star, pL);
      return State::Flux(i, UL, pL) + (U_starL - UL) * SL;
    } else if (0.0 < SU) {
      auto U_starU = U_star(i, UU, SU, S_star, pU);
      return State::Flux(i, UU, pU) + (U_starU - UU) * SU;
    } else if (SU <= 0.0) {
      return State::Flux(i, UU, pU);
    } else {
      return State::Flux(i, UL, 1.0 / 0.0);

      // printf("UL: ");
      // UL.Dump();
      // printf("UU: ");
      // UU.Dump();

      // printf("SL = %.10e, S_star = %.10e, SU = %.10e\n", SL, S_star, SU);
      // printf("aL = %.10e, aU = %.10e, pL = %.10e, pU = %.10e, uL = %.10e, uU = %.10e\n",
      //     aL,
      //     aU,
      //     pL,
      //     pU,
      //     uL,
      //     uU);
      // printf("avg_rho = %.10e, avg_sounds_speed = %.10e, p_star = %.10e, qL = %.10e, qU = %.10e\n",
      //     avg_rho,
      //     avg_sound_speed,
      //     p_star,
      //     qL,
      //     qU);
      // printf("SL = %.10e, SU = %.10e, S_star = %.10e/%.10e\n",
      //     SL,
      //     SU,
      //     (pU - pL + UL.rho * uL * (SL - uL) - UU.rho * uU * (SU - uU)),
      //     (UL.rho * (SL - uL) - UU.rho * (SU - uU)));
      // throw std::runtime_error("Something went wrong in HLLC");
    }
  }

private:
  KOKKOS_INLINE_FUNCTION State U_star(int i, const State &U, double S, double S_star, double p) const {
    State res;

    double u = U.u(i);
    double fac = U.rho * (S - u) / (S - S_star);

    res.rho = fac;

    for (int d = 0; d < int(State::dim); ++d)
      res.mu[d] = fac * (d == i ? S_star : U.u(d));

    res.epsilon = fac * (U.epsilon / U.rho + (S_star - u) * (S_star + p / (U.rho * (S - u))));

    return res;
  }

  const EOS eos_;
};

#endif // ETHON_HLLC_HPP_
