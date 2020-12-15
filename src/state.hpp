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

#ifndef ETHON_STATE_HPP_
#define ETHON_STATE_HPP_

#include <array>
#include <cassert>

#include <Kokkos_Core.hpp>

template <size_t DIM>
struct State_u {
  // mass density in g cm^{-3}
  double rho;

  // momentum density (mu_i = rho * u_i, where u_i is fluid velocity) in g cm^{-2} s^{-1}
  double mu[DIM];

  // total energy density (rho * e + 1/2 * rho * u_i u^i, where e is the internal specific energy)
  // in erg cm^{-3} = g cm^{-1} s^{-2}
  double epsilon;

  static constexpr size_t dim = DIM;
  static constexpr size_t size = DIM + 2;

  KOKKOS_FUNCTION State_u() : rho(0.0), epsilon(0.0) {
    for (size_t d = 0; d < DIM; ++d)
      mu[d] = 0.0;
  }

  template <typename EOS>
  static State_u FromPrimitive(
      const double rho, const std::array<double, DIM> u, const double pressure, const EOS &eos) {
    State_u res;

    res.rho = rho;
    for (size_t d = 0; d < DIM; ++d)
      res.mu[d] = u[d] * rho;
    res.epsilon = pressure / (eos.gamma() - 1.0) + 0.5 * res.mu_squared() / rho;

    return res;
  }

  // operator[] allows us to access the state as a vector
  KOKKOS_FORCEINLINE_FUNCTION double &operator[](const size_t i) {
    assert(i < size);
    return *(&rho + i);
  }

  KOKKOS_FORCEINLINE_FUNCTION double operator[](const size_t i) const {
    assert(i < size);
    return *(&rho + i);
  }

  // convenience functions
  KOKKOS_FORCEINLINE_FUNCTION double u(const size_t i) const {
    assert(i < dim);
    return mu[i] / rho;
  }

  KOKKOS_FORCEINLINE_FUNCTION double mu_squared() const {
    double res = mu[0] * mu[0];
    if (DIM > 1) res += mu[1] * mu[1];
    if (DIM > 2) res += mu[2] * mu[2];
    return res;
  }

  KOKKOS_FORCEINLINE_FUNCTION double specific_internal_energy() const {
    double res = (epsilon - 0.5 * mu_squared() / rho) / rho;
    // if (res < 0.0)
    //   printf("epsilon = %.10e, rho = %.10e, mu2 = %.10e, res = %.10e\n",
    //       epsilon,
    //       rho,
    //       mu_squared(),
    //       res);
    return res;
  }

  /**
   * @brief Get the flux in a particular direction
   *
   * @param i         The direction in whose direction the flux will be returned
   * @param U         The state in this cell
   * @param pressure  The pressure in this cell
   * @return State_u  The flux in the i-direction
   */
  KOKKOS_FUNCTION static State_u Flux(const size_t i, const State_u &U, const double pressure) {
    assert(i < DIM);
    State_u flux;

    flux.rho = U.mu[i];

    // fluid velocity in the i-direction
    double ui = U.u(i);
    for (size_t j = 0; j < DIM; ++j)
      flux.mu[j] = U.mu[j] * ui;

    flux.mu[i] += pressure;
    flux.epsilon = ui * (U.epsilon + pressure);

    return flux;
  }

//#define UNROLL _Pragma("GCC unroll 5")
#define UNROLL _Pragma("unroll 5")

  // operators
  KOKKOS_FORCEINLINE_FUNCTION State_u operator+(const State_u &other) const {
    State_u res;
    // UNROLL
    for (size_t i = 0; i < (DIM + 2); ++i)
      res[i] = (*this)[i] + other[i];

    return res;
  }

  KOKKOS_FORCEINLINE_FUNCTION State_u &operator+=(const State_u &other) {
    // UNROLL
    for (size_t i = 0; i < (DIM + 2); ++i)
      (*this)[i] += other[i];

    return *this;
  }

  KOKKOS_FORCEINLINE_FUNCTION State_u operator-(const State_u &other) const {
    State_u res;
    // UNROLL
    for (size_t i = 0; i < (DIM + 2); ++i)
      res[i] = (*this)[i] - other[i];

    return res;
  }

  KOKKOS_FORCEINLINE_FUNCTION State_u operator*(const double a) const {
    State_u res;
    // UNROLL
    for (size_t i = 0; i < (DIM + 2); ++i)
      res[i] = (*this)[i] * a;

    return res;
  }

  KOKKOS_FORCEINLINE_FUNCTION State_u &operator*=(const double a) {
    // UNROLL
    for (size_t i = 0; i < (DIM + 2); ++i)
      (*this)[i] *= a;

    return *this;
  }

  KOKKOS_FORCEINLINE_FUNCTION State_u operator/(const double a) const {
    State_u res;
    // UNROLL
    for (size_t i = 0; i < (DIM + 2); ++i)
      res[i] = (*this)[i] / a;

    return res;
  }

  void Dump() const {
    printf("rho = %.10e, epsilon = %.10e, spec. int eng = %.10e, mu = ",
        rho,
        epsilon,
        specific_internal_energy());

    for (size_t d = 0; d < DIM; ++d)
      printf("%.10e ", mu[d]);

    printf("\n");
  }
};

#endif // ETHON_STATE_HPP_
