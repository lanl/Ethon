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

#ifndef ETHON_TEST_REGRESSION_HPP_
#define ETHON_TEST_REGRESSION_HPP_

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "state.hpp"
#include "uniform_mesh/uniform_mesh.hpp"

void DumpData(const std::string &fname,
    const std::vector<std::string> &column_names,
    const std::vector<std::vector<double>> &data) {
  std::ofstream ofs(fname, std::ofstream::trunc);

  for (size_t i = 0; i < column_names.size(); ++i) {
    ofs << "# [" << std::setw(2) << i + 1 << "] = " << column_names[i] << std::endl;
  }

  ofs.precision(15);
  ofs.setf(std::ios_base::scientific);
  for (size_t i = 0; i < data[0].size(); ++i) {
    for (size_t c = 0; c < data.size(); ++c) {
      ofs << std::setw(25) << data[c][i];
    }
    ofs << std::endl;
  }
}

template <size_t DIM>
std::pair<std::vector<std::string>, std::vector<std::vector<double>>> GetColumnNamesAndDat(
    const Mesh_u<1> &mesh, const std::vector<State_u<DIM>> &data) {
  std::vector<std::string> columns(3 + DIM);
  std::vector<std::vector<double>> dat(3 + DIM, std::vector<double>(data.size()));

  columns[0] = "x";
  columns[1] = "rho";
  columns[2] = "epsilon";
  for (size_t d = 0; d < DIM; ++d) {
    columns[d + 3] = "mu[" + std::to_string(d) + "]";
  }

  for (size_t i = 0; i < data.size(); ++i) {
    dat[0][i] = mesh.x({i})[0];
    dat[1][i] = data[i].rho;
    dat[2][i] = data[i].epsilon;
    for (size_t d = 0; d < DIM; ++d) {
      dat[3 + d][i] = data[i].mu[d];
    }
  }

  return {columns, dat};
}

template <size_t DIM>
void DumpStateData(
    const std::string &fname, const Mesh_u<1> &mesh, const std::vector<State_u<DIM>> &data) {
  auto col_dat = GetColumnNamesAndDat(mesh, data);
  DumpData(fname, col_dat.first, col_dat.second);
}

std::vector<std::vector<double>> ReadData(const std::string &fname, const size_t num_cols) {
  std::vector<std::vector<double>> res(num_cols);
  std::ifstream ifs(fname);

  // expect num_cols comment lines
  std::string line;
  for (size_t i = 0; i < num_cols; ++i) {
    std::getline(ifs, line);
    if (line[0] != '#') throw std::runtime_error("Expected comment line in file " + fname);
  }

  // now read data
  while (true) {
    for (size_t i = 0; i < num_cols; ++i) {
      double val;
      ifs >> val;
      if (ifs.eof()) break;
      res[i].push_back(val);
    }

    if (ifs.eof()) break;
  }

  return res;
}

template <size_t DIM>
std::pair<std::vector<double>, std::vector<State_u<DIM>>> ReadStateData(const std::string &fname) {
  auto dat = ReadData(fname, 3 + DIM);
  std::vector<double> xs(dat[0].size());
  std::vector<State_u<DIM>> states(dat[0].size());

  for (size_t i = 0; i < xs.size(); ++i) {
    xs[i] = dat[0][i];
    states[i].rho = dat[1][i];
    states[i].epsilon = dat[2][i];
    for (size_t d = 0; d < DIM; ++d)
      states[i].mu[d] = dat[3 + d][i];
  }

  return {xs, states};
}

double RegressionTest(const std::string &fname,
    const std::vector<std::string> &column_names,
    const std::vector<std::vector<double>> &data,
    const bool write_gold = false) {
  if (write_gold) {
    // we'll just write the gold output files without comparing them to anything
    DumpData(fname + ".gold", column_names, data);
    return 0.0;
  } else {
    // we'll dump our outputs and compare them to the gold files
    DumpData(fname + ".out", column_names, data);
    auto gold = ReadData(fname + ".gold", column_names.size());

    // find maximum difference
    if (gold[0].size() != data[0].size())
      throw std::invalid_argument("Size mismatch between data (" + std::to_string(data[0].size()) +
                                  ") and gold (" + std::to_string(gold[0].size()) +
                                  ") in regression test " + fname);

    double max_diff = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
      for (size_t c = 0; c < data.size(); ++c) {
        max_diff = std::max(max_diff, fabs(data[c][i] - gold[c][i]));
      }
    }

    printf("Regression max diff for %s: %.8e\n", fname.c_str(), max_diff);
    return max_diff;
  }
}

template <size_t DIM>
double RegressionTest(const std::string &fname,
    const Mesh_u<1> &mesh,
    const std::vector<State_u<DIM>> &data,
    const bool write_gold = false) {
  auto col_dat = GetColumnNamesAndDat(mesh, data);
  return RegressionTest(fname, col_dat.first, col_dat.second, write_gold);
}

#endif // ETHON_TEST_REGRESSION_HPP_
