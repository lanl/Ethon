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

#ifndef ETHON_BLOCK_MESH_OUTPUT_HPP_
#define ETHON_BLOCK_MESH_OUTPUT_HPP_

#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "hdf5_helper.hpp"

using namespace Kokkos;

namespace output {

template <typename BLOCK, typename EOS>
void DumpBlock(const BLOCK *block,
    double *data,
    double *x1f,
    double *x2f,
    double *x3f,
    int *levels,
    const size_t block_idx,
    const size_t num_blocks,
    const EOS &eos,
    const bool z_slice) {
  const size_t N = BLOCK::N;
  const size_t N_ghost = BLOCK::N_ghost;

  const size_t block_offset = N * N * (z_slice ? 1 : N);
  const size_t field_offset = num_blocks * block_offset;

  levels[block_idx] = block->level();
  for (size_t i = 0; i < N + 1; ++i) {
    auto idx = block_idx * (N + 1) + i;
    x1f[idx] = block->lower_bounds()[0] + double(i) * block->cell_size()[0];
    x2f[idx] = block->lower_bounds()[1] + double(i) * block->cell_size()[1];
    if (!z_slice) x3f[idx] = block->lower_bounds()[2] + double(i) * block->cell_size()[2];
  }

  if (z_slice) {
    x3f[block_idx * 2] = block->lower_bounds()[2];
    x3f[block_idx * 2 + 1] = block->lower_bounds()[2] + block->cell_size()[2];
  }

  // auto rho = block->rho();
  // auto mu0 = block->mu0();
  // auto mu1 = block->mu1();

  // double max = 0.0;
  MinMaxScalar<double> minmax;
  minmax.min_val = 1.0E300;
  minmax.max_val = 0.0;

  for (size_t k = N_ghost - 1; k < N + N_ghost + 1; ++k) {
    for (size_t j = N_ghost - 1; j < N + N_ghost + 1; ++j) {
      for (size_t i = N_ghost - 1; i < N + N_ghost + 1; ++i) {
        int num_outside = 0;
        if ((i < N_ghost) || (i >= N + N_ghost)) ++num_outside;
        if ((j < N_ghost) || (j >= N + N_ghost)) ++num_outside;
        if ((k < N_ghost) || (k >= N + N_ghost)) ++num_outside;

        if (num_outside <= 1) {
          double press = eos.pressure(block->GetState(i, j, k));
          minmax.min_val = std::min(minmax.min_val, press);
          minmax.max_val = std::max(minmax.max_val, press);
        }
      }
    }
  }

  for (size_t K = 0; K < N; ++K) {
    for (size_t J = 0; J < N; ++J) {
      for (size_t I = 0; I < N; ++I) {
        if (z_slice && (K != 0)) return;

        auto i = I + BLOCK::N_ghost;
        auto j = J + BLOCK::N_ghost;
        auto k = K + BLOCK::N_ghost;

        size_t idx = K * N * N + J * N + I;
        auto s = block->GetState(i, j, k);

        data[0 * field_offset + block_idx * block_offset + idx] = s.rho;
        data[1 * field_offset + block_idx * block_offset + idx] = s.u(0);
        data[2 * field_offset + block_idx * block_offset + idx] = s.u(1);
        data[3 * field_offset + block_idx * block_offset + idx] = s.u(2);
        data[4 * field_offset + block_idx * block_offset + idx] = s.epsilon;
        data[5 * field_offset + block_idx * block_offset + idx] = eos.pressure(s);
        data[6 * field_offset + block_idx * block_offset + idx] = s.specific_internal_energy();

        data[7 * field_offset + block_idx * block_offset + idx] = minmax.max_val / minmax.min_val;

        // double vgy =
        //     fabs(mu1(i + 1, j, k) / rho(i + 1, j, k) - mu1(i - 1, j, k) / rho(i - 1, j, k)) *
        //     0.5;
        // double vgx =
        //     fabs(mu0(i, j + 1, k) / rho(i, j + 1, k) - mu0(i, j - 1, k) / rho(i, j - 1, k)) *
        //     0.5;

        // double vg = sqrt(vgx * vgx + vgy * vgy);
        // buffer[7 * field_offset + block_idx * block_offset + idx] = vg;

        // buffer[8 * field_offset + block_idx * block_offset + idx] = max;
      }
    }
  }
}

template <typename BLOCK, typename Mesh_t, typename EOS>
void DumpGrid(const std::string &fname,
    const size_t num_steps,
    const double time,
    const Mesh_t &mesh,
    const EOS &eos,
    const std::vector<BLOCK *> &all_blocks,
    const bool z_slice) {
  // extract filename
  std::string filename;
  auto last_slash = fname.rfind("/");
  if (last_slash == std::string::npos)
    filename = fname;
  else
    filename = fname.substr(last_slash + 1);

  auto h5_fname = filename + ".athdf";
  auto xdmf_fname = fname + ".xdmf";
  std::string data_set_name = "data";

  const size_t N = BLOCK::N;

  // fields to be written to disk
  const std::vector<std::string> fields{
      "rho", "ux", "uy", "uz", "eps", "press", "int_eng", "max_min_press_ratio"};
  const size_t num_fields = fields.size();

  std::vector<BLOCK *> out_blocks;
  if (z_slice) {
    for (auto b : all_blocks)
      if (b->lower_bounds()[2] == 0.0) out_blocks.push_back(b);
  } else {
    out_blocks = all_blocks;
  }
  auto Nb = out_blocks.size();

  // write HDF5 file
  {
    // buffer that is filled in in parallel by all blocks
    double *data = (double *)malloc(sizeof(double) * num_fields * Nb * N * N * (z_slice ? 1 : N));
    double *x1f = (double *)malloc(sizeof(double) * Nb * (N + 1));
    double *x2f = (double *)malloc(sizeof(double) * Nb * (N + 1));
    double *x3f = (double *)malloc(sizeof(double) * Nb * (z_slice ? 2 : N + 1));
    int *levels = (int *)malloc(sizeof(int) * Nb);

    memset(data, 0, sizeof(double) * num_fields * Nb * N * N * (z_slice ? 1 : N));
    memset(x1f, 0, sizeof(double) * Nb * (N + 1));
    memset(x2f, 0, sizeof(double) * Nb * (N + 1));
    memset(x3f, 0, sizeof(double) * Nb * (z_slice ? 2 : N + 1));
    memset(levels, 0, sizeof(int) * Nb);

    int max_level = -1;
#pragma omp parallel for reduction(max : max_level)
    for (size_t i = 0; i < Nb; ++i) {
      DumpBlock(out_blocks[i], data, x1f, x2f, x3f, levels, i, Nb, eos, z_slice);
      max_level = std::max(max_level, out_blocks[i]->level());
    }

    H5::H5File file(fname + ".athdf", H5F_ACC_TRUNC);

    WriteToH5(&file, {num_fields, Nb, z_slice ? 1 : N, N, N}, data_set_name, data);
    WriteToH5(&file, {Nb, N + 1}, "x1f", x1f);
    WriteToH5(&file, {Nb, N + 1}, "x2f", x2f);
    WriteToH5(&file, {Nb, (z_slice ? 2 : N + 1)}, "x3f", x3f);
    WriteToH5(&file, {Nb}, "Levels", levels);

    AddH5Attribute(&file, "Coordinates", "cartesian");
    AddH5Attribute(&file, "DatasetNames", std::vector<std::string>{"data"});
    AddH5Attribute(&file, "MaxLevel", max_level);
    AddH5Attribute(&file, "MeshBlockSize", std::vector<size_t>{N, N, z_slice ? 1 : N});
    AddH5Attribute(&file, "NumCycles", num_steps);
    AddH5Attribute(&file, "NumMeshBlocks", out_blocks.size());
    AddH5Attribute(&file, "NumVariables", std::vector<size_t>{num_fields});
    AddH5Attribute(
        &file, "RootGridSize", std::vector<size_t>{mesh.N()[0], mesh.N()[1], mesh.N()[2]});
    AddH5Attribute(&file,
        "RootGridX1",
        std::vector<double>{mesh.lower_bounds()[0], mesh.upper_bounds()[0], 1.0});
    AddH5Attribute(&file,
        "RootGridX2",
        std::vector<double>{mesh.lower_bounds()[1], mesh.upper_bounds()[1], 1.0});
    AddH5Attribute(&file,
        "RootGridX3",
        std::vector<double>{mesh.lower_bounds()[2], mesh.upper_bounds()[2], 1.0});
    AddH5Attribute(&file, "Time", time);
    AddH5Attribute(&file, "VariableNames", fields);

    file.close();

    free(data);
    free(x1f);
    free(x2f);
    free(x3f);
    free(levels);
  }

  // write XDMF file
  {
    auto f = fopen(xdmf_fname.c_str(), "w");
    fprintf(f, "<?xml version=\"1.0\" ?>\n");
    fprintf(f, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
    fprintf(f, "<Xdmf Version=\"2.0\">\n");
    fprintf(f, "<Information Name=\"TimeVaryingMetaData\" Value=\"True\"/>\n");
    fprintf(f, "<Domain>\n");
    fprintf(f, "<Grid Name=\"Mesh\" GridType=\"Collection\">\n");
    fprintf(f, "  <Time Value=\"%.8f\"/>\n", time);

    for (size_t i = 0; i < out_blocks.size(); ++i) {
      auto lb = out_blocks[i]->lower_bounds();
      auto cs = out_blocks[i]->cell_size();

      fprintf(f, "  <Grid Name=\"block_%lu\" GridType=\"Uniform\">\n", i);
      fprintf(
          f, "    <Geometry Origin=\"\" Type=\"%s\">\n", z_slice ? "ORIGIN_DXDY" : "ORIGIN_DXDYDZ");
      fprintf(f,
          "      <DataItem DataType=\"Float\" Dimensions=\"%i\" Format=\"XML\" "
          "Precision=\"8\">%.10f %.10f",
          z_slice ? 2 : 3,
          lb[0],
          lb[1]);
      if (!z_slice) fprintf(f, " %.10f", lb[2]);
      fprintf(f, "</DataItem>\n");
      fprintf(f,
          "      <DataItem DataType=\"Float\" Dimensions=\"%i\" Format=\"XML\" "
          "Precision=\"8\">%.10f %.10f",
          z_slice ? 2 : 3,
          cs[0],
          cs[1]);
      if (!z_slice) fprintf(f, " %.10f", cs[2]);
      fprintf(f, "</DataItem>\n");
      fprintf(f, "    </Geometry>\n");
      fprintf(f, "    <Topology Dimensions=\"%lu %lu", N + 1, N + 1);
      if (!z_slice) fprintf(f, " %lu", N + 1);
      fprintf(f, "\" Type=\"%s\"/>\n", z_slice ? "2DCoRectMesh" : "3DCoRectMesh");

      for (size_t j = 0; j < fields.size(); ++j) {
        fprintf(f, "    <Attribute Name=\"%s\" Center=\"Cell\">\n", fields[j].c_str());
        fprintf(f, "      <DataItem ItemType=\"HyperSlab\" Dimensions=\"%lu %lu", N, N);
        if (!z_slice) fprintf(f, " %lu", N);
        fprintf(f, "\">\n");
        fprintf(f,
            "        <DataItem Dimensions=\"3 %i\" NumberType=\"Int\"> %lu %lu 0 0 %s 1 1 1 1 1 1 "
            "%lu %lu",
            z_slice ? 4 : 5,
            j,
            i,
            z_slice ? "" : "0 1",
            N,
            N);
        if (!z_slice) fprintf(f, " %lu", N);
        fprintf(f, " </DataItem>\n");
        fprintf(f,
            "        <DataItem Dimensions=\"%lu %lu %lu %lu",
            num_fields,
            out_blocks.size(),
            N,
            N);
        if (!z_slice) fprintf(f, " %lu", N);
        fprintf(
            f, "\" Format=\"HDF\"> %s:/%s </DataItem>\n", h5_fname.c_str(), data_set_name.c_str());
        fprintf(f, "      </DataItem>\n");
        fprintf(f, "    </Attribute>\n");
      }

      fprintf(f, "  </Grid>\n");
    }

    fprintf(f, "</Grid>\n");
    fprintf(f, "</Domain>\n");
    fprintf(f, "</Xdmf>\n");

    fclose(f);
  }
}

} // namespace output

#endif // ETHON_BLOCK_MESH_OUTPUT_HPP_
