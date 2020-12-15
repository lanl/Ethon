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

#ifndef ETHON_HDYRO_3D_HPP_
#define ETHON_HDYRO_3D_HPP_

#include <H5Cpp.h>
#include <Kokkos_Core.hpp>

#include "uniform_mesh/boundary.hpp"
#include "state.hpp"
#include "uniform_mesh/uniform_mesh.hpp"

template <size_t NUM_GHOST>
class Hydro3D {
public:
  using State = State_u<3>;
  using StateData = Kokkos::View<State ***, LayoutStride, HostSpace>;
  using Mesh = Mesh_u<3>;
  using output_func_t =
      std::function<void(const size_t, const double, const Mesh &, const StateData)>;
  using ExecSpace = Kokkos::OpenMP;

  static output_func_t no_output() {
    return [](const size_t, const double, const Mesh &, const StateData) { return; };
  }

  static constexpr size_t num_ghost = NUM_GHOST;

  template <typename EOS>
  static KOKKOS_FORCEINLINE_FUNCTION double min_dt_cell(
      const State &U, const Array<double, 3> &cell_size, const EOS &eos) {
    double cs = eos.sound_speed(U);

    double min_dt = cell_size[0] / (fabs(U.u(0)) + cs);
    min_dt = fmin(min_dt, cell_size[1] / (fabs(U.u(1)) + cs));
    min_dt = fmin(min_dt, cell_size[2] / (fabs(U.u(2)) + cs));

    return min_dt;
  }

  static void Dump_cell(const State &U) {
    printf("rho = %.10e, mu = (%.10e, %.10e, %.10e), epsilon = %.10e\n",
        U.rho,
        U.mu[0],
        U.mu[1],
        U.mu[2],
        U.epsilon);
  }

  template <typename EOS>
  static void DumpGrid(const std::string &fname,
      const double time,
      const Mesh &mesh,
      const StateData data,
      const EOS &eos) {
    std::array<unsigned int, 3> Ns(
        {(unsigned int)mesh.N()[0], (unsigned int)mesh.N()[1], (unsigned int)mesh.N()[2]});
    size_t num_cells = Ns[0] * Ns[1] * Ns[2];

    // extract filename
    std::string filename;
    auto last_slash = fname.rfind("/");
    if (last_slash == std::string::npos)
      filename = fname;
    else
      filename = fname.substr(last_slash + 1);

    auto h5_fname = filename + ".h5";
    auto xdmf_fname = fname + ".xdmf";
    std::string data_set_name = "data";

    // fields to be written to disk
    const std::vector<std::string> fields{
        "rho", "ux", "uy", "uz", "u_mag", "eps", "press", "int_eng", "cs"};
    const size_t num_fields = fields.size();

    // write HDF5 file
    {
      // buffer that is filled in in parallel by all blocks
      double *buffer = (double *)malloc(sizeof(double) * num_fields * num_cells);
      memset(buffer, 0, sizeof(double) * num_fields * num_cells);

      parallel_for(MDRangePolicy<ExecSpace, Rank<3>>({0, 0, 0}, {Ns[0], Ns[1], Ns[2]}),
          [&](const int &I, const int &J, const int &K) {
            auto i = I + num_ghost;
            auto j = J + num_ghost;
            auto k = K + num_ghost;

            auto idx = K * Ns[0] * Ns[1] + J * Ns[0] + I;
            auto s = data(i, j, k);

            buffer[0 * num_cells + idx] = s.rho;
            buffer[1 * num_cells + idx] = s.u(0);
            buffer[2 * num_cells + idx] = s.u(1);
            buffer[3 * num_cells + idx] = s.u(2);
            buffer[4 * num_cells + idx] = s.mu_squared() / (s.rho * s.rho);
            buffer[5 * num_cells + idx] = s.epsilon;
            buffer[6 * num_cells + idx] = eos.pressure(s);
            buffer[7 * num_cells + idx] = s.specific_internal_energy();
            buffer[8 * num_cells + idx] = eos.sound_speed(s);
          });

      H5::H5File file(fname + ".h5", H5F_ACC_TRUNC);

      hsize_t dims[4] = {num_fields, Ns[0], Ns[1], Ns[2]};
      H5::DataSpace data_space(4, dims);

      auto data_type = H5::PredType::NATIVE_DOUBLE;
      auto data_set = file.createDataSet(data_set_name, data_type, data_space);
      data_set.write(buffer, data_type);

      file.close();

      free(buffer);
    }

    // write XDMF file
    {
      auto f = fopen(xdmf_fname.c_str(), "w");

      auto lb = mesh.lower_bounds();
      auto cs = mesh.cell_size();

      fprintf(f, "<?xml version=\"1.0\" ?>\n");
      fprintf(f, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
      fprintf(f, "<Xdmf Version=\"2.0\">\n");
      fprintf(f, "<Information Name=\"TimeVaryingMetaData\" Value=\"True\"/>\n");
      fprintf(f, "<Domain>\n");
      fprintf(f, "<Grid Name=\"Mesh\" GridType=\"Uniform\">\n");
      fprintf(f, "  <Time Value=\"%.8f\"/>\n", time);
      fprintf(f, "  <Geometry Origin=\"\" Type=\"ORIGIN_DXDYDZ\">\n");
      fprintf(f,
          "    <DataItem DataType=\"Float\" Dimensions=\"3\" Format=\"XML\" "
          "Precision=\"8\">%.10f %.10f %.10f</DataItem>\n",
          lb[0],
          lb[1],
          lb[2]);
      fprintf(f,
          "    <DataItem DataType=\"Float\" Dimensions=\"3\" Format=\"XML\" "
          "Precision=\"8\">%.10f %.10f %.10f</DataItem>\n",
          cs[0],
          cs[1],
          cs[2]);
      fprintf(f, "  </Geometry>\n");
      fprintf(f,
          "  <Topology Dimensions=\"%i %i %i\" Type=\"3DCoRectMesh\"/>\n",
          Ns[2] + 1,  // <<< need to specify dimensions backwards here
          Ns[1] + 1,  // <<<
          Ns[0] + 1); // <<<

      for (size_t j = 0; j < fields.size(); ++j) {
        fprintf(f, "  <Attribute Name=\"%s\" Center=\"Cell\">\n", fields[j].c_str());
        fprintf(f,
            "    <DataItem ItemType=\"HyperSlab\" Dimensions=\"%i %i %i\">\n",
            Ns[0],
            Ns[1],
            Ns[2]);
        fprintf(f,
            "      <DataItem Dimensions=\"3 4\" NumberType=\"Int\"> %lu 0 0 0 1 1 1 1 1 %i %i %i "
            "</DataItem>\n",
            j,
            Ns[0],
            Ns[1],
            Ns[2]);
        fprintf(f,
            "      <DataItem Dimensions=\"%lu %i %i %i\" Format=\"HDF\"> %s:/%s </DataItem>\n",
            num_fields,
            Ns[0],
            Ns[1],
            Ns[2],
            h5_fname.c_str(),
            data_set_name.c_str());
        fprintf(f, "    </DataItem>\n");
        fprintf(f, "  </Attribute>\n");
      }

      fprintf(f, "</Grid>\n");
      fprintf(f, "</Domain>\n");
      fprintf(f, "</Xdmf>\n");

      fclose(f);
    }
  }
};

#endif // ETHON_HDYRO_3D_HPP_
