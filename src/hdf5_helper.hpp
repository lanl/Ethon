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

#ifndef ETHON_HDF5_HELPER_HPP_
#define ETHON_HDF5_HELPER_HPP_

#include <stdexcept>
#include <string>
#include <vector>

#include <H5Cpp.h>

template <typename T>
H5::DataType GetH5DataType() {
  if (std::is_same<T, double>::value)
    return H5::DataType(H5::PredType::NATIVE_DOUBLE);
  else if (std::is_same<T, int>::value)
    return H5::DataType(H5::PredType::NATIVE_INT32);
  else if (std::is_same<T, size_t>::value)
    return H5::DataType(H5::PredType::NATIVE_UINT64);
  else if (std::is_same<T, std::string>::value || std::is_same<T, const char *>::value)
    return H5::DataType(H5::StrType(0, H5T_VARIABLE));
  else
    throw std::invalid_argument("Unknown type");
}

template <typename T>
void WriteToH5(H5::H5File *const h5file,
    const std::vector<hsize_t> dims,
    const std::string &name,
    const T *const data) {
  H5::DataSpace data_space(dims.size(), dims.data());

  auto data_type = GetH5DataType<T>();
  auto data_set = h5file->createDataSet(name, data_type, data_space);
  data_set.write(data, data_type);
}

template <typename T>
void AddH5Attribute(H5::H5Object *const obj, const std::string &name, T value) {
  H5::DataSpace ds(H5S_SCALAR);
  auto type = GetH5DataType<T>();
  auto attr = obj->createAttribute(name, type, ds);
  attr.write(type, &value);
}

template <>
void AddH5Attribute(H5::H5Object *const obj, const std::string &name, std::string value) {
  H5::DataSpace ds(H5S_SCALAR);
  auto type = GetH5DataType<std::string>();
  auto attr = obj->createAttribute(name, type, ds);
  attr.write(type, value);
}

template <typename T>
void AddH5Attribute(
    H5::H5Object *const obj, const std::string &name, const std::vector<T> &values) {
  hsize_t size = values.size();
  H5::DataSpace ds(1, &size);
  auto type = GetH5DataType<T>();
  auto attr = obj->createAttribute(name, type, ds);

  attr.write(type, values.data());
}

template <>
void AddH5Attribute(
    H5::H5Object *const obj, const std::string &name, const std::vector<std::string> &values) {
  hsize_t size = values.size();
  H5::DataSpace ds(1, &size);
  auto type = GetH5DataType<std::string>();
  auto attr = obj->createAttribute(name, type, ds);

  std::vector<const char *> char_ptrs;
  for (auto &s : values)
    char_ptrs.push_back(s.c_str());
  attr.write(type, char_ptrs.data());
}

#endif // ETHON_HDF5_HELPER_HPP_
