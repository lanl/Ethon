# Ethon

Ethon is a research code to investigate different methods for hydro simulations with block-based
adaptive mesh refinement (AMR) on CPUs GPUs.

# Building

## Dependencies

Building Ethon requires the following.

- A C++14-capable compiler (test with `g++` 10)
- `CMake` (https://cmake.org/)
- `CUDA` (https://developer.nvidia.com/cuda-toolkit)
- `googletest` (https://github.com/google/googletest)
- `HDF5` (https://www.hdfgroup.org/solutions/hdf5/)
- `OpenMP`
- `p4est` (https://www.p4est.org/)

# Compiling

To obtain the Ethon source code and compile it, follow these steps

```
git clone git@github.com:lanl/Ethon.git
cd Ethon
mkdir build
cd build
CXX=../3rd_party/kokkos/bin/nvcc_wrapper cmake \
  -DKokkos_ENABLE_CUDA=ON     \
  -DKokkos_ARCH_VOLTA70=ON    \
  -DKokkos_ENABLE_OPENMP=ON ..
make -j
make test
```


# Documentation

The `doc` directory contains some notes on the physics implemented in Ethon. Ethon itself consists
of headers only and the `test` directory contains several 1D and 3D examples of how Ethon can be
used.

A code paper describing Ethon in detail is forthcoming.
