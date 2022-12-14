#!/usr/bin/env bash

rm lattice_filter.so
mkdir build_dir
cd build_dir


CUDA_COMPILER=/N/soft/rhel7/cuda/10.2/bin/nvcc
CXX_COMPILER=/N/soft/rhel7/gcc/8.4.0/bin/g++
CUDA_INCLUDE=/N/soft/rhel7/cuda/10.2/include

SPATIAL_DIMS=3
INPUT_CHANNELS=2
REFERENCE_CHANNELS=9
MAKE_TESTS=False

cmake -DCMAKE_BUILD_TYPE=Debug -D CMAKE_CUDA_COMPILER=${CUDA_COMPILER} \
                               -D CMAKE_CXX_COMPILER=${CXX_COMPILER} \
                               -D CMAKE_CUDA_HOST_COMPILER=${CXX_COMPILER} \
                               -D CUDA_INCLUDE=${CUDA_INCLUDE} \
                               -D SPATIAL_DIMS=${SPATIAL_DIMS} \
                               -D INPUT_CHANNELS=${INPUT_CHANNELS} \
                               -D REFERENCE_CHANNELS=${REFERENCE_CHANNELS} \
                               -D MAKE_TESTS=${MAKE_TESTS} \
			       -V \
                               -G "CodeBlocks - Unix Makefiles" ../


make

cp lattice_filter.so ../
cd ..
rm -r build_dir

