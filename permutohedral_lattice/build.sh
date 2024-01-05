#!/usr/bin/env bash

rm lattice_filter.so
mkdir build_dir
cd build_dir


CUDA_COMPILER=/N/soft/sles15/cuda/11.7/bin/nvcc
CXX_COMPILER=/N/soft/rhel8/gcc/11.2.0/bin/g++
CUDA_INCLUDE=/N/soft/sles15/cuda/11.7/include

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
                               -G "CodeBlocks - Unix Makefiles" ../


make

cp lattice_filter.so ../
cd ..
rm -r build_dir