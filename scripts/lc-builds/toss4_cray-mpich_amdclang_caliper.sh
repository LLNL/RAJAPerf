#!/usr/bin/env bash

###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other
# RAJA Project Developers. See top-level LICENSE and COPYRIGHT
# files for dates and other details. No copyright assignment is required
# to contribute to RAJA Performance Suite.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

# Default CMake version if not provided
DEFAULT_CMAKE_VER=3.24.2

if [[ $# -lt 5 ]]; then
  echo
  echo "You must pass 5 or more arguments to the script (in this order): "
  echo "   1) cray-mpich version number"
  echo "   2) HIP compiler version number"
  echo "   3) HIP compute architecture"
  echo "   4) path to caliper cmake directory"
  echo "   5) path to adiak cmake directory"
  echo "   6) optional CMake version to load."
  echo
  echo "For example: "
  echo "    toss4_cray-mpich_amdclang.sh 8.1.14 4.1.0 gfx906 /usr/workspace/wsb/asde/caliper-quartz/share/cmake/caliper /usr/workspace/wsb/asde/caliper-quartz/lib/cmake/adiak [3.27.4]"
  exit 1
fi

MPI_VER=$1
COMP_VER=$2
COMP_ARCH=$3
CALI_DIR=$4
ADIAK_DIR=$5

# Detect optional sixth positional argument as a CMake version if it looks like N.M or N.M.P
# Otherwise, treat it as a normal CMake argument.
if [ -n "$6" ] && [[ "$6" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
  CMAKE_VER=$6
  shift 6
else
  CMAKE_VER=$DEFAULT_CMAKE_VER
  shift 5
fi

HOSTCONFIG="hip_3_X"

if [[ ${COMP_VER} == 4.* ]]
then
##HIP_CLANG_FLAGS="-mllvm -amdgpu-fixed-function-abi=1"
  HOSTCONFIG="hip_4_link_X"
elif [[ ${COMP_VER} == 3.* ]]
then
  HOSTCONFIG="hip_3_X"
else
  echo "Unknown hip version, using ${HOSTCONFIG} host-config"
fi

# if [[ ${COMP_ARCH} == gfx90a ]]
# then
  # note that unsafe atomics require use of coarse grain memory
##HIP_CLANG_FLAGS="-munsafe-fp-atomics"
# fi

BUILD_SUFFIX=lc_toss4-cray-mpich-${MPI_VER}-amdclang-${COMP_VER}-${COMP_ARCH}_caliper
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/toss4/${HOSTCONFIG}.cmake

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo "Using CMake version: ${CMAKE_VER}"
echo "Configuration extra arguments:"
echo "   $@"
echo
echo "To get cmake to work you may have to configure with"
echo "   -DHIP_PLATFORM=amd"
echo
echo "To use fp64 HW atomics you must configure with these options when using gfx90a and hip >= 5.2"
echo "   -DCMAKE_CXX_FLAGS=\"-munsafe-fp-atomics\""
echo
echo "To work around some issues where *_FUSED kernels crash add these options"
echo "   -DCMAKE_CXX_FLAGS=\"-fgpu-rdc\""
echo "   -DCMAKE_EXE_LINKER_FLAGS=\"-fgpu-rdc\""
echo
echo "To work around some issues where *_FUSED kernels perform poorly use this environment variable"
echo "   env HSA_SCRATCH_SINGLE_LIMIT=4000000000"
echo
echo "To work around some issues where the build fails with a weird error about max or fmax add these options"
echo "   -DCMAKE_CXX_FLAGS=\"--hip-version={hip_version:ex=6.1.2}\""
echo "   -DCMAKE_EXE_LINKER_FLAGS=\"--hip-version={hip_version:ex=6.1.2}\""
echo



rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}


module load cmake/${CMAKE_VER}

# unload rocm to avoid configuration problems where the loaded rocm and COMP_VER
# are inconsistent causing the rocprim from the module to be used unexpectedly
module unload rocm rocmcc

if [[ "${COMP_VER}" == *-magic ]]; then
  ROCM_PATH="/usr/tce/packages/rocmcc/rocmcc-${COMP_VER}"
  MPI_ROCM_PATH="/usr/tce/packages/cray-mpich/cray-mpich-${MPI_VER}-rocmcc-${COMP_VER}"
else
  ROCM_PATH="/opt/rocm-${COMP_VER}"
  MPI_ROCM_PATH=/usr/tce/packages/cray-mpich-tce/cray-mpich-${MPI_VER}-rocmcc-${COMP_VER}
fi

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI_C_COMPILER="${MPI_ROCM_PATH}/bin/mpiamdclang" \
  -DMPI_CXX_COMPILER="${MPI_ROCM_PATH}/bin/mpiamdclang++" \
  -DCMAKE_PREFIX_PATH="${ROCM_PATH}/lib/cmake" \
  -DHIP_PLATFORM=amd \
  -DROCM_ROOT_DIR="${ROCM_PATH}" \
  -DHIP_ROOT_DIR="${ROCM_PATH}/hip" \
  -DHIP_PATH="${ROCM_PATH}/llvm/bin" \
  -DCMAKE_C_COMPILER="${ROCM_PATH}/llvm/bin/amdclang" \
  -DCMAKE_CXX_COMPILER="${ROCM_PATH}/llvm/bin/amdclang++" \
  -DCMAKE_HIP_ARCHITECTURES="${COMP_ARCH}" \
  -DGPU_TARGETS="${COMP_ARCH}" \
  -DAMDGPU_TARGETS="${COMP_ARCH}" \
  -DBLT_CXX_STD=c++20 \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_MPI=ON \
  -DENABLE_HIP=ON \
  -DENABLE_OPENMP=OFF \
  -DENABLE_CUDA=OFF \
  -DRAJA_PERFSUITE_USE_CALIPER=ON \
  -Dcaliper_DIR=${CALI_DIR} \
  -Dadiak_DIR=${ADIAK_DIR} \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..

echo
echo "***********************************************************************"
echo
echo "cd into directory build_${BUILD_SUFFIX} and run make to build RAJA"
echo
echo "  Please note that you have to have a consistent build environment"
echo "  when you make RAJA as cmake may reconfigure; unload the rocm module"
echo "  or load the appropriate rocm module (${COMP_VER}) when building."
echo
echo "    module unload rocm"
echo "    srun -n1 make"
echo
echo "  Please note that rocm requires libpgmath.so from rocm/llvm to run."
echo "  Until this is handled transparently in the build system you may add "
echo "  rocm/llvm to your LD_LIBRARY_PATH."
echo
echo "    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm-${COMP_VER}/llvm/lib"
echo
echo "***********************************************************************"
