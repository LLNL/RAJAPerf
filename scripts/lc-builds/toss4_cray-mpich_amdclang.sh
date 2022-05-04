#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [[ $# -lt 2 ]]; then
  echo
  echo "You must pass 2 or more arguments to the script (in this order): "
  echo "   1) cray-mpich compiler version number"
  echo "   1) HIP compiler version number"
  echo "   2) HIP compute architecture"
  echo "   3...) optional arguments to cmake"
  echo
  echo "For example: "
  echo "    toss4_cray-mpich_amdclang.sh 8.1.14 4.1.0 gfx906"
  exit
fi

MPI_VER=$1
COMP_VER=$2
COMP_ARCH=$3
shift 3

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

BUILD_SUFFIX=lc_toss4-cray-mpich-${MPI_VER}-amdclang-${COMP_VER}-${COMP_ARCH}
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/toss4/${HOSTCONFIG}.cmake

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}


module load cmake/3.14.5

# unload rocm to avoid configuration problems where the loaded rocm and COMP_VER
# are inconsistent causing the rocprim from the module to be used unexpectedly
module unload rocm


cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DMPI_C_COMPILER="/usr/tce/packages/cray-mpich-tce/cray-mpich-${MPI_VER}-rocmcc-${COMP_VER}/bin/mpiamdclang" \
  -DMPI_CXX_COMPILER="/usr/tce/packages/cray-mpich-tce/cray-mpich-${MPI_VER}-rocmcc-${COMP_VER}/bin/mpiamdclang++" \
  -DROCM_ROOT_DIR="/opt/rocm-${COMP_VER}" \
  -DHIP_ROOT_DIR="/opt/rocm-${COMP_VER}/hip" \
  -DHIP_PATH=/opt/rocm-${COMP_VER}/llvm/bin \
  -DCMAKE_C_COMPILER=/opt/rocm-${COMP_VER}/llvm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-${COMP_VER}/llvm/bin/amdclang++ \
  -DCMAKE_HIP_ARCHITECTURES="${COMP_ARCH}" \
  -DBLT_CXX_STD=c++14 \
  -C ${RAJA_HOSTCONFIG} \
  -DENABLE_MPI=ON \
  -DENABLE_HIP=ON \
  -DENABLE_OPENMP=OFF \
  -DENABLE_CUDA=OFF \
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
echo "  Please note that cray-mpich requires libmodules.so.1 from cce to run."
echo "  Until this is handled transparently in the build system you may add "
echo "  cce to your LD_LIBRARY_PATH."
echo
echo "    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/tce/packages/cce-tce/cce-13.0.2/cce/x86_64/lib/"
echo
echo "***********************************************************************"
