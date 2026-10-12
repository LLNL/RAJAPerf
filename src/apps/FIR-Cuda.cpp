//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <algorithm>
#include <iostream>

namespace rajaperf
{
namespace apps
{

#define USE_CUDA_CONSTANT_MEMORY
//#undef USE_CUDA_CONSTANT_MEMORY

#if defined(USE_CUDA_CONSTANT_MEMORY)

__constant__ Real_type coeff[FIR_COEFFLEN];

#define FIR_DATA_SETUP_CUDA \
  Real_type *dcoeff_addr; \
  CAMP_CUDA_API_INVOKE_AND_CHECK( cudaGetSymbolAddress, \
      (void**)&dcoeff_addr, coeff ); \
  CAMP_CUDA_API_INVOKE_AND_CHECK( cudaMemcpyAsync, \
      dcoeff_addr, coeff_array, FIR_COEFFLEN * sizeof(Real_type), \
      cudaMemcpyHostToDevice, res.get_stream() );


#define FIR_DATA_TEARDOWN_CUDA

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void fir(Real_ptr out, Real_ptr in,
                    const Index_type coefflen,
                    Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     FIR_BODY;
   }
}

#else  // use global memory for coefficients

#define FIR_DATA_SETUP_CUDA \
  Real_ptr coeff; \
  \
  Real_ptr tcoeff = &coeff_array[0]; \
  allocData(DataSpace::CudaDevice, coeff, FIR_COEFFLEN); \
  copyData(DataSpace::CudaDevice, coeff, DataSpace::Host, tcoeff, FIR_COEFFLEN);


#define FIR_DATA_TEARDOWN_CUDA \
  deallocData(DataSpace::CudaDevice, coeff);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void fir(Real_ptr out, Real_ptr in,
                    Real_ptr coeff,
                    const Index_type coefflen,
                    Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     FIR_BODY;
   }
}

#endif


template < size_t block_size >
void FIR::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  FIR_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    FIR_COEFF;

    FIR_DATA_SETUP_CUDA;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("FIR_1");
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

#if defined(USE_CUDA_CONSTANT_MEMORY)
      RPlaunchCudaKernel( (fir<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          out, in,
                          coefflen,
                          iend ); 
#else
      RPlaunchCudaKernel( (fir<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          out, in,
                          coeff,
                          coefflen,
                          iend );
#endif
      RP_CALI_SUBKERNEL_END("FIR_1");

    }
    stopTimer();

    FIR_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    FIR_COEFF;

    FIR_DATA_SETUP_CUDA;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("FIR_1");
       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         FIR_BODY;
       });
       RP_CALI_SUBKERNEL_END("FIR_1");

    }
    stopTimer();

    FIR_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  FIR : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(FIR, Cuda, Base_CUDA, RAJA_CUDA)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
