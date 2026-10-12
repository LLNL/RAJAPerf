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

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <algorithm>
#include <iostream>

namespace rajaperf
{
namespace apps
{

#define USE_HIP_CONSTANT_MEMORY
// #undef USE_HIP_CONSTANT_MEMORY

#if defined(USE_HIP_CONSTANT_MEMORY)

__constant__ Real_type coeff[FIR_COEFFLEN];

#define FIR_DATA_SETUP_HIP \
  CAMP_HIP_API_INVOKE_AND_CHECK( \
      hipMemcpyToSymbolAsync, HIP_SYMBOL(coeff), \
      coeff_array, FIR_COEFFLEN * sizeof(Real_type), \
      0, hipMemcpyHostToDevice, res.get_stream() );


#define FIR_DATA_TEARDOWN_HIP

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

#else  // use global memry for coefficients

#define FIR_DATA_SETUP_HIP \
  Real_ptr coeff; \
  \
  Real_ptr tcoeff = &coeff_array[0]; \
  allocData(DataSpace::HipDevice, coeff, FIR_COEFFLEN); \
  copyData(DataSpace::HipDevice, coeff, DataSpace::Host, tcoeff, FIR_COEFFLEN);


#define FIR_DATA_TEARDOWN_HIP \
  deallocData(DataSpace::HipDevice, coeff);

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
void FIR::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  FIR_DATA_SETUP;

  if ( vid == Base_HIP ) {

    FIR_COEFF;

    FIR_DATA_SETUP_HIP;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("FIR_1");
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

#if defined(USE_HIP_CONSTANT_MEMORY)
      RPlaunchHipKernel( (fir<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         out, in,
                         coefflen,
                         iend ); 
#else
      RPlaunchHipKernel( (fir<block_size>),
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

    FIR_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    FIR_COEFF;

    FIR_DATA_SETUP_HIP;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("FIR_1");
       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         FIR_BODY;
       });
       RP_CALI_SUBKERNEL_END("FIR_1");

    }
    stopTimer();

    FIR_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  FIR : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(FIR, Hip, Base_HIP, RAJA_HIP)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
