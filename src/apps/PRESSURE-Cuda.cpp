//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define PRESSURE_DATA_SETUP_CUDA \
  Real_ptr compression; \
  Real_ptr bvc; \
  Real_ptr p_new; \
  Real_ptr e_old; \
  Real_ptr vnewc; \
  const Real_type cls = m_cls; \
  const Real_type p_cut = m_p_cut; \
  const Real_type pmin = m_pmin; \
  const Real_type eosvmax = m_eosvmax; \
\
  allocAndInitCudaDeviceData(compression, m_compression, iend); \
  allocAndInitCudaDeviceData(bvc, m_bvc, iend); \
  allocAndInitCudaDeviceData(p_new, m_p_new, iend); \
  allocAndInitCudaDeviceData(e_old, m_e_old, iend); \
  allocAndInitCudaDeviceData(vnewc, m_vnewc, iend);

#define PRESSURE_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_p_new, p_new, iend); \
  deallocCudaDeviceData(compression); \
  deallocCudaDeviceData(bvc); \
  deallocCudaDeviceData(p_new); \
  deallocCudaDeviceData(e_old); \
  deallocCudaDeviceData(vnewc);

__global__ void pressurecalc1(Real_ptr bvc, Real_ptr compression,
                              const Real_type cls,
                              Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     PRESSURE_BODY1;
   }
}

__global__ void pressurecalc2(Real_ptr p_new, Real_ptr bvc, Real_ptr e_old,
                              Real_ptr vnewc,
                              const Real_type p_cut, const Real_type eosvmax,
                              const Real_type pmin,
                              Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     PRESSURE_BODY2;
   }
}


void PRESSURE::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_CUDA ) {

    PRESSURE_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

       pressurecalc1<<<grid_size, block_size>>>( bvc, compression,
                                                 cls,
                                                 iend );

       pressurecalc2<<<grid_size, block_size>>>( p_new, bvc, e_old,
                                                 vnewc,
                                                 p_cut, eosvmax, pmin,
                                                 iend );

    }
    stopTimer();

    PRESSURE_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    PRESSURE_DATA_SETUP_CUDA;

    const bool async = true;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::cuda_exec<block_size, async> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         PRESSURE_BODY1;
       });

       RAJA::forall< RAJA::cuda_exec<block_size, async> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         PRESSURE_BODY2;
       });

    }
    stopTimer();

    PRESSURE_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  PRESSURE : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
