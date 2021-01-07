//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define INIT_VIEW1D_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(a, m_a, getRunSize());

#define INIT_VIEW1D_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_a, a, getRunSize()); \
  deallocCudaDeviceData(a);

//__global__ void initview1d(Real_ptr a, 
//                           Real_type v,
//                           const Index_type iend) 
//{
//   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < iend) {
//     INIT_VIEW1D_BODY; 
//   }
//}


void INIT_VIEW1D::runKokkosCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  INIT_VIEW1D_DATA_SETUP;

#if defined(RUN_KOKKOS)

  if ( vid == Base_CUDA ) {

    INIT_VIEW1D_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       //initview1d<<<grid_size, block_size>>>( a,
       //                                       v, 
       //                                       iend ); 

    }
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_CUDA;

// AJP modified lines below 
  } else if ( vid == Kokkos_Lambda_CUDA ) {

    INIT_VIEW1D_DATA_SETUP_CUDA;

    INIT_VIEW1D_VIEW_RAJA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

//      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
//        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
//        INIT_VIEW1D_BODY_RAJA;
//      });

	Kokkos::parallel_for("INIT_VIEW1D-KokkosCuda Kokkos-Lambda", Kokkos::RangePolicy<Kokkos::Cuda>(ibegin, iend),
		// Here, the function executes on the device / GPU
		[=] __device__ (Index_type i) {INIT_VIEW1D_BODY_RAJA});

    }
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  INIT_VIEW1D : Unknown Cuda variant id = " << vid << std::endl;
  }
#endif //RUN_KOKKOS
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
