//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "IF_QUAD.hpp"

#include "RAJA/RAJA.hpp"

//#if defined(RAJA_ENABLE_CUDA)

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


#define IF_QUAD_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(a, m_a, iend); \
  allocAndInitCudaDeviceData(b, m_b, iend); \
  allocAndInitCudaDeviceData(c, m_c, iend); \
  allocAndInitCudaDeviceData(x1, m_x1, iend); \
  allocAndInitCudaDeviceData(x2, m_x2, iend);

#define IF_QUAD_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_x1, x1, iend); \
  getCudaDeviceData(m_x2, x2, iend); \
  deallocCudaDeviceData(a); \
  deallocCudaDeviceData(b); \
  deallocCudaDeviceData(c); \
  deallocCudaDeviceData(x1); \
  deallocCudaDeviceData(x2);

// AJP started Kokkos-ifying here
void IF_QUAD::runKokkosCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  IF_QUAD_DATA_SETUP;

#if defined(RUN_KOKKOS)

  if ( vid == Base_CUDA ) {

#if defined(RUN_CUDA)

    IF_QUAD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

	// QUESTION: Should "RAJA_DIVIDE_CEILING_INT be changed?
       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       //ifquad<<<grid_size, block_size>>>( x1, x2, a, b, c,
       //                                   iend );

    }
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_CUDA;

#endif // RUN_CUDA

  } else if ( vid == Kokkos_Lambda_CUDA ) {
//  } else if ( vid == RAJA_CUDA ) {

    IF_QUAD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

//       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
//         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
//         IF_QUAD_BODY;
//       });


  Kokkos::parallel_for("Quad Cuda", Kokkos::RangePolicy<Kokkos::Cuda>(ibegin, iend),
	// Here, the function executes on the device / GPU
  	[=] __device__ (Index_type i) {IF_QUAD_BODY});
  	//KOKKOS_LAMBDA (Index_type i) {IF_QUAD_BODY});


//<block_size, true /*async*/> >(
//         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
//         IF_QUAD_BODY;
//       });


    }
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  IF_QUAD : Unknown Cuda variant id = " << vid << std::endl;
  }


#endif  // RUN_KOKKOS
}

} // end namespace basic
} // end namespace rajaperf

//#endif  // RAJA_ENABLE_CUDA
