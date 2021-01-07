//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRAP_INT.hpp"

#include "RAJA/RAJA.hpp"
#include <Kokkos_Core.hpp>
#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

//
// Function used in TRAP_INT loop.
//
KOKKOS_INLINE_FUNCTION
Real_type trap_int_func(Real_type x,
                        Real_type y,
                        Real_type xp,
                        Real_type yp)
{
   Real_type denom = (x - xp)*(x - xp) + (y - yp)*(y - yp);
   denom = 1.0/sqrt(denom);
   return denom;
}


  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define TRAP_INT_DATA_SETUP_CUDA  // nothing to do here...

#define TRAP_INT_DATA_TEARDOWN_CUDA // nothing to do here...


//__global__ void trapint(Real_type x0, Real_type xp,
//                        Real_type y, Real_type yp, 
//                        Real_type h, 
//                        Real_ptr sumx,
//                        Index_type iend)
//{
//  extern __shared__ Real_type psumx[ ];
//
//  Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
//
//  psumx[ threadIdx.x ] = 0.0;
//  for ( ; i < iend ; i += gridDim.x * blockDim.x ) {
//    Real_type x = x0 + i*h;
//    Real_type val = trap_int_func(x, y, xp, yp);
//    psumx[ threadIdx.x ] += val;
//  }
//  __syncthreads();
//
//  for ( i = blockDim.x / 2; i > 0; i /= 2 ) {
//    if ( threadIdx.x < i ) {
//      psumx[ threadIdx.x ] += psumx[ threadIdx.x + i ];
//    }
//     __syncthreads();
//  }
//
//#if 1 // serialized access to shared data;
//  if ( threadIdx.x == 0 ) {
//    RAJA::atomicAdd<RAJA::cuda_atomic>( sumx, psumx[ 0 ] );
//  }
//#else // this doesn't work due to data races
//  if ( threadIdx.x == 0 ) {
//    *sumx += psumx[ 0 ];
//  }
//#endif
//
//}


void TRAP_INT::runKokkosCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  TRAP_INT_DATA_SETUP;

#if defined RUN_KOKKOS

  if ( vid == Base_CUDA ) {

    TRAP_INT_DATA_SETUP_CUDA;

    Real_ptr sumx;
    allocAndInitCudaDeviceData(sumx, &m_sumx_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initCudaDeviceData(sumx, &m_sumx_init, 1); 

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      //trapint<<<grid_size, block_size, 
      //          sizeof(Real_type)*block_size>>>(x0, xp,
      //                                          y, yp,
      //                                          h,
      //                                          sumx,
      //                                          iend);

      Real_type lsumx;
      Real_ptr plsumx = &lsumx;
      getCudaDeviceData(plsumx, sumx, 1);
      m_sumx += lsumx * h;

    }
    stopTimer();

    deallocCudaDeviceData(sumx);

    TRAP_INT_DATA_TEARDOWN_CUDA;

  } else if ( vid == Kokkos_Lambda_CUDA ) {

    TRAP_INT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {


		//	Begin Kokkos translation
		//	A RAJA reduce translates into a
		//	Kokkoss::parallel_reduce
		//	To perform the translation:
		//	Declare and initialize variables
		//	To perform a reduction, you need:
		//	1) an initial value;
		//	2) iterate over an iterable;
		//	3) to be able to extract the result at the end of the reduction (in this case, trap_integral_val)


		Real_type trap_integral_val = m_sumx_init;

		parallel_reduce("TRAP_INT_KokkosCuda Kokkos_Lambda_Seq",
						Kokkos::RangePolicy<Kokkos::Cuda>(ibegin, iend),
						KOKKOS_LAMBDA (const int64_t i, Real_type& sumx) {
							TRAP_INT_BODY},
						trap_integral_val
						);

        m_sumx += static_cast<Real_type>(trap_integral_val) * h;

      }
    stopTimer();

    TRAP_INT_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  TRAP_INT : Unknown Cuda variant id = " << vid << std::endl;
  }
#endif //RUN_KOKKOS
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
