//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 256;


#define GEN_LIN_RECUR_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(b5, m_b5, m_N, qu); \
  allocAndInitSyclDeviceData(stb5, m_stb5, m_N, qu); \
  allocAndInitSyclDeviceData(sa, m_sa, m_N, qu); \
  allocAndInitSyclDeviceData(sb, m_sb, m_N, qu);

#define GEN_LIN_RECUR_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_b5, b5, m_N, qu); \
  deallocSyclDeviceData(b5, qu); \
  deallocSyclDeviceData(stb5, qu); \
  deallocSyclDeviceData(sa, qu); \
  deallocSyclDeviceData(sb, qu);

void GEN_LIN_RECUR::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  GEN_LIN_RECUR_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    GEN_LIN_RECUR_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size1 = block_size * RAJA_DIVIDE_CEILING_INT(N, block_size);
      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class GenLin1>(cl::sycl::nd_range<1> (grid_size1, block_size),
                                      [=] (cl::sycl::nd_item<1> item) {

          Index_type k = item.get_global_id(0);
          if (k < N) {
            GEN_LIN_RECUR_BODY1;
          }
 
        });
      });

      const size_t grid_size2 = block_size * RAJA_DIVIDE_CEILING_INT(N+1, block_size);
      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class GenLin2>(cl::sycl::nd_range<1> (grid_size2, block_size),
                                      [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i > 0 && i < N+1) {
            GEN_LIN_RECUR_BODY2;
          }

        });
      });
    }
    qu.wait();
    stopTimer();

    GEN_LIN_RECUR_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    GEN_LIN_RECUR_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::sycl_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(0, N), [=] (Index_type k) {
         GEN_LIN_RECUR_BODY1;
       });

       RAJA::forall< RAJA::sycl_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(1, N+1), [=] (Index_type i) {
         GEN_LIN_RECUR_BODY2;
       });

    }
    qu.wait();
    stopTimer();

    GEN_LIN_RECUR_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  GEN_LIN_RECUR : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
