//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read RAJAPerf/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <sycl.hpp>

#include <iostream>
#include "common/SyclDataUtils.hpp"

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 256;


#define REDUCE3_INT_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(vec, m_vec, iend, qu);

#define REDUCE3_INT_DATA_TEARDOWN_SYCL \
  deallocSyclDeviceData(vec, qu);


void REDUCE3_INT::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE3_INT_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    REDUCE3_INT_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Int_type vsum = m_vsum_init;
      Int_type vmin = m_vmin_init;
      Int_type vmax = m_vmax_init;

      {
        sycl::buffer<Int_type, 1> buf_vsum(&vsum, 1);
        sycl::buffer<Int_type, 1> buf_vmin(&vmin, 1);
        sycl::buffer<Int_type, 1> buf_vmax(&vmax, 1);

//      buf_vsum.set_final_data(vsum);
//      buf_vmin.set_final_data(vmin);
//      buf_vmax.set_final_data(vmax);

        const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

        qu->submit([&] (sycl::handler& h) {

          auto sumReduction = reduction(buf_vsum, h, sycl::plus<Int_type>());
          auto minReduction = reduction(buf_vmin, h, sycl::minimum<Int_type>());
          auto maxReduction = reduction(buf_vmax, h, sycl::maximum<Int_type>());

          h.parallel_for(sycl::nd_range<1>{grid_size, block_size},
                         sumReduction, minReduction, maxReduction,
                         [=] (sycl::nd_item<1> item, auto& vsum, auto& vmin, auto& vmax) {

           Index_type i = item.get_global_id(0);
           if (i < iend) {
             vsum += vec[i];
	     vmin.combine(vec[i]);
	     vmax.combine(vec[i]);
	   }

          });
        });
      }

      m_vsum += vsum;
      m_vmin = RAJA_MIN(m_vmin, vmin);
      m_vmax = RAJA_MAX(m_vmax, vmax);
      
    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    REDUCE3_INT_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::sycl_reduce, Int_type> vsum(m_vsum_init);
      RAJA::ReduceMin<RAJA::sycl_reduce, Int_type> vmin(m_vmin_init);
      RAJA::ReduceMax<RAJA::sycl_reduce, Int_type> vmax(m_vmax_init);

      RAJA::forall< RAJA::sycl_exec_nontrivial<block_size, false> >(
        RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
        REDUCE3_INT_BODY_RAJA;
      });

      m_vsum += static_cast<Int_type>(vsum.get());
      m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
      m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));

    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  REDUCE3_INT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL

