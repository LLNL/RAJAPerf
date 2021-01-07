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

#include <CL/sycl.hpp>

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
  const Index_type iend = getRunSize();

  REDUCE3_INT_DATA_SETUP;

  if (0) {// vid == Base_SYCL_ ) {

/*
  } else if ( vid == RAJA_SYCL ) {

    REDUCE3_INT_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::sycl_reduce, Int_type> vsum(m_vsum_init);
      RAJA::ReduceMin<RAJA::sycl_reduce, Int_type> vmin(m_vmin_init);
      RAJA::ReduceMax<RAJA::sycl_reduce, Int_type> vmax(m_vmax_init);

      RAJA::forall< RAJA::sycl_exec<block_size, true /*async*//*> >(
        RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
        REDUCE3_INT_BODY_RAJA;
      });

      m_vsum += static_cast<Int_type>(vsum.get());
      m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
      m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));

    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_SYCL;
*/
  } else {
     std::cout << "\n  REDUCE3_INT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL

