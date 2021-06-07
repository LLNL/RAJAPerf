//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INDEXLIST.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for HIP execution
  //
  const size_t block_size = 256;


#define INDEXLIST_DATA_SETUP_HIP \
  Index_type* counts; \
  allocHipDeviceData(counts, getRunSize()+1); \
  allocAndInitHipDeviceData(x, m_x, iend); \
  allocAndInitHipDeviceData(list, m_list, iend);

#define INDEXLIST_DATA_TEARDOWN_HIP \
  deallocHipDeviceData(counts); \
  getHipDeviceData(m_list, list, iend); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(list);


void INDEXLIST::runHipVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  INDEXLIST_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    INDEXLIST_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::hip_reduce, Index_type> len(0);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend),
        [=] __device__ (Index_type i) {
        counts[i] = (INDEXLIST_CONDITIONAL) ? 1 : 0;
      });

      RAJA::exclusive_scan_inplace< RAJA::hip_exec<block_size, true /*async*/> >(
          RAJA::make_span(counts+ibegin, iend+1-ibegin));

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend),
        [=] __device__ (Index_type i) {
        if (counts[i] != counts[i+1]) {
          list[counts[i]] = i;
          len += 1;
        }
      });

      m_len = len.get();

    }
    stopTimer();

    INDEXLIST_DATA_TEARDOWN_HIP;

  } else {
    std::cout << "\n  INDEXLIST : Unknown variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
