//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_SUM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>


namespace rajaperf
{
namespace algorithm
{

template <size_t work_group_size >
void REDUCE_SUM::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    Real_ptr sum;
    allocAndInitSyclDeviceData(sum, &m_sum_init, 1, qu);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      initSyclDeviceData(sum, &m_sum_init, 1, qu); 

      qu->submit([&] (sycl::handler& h) {

        auto sumReduction = sycl::reduction(sum, sycl::plus<Real_type>());

        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size),
                       sumReduction,
                       [=] (sycl::nd_item<1> item, auto& sum) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            REDUCE_SUM_BODY;
          }

        });
      });

      Real_type lsum;
      Real_ptr plsum = &lsum;
      getSyclDeviceData(plsum, sum, 1, qu);
      m_sum = lsum;

    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       Real_type tsum = m_sum_init;

       RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >( 
         res,
         RAJA::RangeSegment(ibegin, iend), 
         RAJA::expt::Reduce<RAJA::operators::plus>(&tsum),
         [=]  (Index_type i,
           RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& sum) {
           REDUCE_SUM_BODY;
         }
       );

       m_sum = static_cast<Real_type>(tsum);

    }
    stopTimer();

  } else {
     std::cout << "\n  REDUCE_SUM : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(REDUCE_SUM, Sycl)

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
