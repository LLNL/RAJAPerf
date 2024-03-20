//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DOT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>


namespace rajaperf
{
namespace stream
{

template <size_t work_group_size >
void DOT::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DOT_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    if (work_group_size != 0) {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        Real_type dot = m_dot_init;
 
        {
          sycl::buffer<Real_type, 1> buf_dot(&dot, 1);

          const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

          qu->submit([&] (sycl::handler& h) {

            auto sumReduction = reduction(buf_dot, h, sycl::plus<Real_type>());

            h.parallel_for(sycl::nd_range<1>{global_size, work_group_size},
                           sumReduction,
                           [=] (sycl::nd_item<1> item, auto& dot) {

              Index_type i = item.get_global_id(0);
              if (i < iend) {
                DOT_BODY;
              }

            });
          });
        }

        m_dot += dot;

      }
      stopTimer();
    }

  } else if ( vid == RAJA_SYCL ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::ReduceSum<RAJA::sycl_reduce, Real_type> dot(m_dot_init);

       RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=]  (Index_type i) {
         DOT_BODY;
       });

       m_dot += static_cast<Real_type>(dot.get());

    }
    stopTimer();

  } else {
     std::cout << "\n  DOT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(DOT, Sycl)

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
