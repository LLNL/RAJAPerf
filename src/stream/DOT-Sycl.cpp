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

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  DOT_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    Real_ptr dot;
    allocAndInitSyclDeviceData(dot, &m_dot_init, 1, qu);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      initSyclDeviceData(dot, &m_dot_init, 1, qu); 

      qu->submit([&] (sycl::handler& h) {

        auto sumReduction = sycl::reduction(dot, sycl::plus<Real_type>());

        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size),
                       sumReduction,
                       [=] (sycl::nd_item<1> item, auto& dot) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            DOT_BODY;
          }

        });
      });

      Real_type ldot;
      Real_ptr pldot = &ldot;
      getSyclDeviceData(pldot, dot, 1, qu);
      m_dot += ldot;       

    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       Real_type tdot = m_dot_init;

       RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >( 
         res,
         RAJA::RangeSegment(ibegin, iend), 
         RAJA::expt::Reduce<RAJA::operators::plus>(&tdot),
         [=]  (Index_type i,
           RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& dot) {
           DOT_BODY;
         }
       );

       m_dot += static_cast<Real_type>(tdot);

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
