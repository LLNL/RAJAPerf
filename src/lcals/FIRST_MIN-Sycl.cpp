//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_MIN.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>


namespace rajaperf
{
namespace lcals
{

template <size_t work_group_size >
void FIRST_MIN::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  FIRST_MIN_DATA_SETUP;

  if ( vid == Base_SYCL ) {

#if 0 // RDH
    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      define and init reduction...

      qu->submit([&] (sycl::handler& h) {

        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size),
                       pass reduction...,
                       [=] (sycl::nd_item<1> item, auto& dot) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            // body
          }

        });
      });

      m_minloc = get loc value... 

    }
    stopTimer();
#endif

  } else if ( vid == RAJA_SYCL ) {

    using VL_TYPE = RAJA::expt::ValLoc<Real_type>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       VL_TYPE tloc(m_xmin_init, m_initloc);

       RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >(
         res,
         RAJA::RangeSegment(ibegin, iend), 
         RAJA::expt::Reduce<RAJA::operators::minimum>(&tloc),
         [=]  (Index_type i, VL_TYPE& loc) {
           loc.min(x[i], i);
         }
       );

       m_minloc = static_cast<Index_type>(tloc.getLoc());

    }
    stopTimer();

  } else {
     std::cout << "\n  FIRST_MIN : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(FIRST_MIN, Sycl)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
