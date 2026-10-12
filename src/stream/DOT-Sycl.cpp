//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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
  setBlockSize(work_group_size);

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
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DOT_1");
      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      initSyclDeviceData(dot, &m_dot_init, 1, qu); 

      qu.submit([&] (sycl::handler& h) {

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
      RP_CALI_SUBKERNEL_END("DOT_1");

    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("DOT_1");
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
       RP_CALI_SUBKERNEL_END("DOT_1");

    }
    stopTimer();

  } else {
     std::cout << "\n  DOT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(DOT, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
