//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
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

template <typename VAL_TYPE, typename IDX_TYPE>
struct reduce_pair {
  bool operator<(const reduce_pair& o) const {
    return (val < o.val);
  }
  VAL_TYPE val;
  IDX_TYPE idx;
};

template <size_t work_group_size >
void FIRST_MIN::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  FIRST_MIN_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    using result_type = reduce_pair<Real_type, Index_type>;

    auto result = sycl::malloc_shared< result_type >(1, qu); 
 
    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("FIRST_MIN_1");
      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      result_type result_init = { m_xmin_init, m_initloc };
      *result = result_init;
      auto reduction_obj = sycl::reduction( result, result_init, sycl::minimum<result_type>() ); 

      qu.submit([&] (sycl::handler& h) {

        h.parallel_for(sycl::nd_range<1>(global_size, work_group_size),
                       reduction_obj,
                       [=] (sycl::nd_item<1> item, auto& loc) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            loc.combine( {x[i], i} );
          }

        });

      });

      qu.wait();

      m_minloc = static_cast<Index_type>(result->idx);
      RP_CALI_SUBKERNEL_END("FIRST_MIN_1");

    }
    stopTimer();

    sycl::free(result, qu);

  } else if ( vid == RAJA_SYCL ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("FIRST_MIN_1");
       RAJA::expt::ValLoc<Real_type, Index_type> tminloc(m_xmin_init,
                                                         m_initloc);

       RAJA::forall< RAJA::sycl_exec<work_group_size, true /*async*/> >(
         res,
         RAJA::RangeSegment(ibegin, iend), 
         RAJA::expt::Reduce<RAJA::operators::minimum>(&tminloc),
         [=]  (Index_type i,
           RAJA::expt::ValLocOp<Real_type, Index_type,
                                 RAJA::operators::minimum>& minloc) {
           FIRST_MIN_BODY_RAJA;
         }
       );

       m_minloc = static_cast<Index_type>(tminloc.getLoc());
       RP_CALI_SUBKERNEL_END("FIRST_MIN_1");

    }
    stopTimer();

  } else {
     std::cout << "\n  FIRST_MIN : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(FIRST_MIN, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
