//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRAP_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "TRAP_INT-func.hpp"

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


template <size_t work_group_size >
void TRAP_INT::runSyclVariantImpl(VariantID vid)
{
  setBlockSize(work_group_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  TRAP_INT_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    Real_ptr sumx;
    allocAndInitSyclDeviceData(sumx, &m_sumx_init, 1, qu);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("TRAP_INT_1");
      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      initSyclDeviceData(sumx, &m_sumx_init, 1, qu);
  
      qu.submit([&] (sycl::handler& hdl) {

        auto sum_reduction = sycl::reduction(sumx, sycl::plus<>());

        hdl.parallel_for(sycl::nd_range<1>(global_size, work_group_size),
                       sum_reduction,
                       [=] (sycl::nd_item<1> item, auto& sumx) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            TRAP_INT_BODY
          }

        });
      });

      Real_type lsumx;
      Real_ptr plsumx = &lsumx;
      getSyclDeviceData(plsumx, sumx, 1, qu);
      m_sumx += lsumx * h;
      RP_CALI_SUBKERNEL_END("TRAP_INT_1");

    }
    stopTimer();
  
    deallocSyclDeviceData(sumx, qu);

  } else if ( vid == RAJA_SYCL ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("TRAP_INT_1");
      Real_type tsumx = m_sumx_init;

      RAJA::forall< RAJA::sycl_exec<work_group_size, false /*async*/> >(
        res,
        RAJA::RangeSegment(ibegin, iend),
        RAJA::expt::Reduce<RAJA::operators::plus>(&tsumx),
        [=] (Index_type i,
          RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& sumx) {
          TRAP_INT_BODY;
        }
      );

      m_sumx += static_cast<Real_type>(tsumx) * h;
      RP_CALI_SUBKERNEL_END("TRAP_INT_1");

    }
    stopTimer();

  } else {
     std::cout << "\n  TRAP_INT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(TRAP_INT, Sycl, Base_SYCL, RAJA_SYCL)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
