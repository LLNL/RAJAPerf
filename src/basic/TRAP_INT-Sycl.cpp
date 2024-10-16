//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      initSyclDeviceData(sumx, &m_sumx_init, 1, qu);
  
      qu->submit([&] (sycl::handler& hdl) {

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

    }
    stopTimer();
  
    deallocSyclDeviceData(sumx, qu);

  } else if ( vid == RAJA_SYCL ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

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

    }
    stopTimer();

  } else {
     std::cout << "\n  TRAP_INT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(TRAP_INT, Sycl)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
