//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>
#include <utility>
#include <type_traits>
#include <limits>


namespace rajaperf
{
namespace basic
{


template < size_t work_group_size >
void PI_REDUCE::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  PI_REDUCE_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    Real_ptr pi;
    allocAndInitSyclDeviceData(pi, &m_pi_init, 1, qu);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

      initSyclDeviceData(pi, &m_pi_init, 1, qu);

      qu->submit([&] (sycl::handler& hdl) {

        auto sum_reduction = sycl::reduction(pi, sycl::plus<>());

        hdl.parallel_for(sycl::nd_range<1>(global_size, work_group_size),
                         sum_reduction,
                         [=] (sycl::nd_item<1> item, auto& pi) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            PI_REDUCE_BODY;
          }

        });
      });

      Real_type lpi;
      Real_ptr plpi = &lpi;
      getSyclDeviceData(plpi, pi, 1, qu);
      m_pi = 4.0 * lpi;

    }
    stopTimer();

    deallocSyclDeviceData(pi, qu);

  } else if ( vid == RAJA_SYCL ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type tpi = m_pi_init;

      RAJA::forall< RAJA::sycl_exec<work_group_size, false /*async*/> >(
        res,
        RAJA::RangeSegment(ibegin, iend),
        RAJA::expt::Reduce<RAJA::operators::plus>(&tpi),
        [=] (Index_type i,
          RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& pi) {
          PI_REDUCE_BODY;
        }
      );

      m_pi = static_cast<Real_type>(tpi) * 4.0;

    }
    stopTimer();

  } else {
     getCout() << "\n  PI_REDUCE : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(PI_REDUCE, Sycl)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
