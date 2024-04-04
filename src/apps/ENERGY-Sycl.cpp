//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace apps
{

template <size_t work_group_size >
void ENERGY::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  ENERGY_DATA_SETUP;

  using sycl::sqrt;
  using sycl::fabs;

  if ( vid == Base_SYCL ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size); 

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (grid_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if(i < iend) {
            ENERGY_BODY1
          }

        });
      });

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (grid_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {
            
          Index_type i = item.get_global_id(0);            
          if(i < iend) {
            ENERGY_BODY2
          }

        });
      });

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (grid_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if(i < iend) {
            ENERGY_BODY3
          }
        });
      });

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (grid_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if(i < iend) {
            ENERGY_BODY4
          }

        });
      });

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (grid_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if(i < iend) {
            ENERGY_BODY5
          }

        });
      });

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1> (grid_size, work_group_size),
                       [=] (sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if(i < iend) {
            ENERGY_BODY6
          }

        });
      });
    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    const bool async = true;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RAJA::forall< RAJA::sycl_exec<work_group_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY1;
        });

        RAJA::forall< RAJA::sycl_exec<work_group_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY2;
        });

        RAJA::forall< RAJA::sycl_exec<work_group_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY3;
        });

        RAJA::forall< RAJA::sycl_exec<work_group_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY4;
        });

        RAJA::forall< RAJA::sycl_exec<work_group_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY5;
        });

        RAJA::forall< RAJA::sycl_exec<work_group_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY6;
        });

      }); // end sequential region (for single-source code)

    }
    stopTimer();

  } else {
     std::cout << "\n  ENERGY : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(ENERGY, Sycl)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
