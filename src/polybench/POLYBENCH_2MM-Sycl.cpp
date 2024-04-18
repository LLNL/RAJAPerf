//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_2MM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>
#include <cmath>

namespace rajaperf 
{
namespace polybench
{

  //
  // Define work-group shape for SYCL execution
  //
#define in_wg_sz (32)
#define out_wg_sz (work_group_size / in_wg_sz)


template <size_t work_group_size >
void POLYBENCH_2MM::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getSyclResource()};
  auto qu = res.get_queue();

  POLYBENCH_2MM_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    sycl::range<2> global_dim1(out_wg_sz * RAJA_DIVIDE_CEILING_INT(ni, out_wg_sz),
                               in_wg_sz * RAJA_DIVIDE_CEILING_INT(nj, in_wg_sz));

    sycl::range<2> global_dim2(out_wg_sz * RAJA_DIVIDE_CEILING_INT(ni, out_wg_sz),
                               in_wg_sz * RAJA_DIVIDE_CEILING_INT(nl, in_wg_sz));

    sycl::range<2> wkgroup_dim(out_wg_sz, in_wg_sz);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>( global_dim1, wkgroup_dim), 
                       [=] (sycl::nd_item<2> item) {

          Index_type i = item.get_global_id(0); 
          Index_type j = item.get_global_id(1); 

          if (i < ni && j < nj) {
            POLYBENCH_2MM_BODY1;
            for (Index_type k=0; k < nk; ++k) {
              POLYBENCH_2MM_BODY2;
            }
            POLYBENCH_2MM_BODY3;
          }

        });
      });

      qu->submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>( global_dim2, wkgroup_dim),
                       [=] (sycl::nd_item<2> item) {

         Index_type i = item.get_global_id(0); 
         Index_type l = item.get_global_id(1);

         if (i < ni && l < nl) {        
           POLYBENCH_2MM_BODY4;
           for (Index_type j=0; j < nj; ++j) {
              POLYBENCH_2MM_BODY5;
           }
           POLYBENCH_2MM_BODY6;
         }

        });
      });

    }
    stopTimer();

  } else if (vid == RAJA_SYCL) {
    
    POLYBENCH_2MM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernelAsync<
          RAJA::statement::For<0, RAJA::sycl_global_0<out_wg_sz>,
            RAJA::statement::For<1, RAJA::sycl_global_1<in_wg_sz>,
              RAJA::statement::Lambda<0, RAJA::Params<0>>,
              RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<1, RAJA::Segs<0,1,2>, RAJA::Params<0>>
              >,
              RAJA::statement::Lambda<2, RAJA::Segs<0,1>, RAJA::Params<0>>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel_param<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nj},
                         RAJA::RangeSegment{0, nk}),
        RAJA::tuple<Real_type>{0.0},

        [=]  (Real_type &dot) {
          POLYBENCH_2MM_BODY1_RAJA;
        },
        [=] (Index_type i, Index_type j, Index_type k,
             Real_type &dot) {
          POLYBENCH_2MM_BODY2_RAJA;
        },
        [=] (Index_type i, Index_type j,
             Real_type &dot) {
          POLYBENCH_2MM_BODY3_RAJA;
        }
      );
 
      RAJA::kernel_param<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nl},
                         RAJA::RangeSegment{0, nj}),
        RAJA::tuple<Real_type>{0.0},

        [=]  (Real_type &dot) {
          POLYBENCH_2MM_BODY4_RAJA;
        },
        [=] (Index_type i, Index_type l, Index_type j,
             Real_type &dot) {
          POLYBENCH_2MM_BODY5_RAJA;
        },
        [=]  (Index_type i, Index_type l,
              Real_type &dot) {
          POLYBENCH_2MM_BODY6_RAJA;
        }
      );

    }
    stopTimer();

  } else {
      std::cout << "\n  POLYBENCH_2MM : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_2MM, Sycl)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
  
