  
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read RAJAPerf/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_2MM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>
#include <cmath>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace polybench
{

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 16;

#define POLYBENCH_2MM_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(tmp, m_tmp, m_ni * m_nj, qu); \
  allocAndInitSyclDeviceData(A, m_A, m_ni * m_nk, qu); \
  allocAndInitSyclDeviceData(B, m_B, m_nk * m_nj, qu); \
  allocAndInitSyclDeviceData(C, m_C, m_nj * m_nl, qu); \
  allocAndInitSyclDeviceData(D, m_D, m_ni * m_nl, qu); \
\
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \

/*  cl::sycl::buffer<Real_type> d_tmp {m_tmp, m_ni * m_nj}; \
  cl::sycl::buffer<Real_type> d_A {m_A, m_ni * m_nk}; \
  cl::sycl::buffer<Real_type> d_B {m_B, m_nk * m_nj}; \
  cl::sycl::buffer<Real_type> d_C {m_C, m_nj * m_nl}; \
  cl::sycl::buffer<Real_type> d_D {m_D, m_ni * m_nl}; \*/
\

#define POLYBENCH_2MM_TEARDOWN_SYCL \
  getSyclDeviceData(m_D, D, m_ni * m_nl, qu); \
  deallocSyclDeviceData(tmp, qu); \
  deallocSyclDeviceData(A, qu); \
  deallocSyclDeviceData(B, qu); \
  deallocSyclDeviceData(C, qu); \
  deallocSyclDeviceData(D, qu);

void POLYBENCH_2MM::runSyclVariant(VariantID vid)
{
  const unsigned long run_reps = getRunReps();

  POLYBENCH_2MM_DATA_SETUP;

  if ( vid == Base_SYCL ) {
    {
      POLYBENCH_2MM_DATA_SETUP_SYCL;

      const size_t ni_grid_size = block_size * RAJA_DIVIDE_CEILING_INT(m_ni, block_size);
      const size_t nj_grid_size = block_size * RAJA_DIVIDE_CEILING_INT(m_nj, block_size);
      const size_t nl_grid_size = block_size * RAJA_DIVIDE_CEILING_INT(m_nl, block_size);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        qu->submit([&] (cl::sycl::handler& h)
        {

          h.parallel_for<class polybench2MM_1>(cl::sycl::nd_range<2> 
                                                 {cl::sycl::range<2> {ni_grid_size, nj_grid_size},
                                                  cl::sycl::range<2> {block_size, block_size}},
                                               [=] (cl::sycl::nd_item<2> item) {

           Index_type i = item.get_global_id(0); //get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
           Index_type j = item.get_global_id(1); //roup(1) * item.get_local_range().get(1) + item.get_local_id(1);

           if (i < ni && j < nj) {
             POLYBENCH_2MM_BODY1;
             for (Index_type k=0; k < nk; ++k) {
                POLYBENCH_2MM_BODY2;
              }
              POLYBENCH_2MM_BODY3;
            }
          });
        });

        qu->submit([&] (cl::sycl::handler& h)
        {

          h.parallel_for<class polybench2MM_2>(cl::sycl::nd_range<2>
                                                 {cl::sycl::range<2> {ni_grid_size, nl_grid_size},
                                                  cl::sycl::range<2> {block_size, block_size}},
                                               [=] (cl::sycl::nd_item<2> item) {

           Index_type i = item.get_global_id(0); //group(0) * item.get_local_range().get(0) + item.get_local_id(0);
           Index_type l = item.get_global_id(1); //roup(1) * item.get_local_range().get(1) + item.get_local_id(1);

           if(i < ni && l < nl) {        
             POLYBENCH_2MM_BODY4;
             for (Index_type j=0; j < nj; ++j) {
                POLYBENCH_2MM_BODY5;
              }
              POLYBENCH_2MM_BODY6;
            }
          });
        });
      }
      qu->wait(); // Wait for computation to finish before stopping timer
      stopTimer();
    }

    POLYBENCH_2MM_TEARDOWN_SYCL;

  } else if (vid == RAJA_SYCL) {
    
    POLYBENCH_2MM_DATA_SETUP_SYCL;

    POLYBENCH_2MM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernelNonTrivial<
          RAJA::statement::For<0, RAJA::sycl_global_1<1>,
            RAJA::statement::For<1, RAJA::sycl_global_0<256>,
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

    POLYBENCH_2MM_TEARDOWN_SYCL;
  
  } else {
      std::cout << "\n  POLYBENCH_2MM : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_Sycl
  
