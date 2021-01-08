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

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace apps
{

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 256;


#define ENERGY_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(e_new, m_e_new, iend, qu); \
  allocAndInitSyclDeviceData(e_old, m_e_old, iend, qu); \
  allocAndInitSyclDeviceData(delvc, m_delvc, iend, qu); \
  allocAndInitSyclDeviceData(p_new, m_p_new, iend, qu); \
  allocAndInitSyclDeviceData(p_old, m_p_old, iend, qu); \
  allocAndInitSyclDeviceData(q_new, m_q_new, iend, qu); \
  allocAndInitSyclDeviceData(q_old, m_q_old, iend, qu); \
  allocAndInitSyclDeviceData(work, m_work, iend, qu); \
  allocAndInitSyclDeviceData(compHalfStep, m_compHalfStep, iend, qu); \
  allocAndInitSyclDeviceData(pHalfStep, m_pHalfStep, iend, qu); \
  allocAndInitSyclDeviceData(bvc, m_bvc, iend, qu); \
  allocAndInitSyclDeviceData(pbvc, m_pbvc, iend, qu); \
  allocAndInitSyclDeviceData(ql_old, m_ql_old, iend, qu); \
  allocAndInitSyclDeviceData(qq_old, m_qq_old, iend, qu); \
  allocAndInitSyclDeviceData(vnewc, m_vnewc, iend, qu);

#define ENERGY_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_e_new, e_new, iend, qu); \
  getSyclDeviceData(m_q_new, q_new, iend, qu); \
  deallocSyclDeviceData(e_new, qu); \
  deallocSyclDeviceData(e_old, qu); \
  deallocSyclDeviceData(delvc, qu); \
  deallocSyclDeviceData(p_new, qu); \
  deallocSyclDeviceData(p_old, qu); \
  deallocSyclDeviceData(q_new, qu); \
  deallocSyclDeviceData(q_old, qu); \
  deallocSyclDeviceData(work, qu); \
  deallocSyclDeviceData(compHalfStep, qu); \
  deallocSyclDeviceData(pHalfStep, qu); \
  deallocSyclDeviceData(bvc, qu); \
  deallocSyclDeviceData(pbvc, qu); \
  deallocSyclDeviceData(ql_old, qu); \
  deallocSyclDeviceData(qq_old, qu); \
  deallocSyclDeviceData(vnewc, qu);

void ENERGY::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  ENERGY_DATA_SETUP;

  using cl::sycl::sqrt;
  using cl::sycl::fabs;

  if ( vid == Base_SYCL ) {

    ENERGY_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size); 

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class Energy_1>(cl::sycl::nd_range<1> (grid_size, block_size),
                                       [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if(i < iend) {
            ENERGY_BODY1
          }

        });
      });

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class Energy_2>(cl::sycl::nd_range<1> (grid_size, block_size),
                                       [=] (cl::sycl::nd_item<1> item) {
            
          Index_type i = item.get_global_id(0);            
          if(i < iend) {
            ENERGY_BODY2
          }

        });
      });

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class Energy_3>(cl::sycl::nd_range<1> (grid_size, block_size),
                                       [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if(i < iend) {
            ENERGY_BODY3
          }
        });
      });

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class Energy_4>(cl::sycl::nd_range<1> (grid_size, block_size),
                                       [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if(i < iend) {
            ENERGY_BODY4
          }

        });
      });

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class Energy_5>(cl::sycl::nd_range<1> (grid_size, block_size),
                                       [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if(i < iend) {
            ENERGY_BODY5
          }

        });
      });

      qu.submit([&] (cl::sycl::handler& h)
      {
        h.parallel_for<class Energy_6>(cl::sycl::nd_range<1> (grid_size, block_size),
                                       [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if(i < iend) {
            ENERGY_BODY6
          }

        });
      });
    }
    qu.wait(); // Wait for computation to finish before stopping timer
    stopTimer();

    ENERGY_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    ENERGY_DATA_SETUP_SYCL;

    const bool async = true;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RAJA::forall< RAJA::sycl_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY1;
        });

        RAJA::forall< RAJA::sycl_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY2;
        });

        RAJA::forall< RAJA::sycl_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY3;
        });

        RAJA::forall< RAJA::sycl_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY4;
        });

        RAJA::forall< RAJA::sycl_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY5;
        });

        RAJA::forall< RAJA::sycl_exec<block_size, async> >(
          RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
          ENERGY_BODY6;
        });

      }); // end sequential region (for single-source code)

    }
    qu.wait();
    stopTimer();

    ENERGY_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  ENERGY : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
