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

#include "LTIMES.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace apps
{

#define LTIMES_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(phidat, m_phidat, m_philen, qu); \
  allocAndInitSyclDeviceData(elldat, m_elldat, m_elllen, qu); \
  allocAndInitSyclDeviceData(psidat, m_psidat, m_psilen, qu);

#define LTIMES_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_phidat, phidat, m_philen, qu); \
  deallocSyclDeviceData(phidat, qu); \
  deallocSyclDeviceData(elldat, qu); \
  deallocSyclDeviceData(psidat, qu);

void LTIMES::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  LTIMES_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    LTIMES_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      qu->submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class LTIMES>(cl::sycl::nd_range<3> (
                                       cl::sycl::range<3>(num_z, num_g, num_m),
                                       cl::sycl::range<3>(1,1,1)),
                                     [=] (cl::sycl::nd_item<3> item) {

          Index_type z = item.get_global_id(0);
          Index_type g = item.get_global_id(1);
          Index_type m = item.get_global_id(2);

          for (Index_type d = 0; d < num_d; ++d) {
            LTIMES_BODY
          }
        });
      });
    }
    qu->wait(); // Wait for computation to finish before stopping timer      
    stopTimer();

    LTIMES_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    LTIMES_DATA_SETUP_SYCL;

    LTIMES_VIEWS_RANGES_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernel<
          RAJA::statement::For<1, RAJA::sycl_global_1<1>,      //z 
            RAJA::statement::For<2, RAJA::sycl_global_2<1>,    //g
              RAJA::statement::For<3, RAJA::sycl_global_3<1>, //m
                RAJA::statement::For<0, RAJA::seq_exec,       //d
                  RAJA::statement::Lambda<0>
                >
              >
            >
          >
        >
      >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXEC_POL>( RAJA::make_tuple(IDRange(0, num_d),
                                                 IZRange(0, num_z),
                                                 IGRange(0, num_g),
                                                 IMRange(0, num_m)),
          [=] (ID d, IZ z, IG g, IM m) {
          LTIMES_BODY_RAJA;
        });

      }
      qu->wait();
      stopTimer();

      LTIMES_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n LTIMES : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
