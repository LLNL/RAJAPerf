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

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace lcals
{

#define HYDRO_2D_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(zadat, m_za, m_array_length, qu); \
  allocAndInitSyclDeviceData(zbdat, m_zb, m_array_length, qu); \
  allocAndInitSyclDeviceData(zmdat, m_zm, m_array_length, qu); \
  allocAndInitSyclDeviceData(zpdat, m_zp, m_array_length, qu); \
  allocAndInitSyclDeviceData(zqdat, m_zq, m_array_length, qu); \
  allocAndInitSyclDeviceData(zrdat, m_zr, m_array_length, qu); \
  allocAndInitSyclDeviceData(zudat, m_zu, m_array_length, qu); \
  allocAndInitSyclDeviceData(zvdat, m_zv, m_array_length, qu); \
  allocAndInitSyclDeviceData(zzdat, m_zz, m_array_length, qu); \
  allocAndInitSyclDeviceData(zroutdat, m_zrout, m_array_length, qu); \
  allocAndInitSyclDeviceData(zzoutdat, m_zzout, m_array_length, qu);

#define HYDRO_2D_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_zrout, zroutdat, m_array_length, qu); \
  getSyclDeviceData(m_zzout, zzoutdat, m_array_length, qu); \
  deallocSyclDeviceData(zadat, qu); \
  deallocSyclDeviceData(zbdat, qu); \
  deallocSyclDeviceData(zmdat, qu); \
  deallocSyclDeviceData(zpdat, qu); \
  deallocSyclDeviceData(zqdat, qu); \
  deallocSyclDeviceData(zrdat, qu); \
  deallocSyclDeviceData(zudat, qu); \
  deallocSyclDeviceData(zvdat, qu); \
  deallocSyclDeviceData(zzdat, qu); \
  deallocSyclDeviceData(zroutdat, qu); \
  deallocSyclDeviceData(zzoutdat, qu);

void HYDRO_2D::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  HYDRO_2D_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    HYDRO_2D_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      qu->submit([&] (cl::sycl::handler& h) { 
        h.parallel_for<class syclHydro2dBody1>(cl::sycl::range<2>(kn-2, jn-2),
                                               cl::sycl::id<2>(1, 1), // offset to start a idx 1
                                               [=] (cl::sycl::item<2> item ) {
          int j = item.get_id(1);
          int k = item.get_id(0); 
          HYDRO_2D_BODY1

        });
      });

      qu->submit([&] (cl::sycl::handler& h) { 
        h.parallel_for<class syclHydro2dBody2>(cl::sycl::range<2>(kn-2, jn-2),
                                               cl::sycl::id<2>(1, 1), // offset to start a idx 1
                                               [=] (cl::sycl::item<2> item ) {
          int j = item.get_id(1);
          int k = item.get_id(0);
          HYDRO_2D_BODY2

        });
      });

      qu->submit([&] (cl::sycl::handler& h) { 
        h.parallel_for<class syclHydro2dBody3>(cl::sycl::range<2>(kn-2, jn-2),
                                               cl::sycl::id<2>(1, 1), // offset to start a idx 1
                                               [=] (cl::sycl::item<2> item ) {
          int j = item.get_id(1);
          int k = item.get_id(0);
          HYDRO_2D_BODY3

        });
      });

    }
    qu->wait(); // Wait for computation to finish before stopping timer
    stopTimer();

    HYDRO_2D_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    HYDRO_2D_DATA_SETUP_SYCL;

    HYDRO_2D_VIEWS_RAJA;

      using EXECPOL =
        RAJA::KernelPolicy<
          RAJA::statement::SyclKernel<
            RAJA::statement::For<0, RAJA::sycl_global_1<1>,  // k
              RAJA::statement::For<1, RAJA::sycl_global_0<256>,  // j
                RAJA::statement::Lambda<0>
              >
            >
          >
        >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY1_RAJA;
      });

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY2_RAJA;
      });

      RAJA::kernel<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        [=] (Index_type k, Index_type j) {
        HYDRO_2D_BODY3_RAJA;
      });

    }
    qu->wait();
    stopTimer();

    HYDRO_2D_DATA_TEARDOWN_SYCL;

  } else { 
     std::cout << "\n  HYDRO_2D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
