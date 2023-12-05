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

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define REDUCE3_INT_DATA_SETUP_SYCL \
    Int_ptr hsum; \
    allocAndInitSyclDeviceData(hsum, &m_vsum_init, 1, qu); \
    Int_ptr hmin; \
    allocAndInitSyclDeviceData(hmin, &m_vmin_init, 1, qu); \
    Int_ptr hmax; \
    allocAndInitSyclDeviceData(hmax, &m_vmax_init, 1, qu);

#define REDUCE3_INT_DATA_TEARDOWN_SYCL \
  deallocSyclDeviceData(hsum, qu); \
  deallocSyclDeviceData(hmin, qu); \
  deallocSyclDeviceData(hmax, qu);

template <size_t work_group_size >
void REDUCE3_INT::runSyclVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE3_INT_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    REDUCE3_INT_DATA_SETUP_SYCL;


    if (work_group_size > 0) {
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
  
        const size_t global_size = work_group_size * RAJA_DIVIDE_CEILING_INT(iend, work_group_size);

        initSyclDeviceData(hsum, &m_vsum_init, 1, qu);
        initSyclDeviceData(hmin, &m_vmin_init, 1, qu);
        initSyclDeviceData(hmax, &m_vmax_init, 1, qu);

        qu->submit([&] (sycl::handler& h) {

          auto sum_reduction = sycl::reduction(hsum, sycl::plus<>());
          auto min_reduction = sycl::reduction(hmin, sycl::minimum<>());
          auto max_reduction = sycl::reduction(hmax, sycl::maximum<>());

          h.parallel_for(sycl::nd_range<1>(global_size, work_group_size),
                         sum_reduction, min_reduction, max_reduction,
                         [=] (sycl::nd_item<1> item, auto& vsum, auto& vmin, auto& vmax) {
  
            Index_type i = item.get_global_id(0);
            if (i < iend) {
             // REDUCE3_INT_BODY
                vsum += vec[i];
                vmin.combine(vec[i]); 
                vmax.combine(vec[i]);
            }
  
          });
        });
  
        Int_type lsum;
        Int_ptr plsum = &lsum;
        getSyclDeviceData(plsum, hsum, 1, qu);
        m_vsum += lsum;
  
        Int_type lmin;
        Int_ptr plmin = &lmin;
        getSyclDeviceData(plmin, hmin, 1, qu);
        m_vmin = RAJA_MIN(m_vmin, lmin);
  
        Int_type lmax;
        Int_ptr plmax = &lmax;
        getSyclDeviceData(plmax, hmax, 1, qu);
        m_vmax = RAJA_MAX(m_vmax, lmax);

      }
      qu->wait();
      stopTimer();
  
    } else {
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        initSyclDeviceData(hsum, &m_vsum_init, 1, qu);
        initSyclDeviceData(hmin, &m_vmin_init, 1, qu);
        initSyclDeviceData(hmax, &m_vmax_init, 1, qu);


        qu->submit([&] (sycl::handler& h) {

          auto sum_reduction = sycl::reduction(hsum, sycl::plus<>());
          auto min_reduction = sycl::reduction(hmin, sycl::minimum<>());
          auto max_reduction = sycl::reduction(hmax, sycl::maximum<>());

          h.parallel_for(sycl::range<1>(iend),
                         sum_reduction, min_reduction, max_reduction,
                         [=] (sycl::item<1> item, auto& vsum, auto& vmin, auto& vmax ) {
  
            Index_type i = item.get_id(0);
            vsum += vec[i];
            vmin.combine(vec[i]);       
            vmax.combine(vec[i]);

          });
        });

        Int_type lsum;
        Int_ptr plsum = &lsum;
        getSyclDeviceData(plsum, hsum, 1, qu);
        m_vsum += lsum;
  
        Int_type lmin;
        Int_ptr plmin = &lmin;
        getSyclDeviceData(plmin, hmin, 1, qu);
        m_vmin = RAJA_MIN(m_vmin, lmin);
  
        Int_type lmax;
        Int_ptr plmax = &lmax;
        getSyclDeviceData(plmax, hmax, 1, qu);
        m_vmax = RAJA_MAX(m_vmax, lmax);

      }
      qu->wait();
      stopTimer();
  
    } 

    REDUCE3_INT_DATA_TEARDOWN_SYCL;
  } else if ( vid == RAJA_SYCL ) {

    if ( work_group_size == 0 ) {
      std::cout << "\n  REDUCE3_INT : RAJA_SYCL does not support auto work group size" << std::endl;
      return;
    }

    REDUCE3_INT_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::sycl_reduce, Int_type> vsum(m_vsum_init);
      RAJA::ReduceMin<RAJA::sycl_reduce, Int_type> vmin(m_vmin_init);
      RAJA::ReduceMax<RAJA::sycl_reduce, Int_type> vmax(m_vmax_init);

      RAJA::forall< RAJA::sycl_exec<work_group_size, false> >(
        RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
        REDUCE3_INT_BODY_RAJA;
      });

      m_vsum += static_cast<Int_type>(vsum.get());
      m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
      m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));

    }
    qu->wait();
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  REDUCE3_INT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(REDUCE3_INT, Sycl)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
