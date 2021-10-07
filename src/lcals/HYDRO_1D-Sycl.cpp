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

#include "HYDRO_1D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace lcals
{

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 256;


#define HYDRO_1D_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(x, m_x, m_array_length, qu); \
  allocAndInitSyclDeviceData(y, m_y, m_array_length, qu); \
  allocAndInitSyclDeviceData(z, m_z, m_array_length, qu);

#define HYDRO_1D_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_x, x, m_array_length, qu); \
  deallocSyclDeviceData(x, qu); \
  deallocSyclDeviceData(y, qu); \
  deallocSyclDeviceData(z, qu);

void HYDRO_1D::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  HYDRO_1D_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    HYDRO_1D_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);
      qu.submit([&] (cl::sycl::handler& h)
      {
        h.parallel_for<class syclHydro1D>(cl::sycl::nd_range<1>(grid_size, block_size),
                                          [=] (cl::sycl::nd_item<1> item ) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            HYDRO_1D_BODY
          }

        });
      });
    }
    qu.wait(); // Wait for computation to finish before stopping timer
    stopTimer();

    HYDRO_1D_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    HYDRO_1D_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::sycl_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
         HYDRO_1D_BODY;
       });

    }
    qu.wait();
    stopTimer();

    HYDRO_1D_DATA_TEARDOWN_SYCL;

  } else { 
     std::cout << "\n  HYDRO_1D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
