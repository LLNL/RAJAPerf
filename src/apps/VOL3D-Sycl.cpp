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

#include "VOL3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "AppsData.hpp"

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


#define VOL3D_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(x, m_x, m_array_length, qu); \
  allocAndInitSyclDeviceData(y, m_y, m_array_length, qu); \
  allocAndInitSyclDeviceData(z, m_z, m_array_length, qu); \
  allocAndInitSyclDeviceData(vol, m_vol, m_array_length, qu);

#define VOL3D_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_vol, vol, m_array_length, qu); \
  deallocSyclDeviceData(x, qu); \
  deallocSyclDeviceData(y, qu); \
  deallocSyclDeviceData(z, qu); \
  deallocSyclDeviceData(vol, qu);

void VOL3D::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = m_domain->fpz;
  const Index_type iend = m_domain->lpz+1;

  VOL3D_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    VOL3D_DATA_SETUP_SYCL;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
 
      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend - ibegin, block_size);

      qu->submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class VOL3D>(cl::sycl::nd_range<1> (grid_size, block_size),
                                    [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          i += ibegin;
          if(i < iend) {
            VOL3D_BODY
          }

        });
      });
    }
    qu->wait(); // Wait for computation to finish before stopping timer
    stopTimer();
 
    VOL3D_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    VOL3D_DATA_SETUP_SYCL;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::sycl_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
        VOL3D_BODY;
      });

    }
    qu->wait();
    stopTimer();

    VOL3D_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  VOL3D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
