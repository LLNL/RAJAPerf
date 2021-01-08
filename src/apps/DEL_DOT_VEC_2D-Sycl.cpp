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

#include "DEL_DOT_VEC_2D.hpp"

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


#define DEL_DOT_VEC_2D_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(x, m_x, m_array_length, qu); \
  allocAndInitSyclDeviceData(y, m_y, m_array_length, qu); \
  allocAndInitSyclDeviceData(xdot, m_xdot, m_array_length, qu); \
  allocAndInitSyclDeviceData(ydot, m_ydot, m_array_length, qu); \
  allocAndInitSyclDeviceData(div, m_div, m_array_length, qu); \
  allocAndInitSyclDeviceData(real_zones, m_domain->real_zones, iend, qu);

#define DEL_DOT_VEC_2D_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_div, div, m_array_length, qu); \
  deallocSyclDeviceData(x, qu); \
  deallocSyclDeviceData(y, qu); \
  deallocSyclDeviceData(xdot, qu); \
  deallocSyclDeviceData(ydot, qu); \
  deallocSyclDeviceData(div, qu); \
  deallocSyclDeviceData(real_zones, qu);

void DEL_DOT_VEC_2D::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = m_domain->n_real_zones;

  DEL_DOT_VEC_2D_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    DEL_DOT_VEC_2D_DATA_SETUP_SYCL;

    NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
    NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
    NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
    NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class DelDotVec>(cl::sycl::nd_range<1> (grid_size, block_size),
                                        [=] (cl::sycl::nd_item<1> item) {

          Index_type ii = item.get_global_id(0);
          if (ii < iend) {
            DEL_DOT_VEC_2D_BODY_INDEX
            DEL_DOT_VEC_2D_BODY
          }

        });
      });

    }
    qu.wait(); // Wait for computation to finish before stopping timer
    stopTimer();

    DEL_DOT_VEC_2D_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    DEL_DOT_VEC_2D_DATA_SETUP_SYCL;

    NDSET2D(m_domain->jp, x,x1,x2,x3,x4) ;
    NDSET2D(m_domain->jp, y,y1,y2,y3,y4) ;
    NDSET2D(m_domain->jp, xdot,fx1,fx2,fx3,fx4) ;
    NDSET2D(m_domain->jp, ydot,fy1,fy2,fy3,fy4) ;

    RAJA::ListSegment zones(m_domain->real_zones, m_domain->n_real_zones);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::sycl_exec<block_size, true /*async*/> >(
         zones, [=] (Index_type i) {
         DEL_DOT_VEC_2D_BODY;
       });

    }
    qu.wait();
    stopTimer();

    DEL_DOT_VEC_2D_DATA_TEARDOWN_SYCL;


  } else {
     std::cout << "\n  DEL_DOT_VEC_2D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
