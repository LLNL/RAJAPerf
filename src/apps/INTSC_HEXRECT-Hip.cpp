//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INTSC_HEXRECT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

template < Size_type block_size >
__launch_bounds__(block_size,3)
__global__ void intsc_hexrect_hip
    ( Real_ptr xdnode,     // [ndnodes] x coordinates for donor
      Real_ptr ydnode,     // [ndnodes] y coordinates for donor
      Real_ptr zdnode,     // [ndnodes] z coordinates for donor
      Int_ptr znlist,      // [donor zones][8] donor zone node list
      Char_ptr ncord_gpu,     //  target dimensions and coordinates
      Int_ptr intsc_d,   // [nrecords] donor zones to intersect
      Int_ptr intsc_t,   // [nrecords] target zones to intersect
      Index_type const nrecords,  // Number of threads (one thread per record)
      Real_ptr records )  // output volumes, moments
{
  Index_type blksize = block_size ;  // blocksize = 64  must <= nth_per_isc
  Index_type blk     = blockIdx.x ;
  Index_type irec    = blk*blksize + threadIdx.x ;   // which thread with offset

  __shared__ Real_type xd_work[ (3 * max_polygon_pts+1) * block_size ] ;

  // polygons (an odd number of doubles per thread to reduce bank conflicts)
  Real_ptr my_qx = xd_work + (3 * max_polygon_pts+1) * threadIdx.x ;

  INTSC_HEXRECT_BODY ;
}



template < Size_type block_size >
void INTSC_HEXRECT::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin   = 0 ;
  const Index_type iend     = m_nrecords ;

  auto res{getHipResource()};

  INTSC_HEXRECT_DATA_SETUP;

  //  Insert a warmup call to the kernel in order to remove the
  //  time of initialization that affects the first call to the kernel.
  //
  Bool_type const do_warmup = true ;
  if ( do_warmup ) {
    const Size_type grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
    constexpr Size_type shmem = 0;

    RPlaunchHipKernel( (intsc_hexrect_hip<block_size>),
                       grid_size, block_size,
                       shmem, res.get_stream(),
                       m_xdnode, m_ydnode, m_zdnode, m_znlist,
                       m_ncord, m_intsc_d, m_intsc_t,
                       m_nrecords, m_records ) ;
  }

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXRECT_1");
      const Size_type grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr Size_type shmem = 0;

      RPlaunchHipKernel( (intsc_hexrect_hip<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         m_xdnode, m_ydnode, m_zdnode, m_znlist,
                         m_ncord, m_intsc_d, m_intsc_t,
                         m_nrecords, m_records ) ;
      RP_CALI_SUBKERNEL_END("INTSC_HEXRECT_1");
    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXRECT_1");
      const Size_type grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      auto intsc_hexrect_lambda = [=] __device__ ( Index_type i ) {

          Index_type irec    = i ;
          Index_type thridx  = i % block_size ;

          __shared__ Real_type xd_work[(3 * max_polygon_pts+1) * block_size] ;

          Real_ptr my_qx = xd_work + (3 * max_polygon_pts+1) * thridx ;

          INTSC_HEXRECT_BODY;
      } ;

      constexpr Size_type shmem = 0;

      RPlaunchHipKernel( (lambda_hip_forall<block_size,
                          decltype(intsc_hexrect_lambda)>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         ibegin, iend,
                         intsc_hexrect_lambda );
      RP_CALI_SUBKERNEL_END("INTSC_HEXRECT_1");

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXRECT_1");
      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i)
          {
            Index_type irec    = i ;
            Index_type thridx  = i % block_size ;

            __shared__ Real_type xd_work[(3 * max_polygon_pts+1) * block_size];

            Real_ptr my_qx = xd_work + (3 * max_polygon_pts+1) * thridx ;

            INTSC_HEXRECT_BODY;
          }
      ) ;
      RP_CALI_SUBKERNEL_END("INTSC_HEXRECT_1");
    }
    stopTimer();

  } else {
     getCout() << "\n  INTSC_HEXRECT : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(INTSC_HEXRECT, Hip, Base_HIP, Lambda_HIP, RAJA_HIP)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
