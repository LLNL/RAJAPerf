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

#include "TRAP_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <sycl.hpp>

#include <iostream>

namespace rajaperf
{
namespace basic
{

//
// Function used in TRAP_INT loop.
//
RAJA_INLINE
RAJA_DEVICE
Real_type trap_int_func(Real_type x,
                        Real_type y,
                        Real_type xp,
                        Real_type yp)
{
   Real_type denom = (x - xp)*(x - xp) + (y - yp)*(y - yp);
   denom = 1.0/sqrt(denom);
   return denom;
}

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 256;


#define TRAP_INT_DATA_SETUP_SYCL  // nothing to do here...

#define TRAP_INT_DATA_TEARDOWN_SYCL // nothing to do here...

void TRAP_INT::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRAP_INT_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    TRAP_INT_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      Real_type sumx = m_sumx_init;
      
      {
        sycl::buffer<Real_type, 1> buf_sumx(&sumx, 1);

        const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

        qu->submit([&] (sycl::handler& cgh) {

         auto sumReduction = reduction(buf_sumx, cgh, sycl::plus<Real_type>());

         cgh.parallel_for(sycl::nd_range<1>{grid_size, block_size},
       	                sumReduction,
                        [=] (sycl::nd_item<1> item, auto& sumx) {

            Index_type i = item.get_global_id(0);
	    if (i < iend) {
              TRAP_INT_BODY;
            }

          });
        });
      }

      m_sumx += sumx *h;
      
    }
    stopTimer();

  } else if ( vid == RAJA_SYCL ) {

    TRAP_INT_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::sycl_reduce, Real_type> sumx(m_sumx_init);

      RAJA::forall< RAJA::sycl_exec_nontrivial<block_size, false /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] (Index_type i) {
        TRAP_INT_BODY;
      });

      m_sumx += static_cast<Real_type>(sumx.get()) * h;

    }
    stopTimer();

    TRAP_INT_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  TRAP_INT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
