//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D_OFFSET.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

#define INIT_VIEW1D_OFFSET_DATA_SETUP_CPU \
  Real_ptr a = m_a; \
  const Real_type v = m_val;

#define INIT_VIEW1D_OFFSET_DATA_RAJA_SETUP_CPU \
  Real_ptr a = m_a; \
  const Real_type v = m_val; \
\
  ViewType view(a, RAJA::make_offset_layout<1>({{1}}, {{iend+1}}));


INIT_VIEW1D_OFFSET::INIT_VIEW1D_OFFSET(const RunParams& params)
  : KernelBase(rajaperf::Basic_INIT_VIEW1D_OFFSET, params)
{
   setDefaultSize(500000);
   setDefaultReps(5000);
}

INIT_VIEW1D_OFFSET::~INIT_VIEW1D_OFFSET() 
{
}

void INIT_VIEW1D_OFFSET::setUp(VariantID vid)
{
  allocAndInitDataConst(m_a, getRunSize(), 0.0, vid);
  m_val = 0.123;  
}

void INIT_VIEW1D_OFFSET::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getRunSize()+1;

  using ViewType = RAJA::View<Real_type, RAJA::OffsetLayout<1> >;

  switch ( vid ) {

    case Base_Seq : {

      INIT_VIEW1D_OFFSET_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          INIT_VIEW1D_OFFSET_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      INIT_VIEW1D_OFFSET_DATA_RAJA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          INIT_VIEW1D_OFFSET_BODY_RAJA;
        });

      }
      stopTimer();

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)
    case Base_OpenMP : {

      INIT_VIEW1D_OFFSET_DATA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          INIT_VIEW1D_OFFSET_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      INIT_VIEW1D_OFFSET_DATA_RAJA_SETUP_CPU;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), [=](Index_type i) {
          INIT_VIEW1D_OFFSET_BODY_RAJA;
        });

      }
      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
    {
      runOpenMPTargetVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA :
    case RAJA_CUDA :
    {
      runCudaVariant(vid);
      break;
    }
#endif

    default : {
      std::cout << "\n  INIT_VIEW1D_OFFSET : Unknown variant id = " << vid << std::endl;
    }

  }

}

void INIT_VIEW1D_OFFSET::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_a, getRunSize());
}

void INIT_VIEW1D_OFFSET::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
}

} // end namespace basic
} // end namespace rajaperf
