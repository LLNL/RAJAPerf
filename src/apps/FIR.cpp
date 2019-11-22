//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <algorithm>
#include <iostream>

namespace rajaperf 
{
namespace apps
{


#define FIR_DATA_SETUP_CPU \
  ResReal_ptr in = m_in; \
  ResReal_ptr out = m_out; \
\
  Real_type coeff[FIR_COEFFLEN]; \
  std::copy(std::begin(coeff_array), std::end(coeff_array), std::begin(coeff));\
\
  const Index_type coefflen = m_coefflen;


FIR::FIR(const RunParams& params)
  : KernelBase(rajaperf::Apps_FIR, params)
{
  setDefaultSize(100000);
  setDefaultReps(1600);

  m_coefflen = FIR_COEFFLEN;
}

FIR::~FIR() 
{
}

Index_type FIR::getItsPerRep() const { 
  return getRunSize() - m_coefflen;
}

void FIR::setUp(VariantID vid)
{
  allocAndInitData(m_in, getRunSize(), vid);
  allocAndInitDataConst(m_out, getRunSize(), 0.0, vid);
}

void FIR::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize() - m_coefflen;

  FIR_COEFF;

  FIR_DATA_SETUP_CPU;

  auto fir_lam = [=](int i) {
                   FIR_BODY;
                 };
  
  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          FIR_BODY;
        }

      }
      stopTimer();

      break;
    } 

#if defined(RUN_RAJA_SEQ)     
    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), fir_lam);

      }
      stopTimer(); 

      break;
    }
#endif // RUN_RAJA_SEQ

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)                        
    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
           FIR_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), fir_lam);

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
      std::cout << "\n  FIR : Unknown variant id = " << vid << std::endl;
    }

  }
}

void FIR::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_out, getRunSize());
}

void FIR::tearDown(VariantID vid)
{
  (void) vid;
 
  deallocData(m_in);
  deallocData(m_out);
}

} // end namespace apps
} // end namespace rajaperf
