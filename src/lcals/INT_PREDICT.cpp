//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INT_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


INT_PREDICT::INT_PREDICT(const RunParams& params)
  : KernelBase(rajaperf::Lcals_INT_PREDICT, params)
{
   setDefaultSize(100000);
   setDefaultReps(4000);
}

INT_PREDICT::~INT_PREDICT() 
{
}

void INT_PREDICT::setUp(VariantID vid)
{
  m_array_length = getRunSize() * 13;
  m_offset = getRunSize();

  m_px_initval = 1.0;
  allocAndInitDataConst(m_px, m_array_length, m_px_initval, vid);

  initData(m_dm22);
  initData(m_dm23);
  initData(m_dm24);
  initData(m_dm25);
  initData(m_dm26);
  initData(m_dm27);
  initData(m_dm28);
  initData(m_c0);
}

void INT_PREDICT::runKernel(VariantID vid)
{

  switch ( vid ) {

    case Base_Seq :
#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq :
    case RAJA_Seq :
#endif
    {
      runSeqVariant(vid);
      break;
    }

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP :
    case Lambda_OpenMP :
    case RAJA_OpenMP :
    {
      runOpenMPVariant(vid);
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
      std::cout << "\n  INT_PREDICT : Unknown variant id = " << vid << std::endl;
    }

  }

}

void INT_PREDICT::updateChecksum(VariantID vid)
{
  for (Index_type i = 0; i < getRunSize(); ++i) {
    m_px[i] -= m_px_initval;
  }
  
  checksum[vid] += calcChecksum(m_px, getRunSize());
}

void INT_PREDICT::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_px);
}

} // end namespace lcals
} // end namespace rajaperf
