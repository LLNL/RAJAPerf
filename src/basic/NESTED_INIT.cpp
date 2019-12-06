//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

//#define USE_OMP_COLLAPSE
#undef USE_OMP_COLLAPSE


NESTED_INIT::NESTED_INIT(const RunParams& params)
  : KernelBase(rajaperf::Basic_NESTED_INIT, params)
{
  m_ni = 500;
  m_nj = 500;
  m_nk = m_nk_init = 50;

  setDefaultSize(m_ni * m_nj * m_nk);
  setDefaultReps(100);
}

NESTED_INIT::~NESTED_INIT() 
{
}

void NESTED_INIT::setUp(VariantID vid)
{
  m_nk = m_nk_init * static_cast<Real_type>( getRunSize() ) / getDefaultSize();
  m_array_length = m_ni * m_nj * m_nk;

  allocAndInitDataConst(m_array, m_array_length, 0.0, vid);
}

void NESTED_INIT::runKernel(VariantID vid)
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
      std::cout << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
    }

  }

}

void NESTED_INIT::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_array, m_array_length);
}

void NESTED_INIT::tearDown(VariantID vid)
{
  (void) vid;
  RAJA::free_aligned(m_array);
  m_array = 0; 
}

} // end namespace basic
} // end namespace rajaperf
