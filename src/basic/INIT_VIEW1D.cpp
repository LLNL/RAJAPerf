//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


INIT_VIEW1D::INIT_VIEW1D(const RunParams& params)
  : KernelBase(rajaperf::Basic_INIT_VIEW1D, params)
{
   setDefaultSize(500000);
   setDefaultReps(5000);
}

INIT_VIEW1D::~INIT_VIEW1D() 
{
}

void INIT_VIEW1D::setUp(VariantID vid)
{
  allocAndInitDataConst(m_a, getRunSize(), 0.0, vid);
  m_val = 0.123;
}

void INIT_VIEW1D::runKernel(VariantID vid)
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
      std::cout << "\n  INIT_VIEW1D : Unknown variant id = " << vid << std::endl;
    }

  }

}

void INIT_VIEW1D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_a, getRunSize());
}

void INIT_VIEW1D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
}

} // end namespace basic
} // end namespace rajaperf
