//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRAP_INT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{


TRAP_INT::TRAP_INT(const RunParams& params)
  : KernelBase(rajaperf::Basic_TRAP_INT, params)
{
   setDefaultSize(100000);
   setDefaultReps(2000);
}

TRAP_INT::~TRAP_INT() 
{
}

void TRAP_INT::setUp(VariantID vid)
{
  Real_type xn; 
  initData(xn, vid);

  initData(m_x0, vid);
  initData(m_xp, vid);
  initData(m_y,  vid);
  initData(m_yp, vid);

  m_h = xn - m_x0;

  m_sumx_init = 0.0;

  m_sumx = 0;
}

void TRAP_INT::runKernel(VariantID vid)
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
      std::cout << "\n  TRAP_INT : Unknown variant id = " << vid << std::endl;
    }

  }

}

void TRAP_INT::updateChecksum(VariantID vid)
{
  checksum[vid] += m_sumx;
}

void TRAP_INT::tearDown(VariantID vid)
{
  (void) vid;
}

} // end namespace basic
} // end namespace rajaperf
