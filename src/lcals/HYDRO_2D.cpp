//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


HYDRO_2D::HYDRO_2D(const RunParams& params)
  : KernelBase(rajaperf::Lcals_HYDRO_2D, params)
{
   m_jn = 1000;
   m_kn = 1000;

   m_s = 0.0041;
   m_t = 0.0037;

   setDefaultSize(m_jn);
   setDefaultReps(200);
}

HYDRO_2D::~HYDRO_2D() 
{
}

void HYDRO_2D::setUp(VariantID vid)
{
  m_kn = getRunSize();
  m_array_length = m_kn * m_jn;

  allocAndInitDataConst(m_zrout, m_array_length, 0.0, vid);
  allocAndInitDataConst(m_zzout, m_array_length, 0.0, vid);
  allocAndInitData(m_za, m_array_length, vid);
  allocAndInitData(m_zb, m_array_length, vid);
  allocAndInitData(m_zm, m_array_length, vid);
  allocAndInitData(m_zp, m_array_length, vid);
  allocAndInitData(m_zq, m_array_length, vid);
  allocAndInitData(m_zr, m_array_length, vid);
  allocAndInitData(m_zu, m_array_length, vid);
  allocAndInitData(m_zv, m_array_length, vid);
  allocAndInitData(m_zz, m_array_length, vid);
}

void HYDRO_2D::runKernel(VariantID vid)
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
      std::cout << "\n  HYDRO_2D : Unknown variant id = " << vid << std::endl;
    }

  }

}

void HYDRO_2D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_zzout, m_array_length);
  checksum[vid] += calcChecksum(m_zrout, m_array_length);
}

void HYDRO_2D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_zrout);
  deallocData(m_zzout);
  deallocData(m_za);
  deallocData(m_zb);
  deallocData(m_zm);
  deallocData(m_zp);
  deallocData(m_zq);
  deallocData(m_zr);
  deallocData(m_zu);
  deallocData(m_zv);
  deallocData(m_zz);
}

} // end namespace lcals
} // end namespace rajaperf
