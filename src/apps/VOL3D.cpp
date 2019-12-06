//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "VOL3D.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"
#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{


VOL3D::VOL3D(const RunParams& params)
  : KernelBase(rajaperf::Apps_VOL3D, params)
{
  setDefaultSize(64);  // See rzmax in ADomain struct
  setDefaultReps(300);

  m_domain = new ADomain(getRunSize(), /* ndims = */ 3);

  m_array_length = m_domain->nnalls;;
}

VOL3D::~VOL3D() 
{
  delete m_domain;
}

Index_type VOL3D::getItsPerRep() const { 
  return m_domain->lpz+1 - m_domain->fpz;
}

void VOL3D::setUp(VariantID vid)
{
  allocAndInitDataConst(m_x, m_array_length, 0.0, vid);
  allocAndInitDataConst(m_y, m_array_length, 0.0, vid);
  allocAndInitDataConst(m_z, m_array_length, 0.0, vid);

  Real_type dx = 0.3;
  Real_type dy = 0.2;
  Real_type dz = 0.1;
  setMeshPositions_3d(m_x, dx, m_y, dy, m_z, dz, *m_domain);

  allocAndInitDataConst(m_vol, m_array_length, 0.0, vid);

  m_vnormq = 0.083333333333333333; /* vnormq = 1/12 */  
}

void VOL3D::runKernel(VariantID vid)
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
      std::cout << "\n  VOL3D : Unknown variant id = " << vid << std::endl;
    }

  }
}

void VOL3D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_vol, m_array_length);
}

void VOL3D::tearDown(VariantID vid)
{
  (void) vid;

  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_z);
  deallocData(m_vol);
}

} // end namespace apps
} // end namespace rajaperf
