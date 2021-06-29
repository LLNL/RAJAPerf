//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES_NOVIEW.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace apps
{


LTIMES_NOVIEW::LTIMES_NOVIEW(const RunParams& params)
  : KernelBase(rajaperf::Apps_LTIMES_NOVIEW, params)
{
  m_num_d_default = 64;
  m_num_z_default = 488;
  m_num_g_default = 32;
  m_num_m_default = 25;

  setDefaultSize(m_num_d_default * m_num_g_default * m_num_z_default);
  setDefaultReps(50);

  m_num_z = getRunSize() / (m_num_d_default * m_num_g_default);
  m_num_g = m_num_g_default;
  m_num_m = m_num_m_default;
  m_num_d = m_num_d_default;

  m_philen = m_num_m * m_num_g * m_num_z;
  m_elllen = m_num_d * m_num_m;
  m_psilen = m_num_d * m_num_g * m_num_z;

  setProblemSize( m_num_d * m_num_g * m_num_z );

  setItsPerRep( getProblemSize() );
  setKernelsPerRep(1);
  // using total data size instead of writes and reads
  setBytesPerRep( (1*sizeof(Real_type) + 1*sizeof(Real_type)) * m_philen +
                  (0*sizeof(Real_type) + 1*sizeof(Real_type)) * m_elllen +
                  (0*sizeof(Real_type) + 1*sizeof(Real_type)) * m_psilen );
  setFLOPsPerRep(2 * m_num_z * m_num_g * m_num_m * m_num_d);

  setUsesFeature(Kernel);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_OpenMPTarget );
  setVariantDefined( RAJA_OpenMPTarget );

  setVariantDefined( Base_CUDA );
  setVariantDefined( Lambda_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( Lambda_HIP );
  setVariantDefined( RAJA_HIP );
}

LTIMES_NOVIEW::~LTIMES_NOVIEW()
{
}

void LTIMES_NOVIEW::setUp(VariantID vid)
{
  allocAndInitDataConst(m_phidat, int(m_philen), Real_type(0.0), vid);
  allocAndInitData(m_elldat, int(m_elllen), vid);
  allocAndInitData(m_psidat, int(m_psilen), vid);
}

void LTIMES_NOVIEW::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_phidat, m_philen);
}

void LTIMES_NOVIEW::tearDown(VariantID vid)
{
  (void) vid;

  deallocData(m_phidat);
  deallocData(m_elldat);
  deallocData(m_psidat);
}

} // end namespace apps
} // end namespace rajaperf
