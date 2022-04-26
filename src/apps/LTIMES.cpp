//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <algorithm>

namespace rajaperf
{
namespace apps
{


LTIMES::LTIMES(const RunParams& params)
  : KernelBase(rajaperf::Apps_LTIMES, params)
{
  m_num_d_default = 64;
  m_num_z_default = 488;
  m_num_g_default = 32;
  m_num_m_default = 25;

  setDefaultProblemSize(m_num_d_default * m_num_g_default * m_num_z_default);
  setDefaultReps(50);

  m_num_z = std::max( getTargetProblemSize() /
                      (m_num_d_default * m_num_g_default),
                      Index_type(1) );
  m_num_g = m_num_g_default;
  m_num_m = m_num_m_default;
  m_num_d = m_num_d_default;

  m_philen = m_num_m * m_num_g * m_num_z;
  m_elllen = m_num_d * m_num_m;
  m_psilen = m_num_d * m_num_g * m_num_z;

  setActualProblemSize( m_psilen );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  // using total data size instead of writes and reads
  setBytesPerRep( (1*sizeof(Real_type) + 1*sizeof(Real_type)) * m_philen +
                  (0*sizeof(Real_type) + 1*sizeof(Real_type)) * m_elllen +
                  (0*sizeof(Real_type) + 1*sizeof(Real_type)) * m_psilen );
  setFLOPsPerRep(2 * m_num_z * m_num_g * m_num_m * m_num_d);

  checksum_scale_factor = 0.001 *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );

  setUsesFeature(Kernel);
  setUsesFeature(View);

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

LTIMES::~LTIMES()
{
}

void LTIMES::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataConst(m_phidat, int(m_philen), Real_type(0.0), vid);
  allocAndInitData(m_elldat, int(m_elllen), vid);
  allocAndInitData(m_psidat, int(m_psilen), vid);
}

void LTIMES::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_phidat, m_philen, checksum_scale_factor );
}

void LTIMES::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;

  deallocData(m_phidat);
  deallocData(m_elldat);
  deallocData(m_psidat);
}

} // end namespace apps
} // end namespace rajaperf
