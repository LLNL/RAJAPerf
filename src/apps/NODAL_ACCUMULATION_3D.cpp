//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NODAL_ACCUMULATION_3D.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"
#include "common/DataUtils.hpp"

#include <cmath>


namespace rajaperf
{
namespace apps
{


NODAL_ACCUMULATION_3D::NODAL_ACCUMULATION_3D(const RunParams& params)
  : KernelBase(rajaperf::Apps_NODAL_ACCUMULATION_3D, params)
{
  setDefaultProblemSize(100*100*100);  // See rzmax in ADomain struct
  setDefaultReps(100);

  Index_type rzmax = std::cbrt(getTargetProblemSize())+1;
  m_domain = new ADomain(rzmax, /* ndims = */ 3);

  m_nodal_array_length = m_domain->nnalls;
  m_zonal_array_length = m_domain->lpz+1;

  setActualProblemSize( m_domain->n_real_zones );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  // touched data size, not actual number of stores and loads
  setBytesPerRep( (0*sizeof(Index_type) + 1*sizeof(Index_type)) * getItsPerRep() +
                  (0*sizeof(Real_type) + 1*sizeof(Real_type)) * getItsPerRep() +
                  (1*sizeof(Real_type) + 1*sizeof(Real_type)) * (m_domain->imax+1 - m_domain->imin)*(m_domain->jmax+1 - m_domain->jmin)*(m_domain->kmax+1 - m_domain->kmin));
  setFLOPsPerRep(9 * getItsPerRep());

  checksum_scale_factor = 0.001 *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );

  setUsesFeature(Forall);
  setUsesFeature(Atomic);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_OpenMPTarget );
  setVariantDefined( RAJA_OpenMPTarget );

  setVariantDefined( Base_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );
}

NODAL_ACCUMULATION_3D::~NODAL_ACCUMULATION_3D()
{
  delete m_domain;
}

void NODAL_ACCUMULATION_3D::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataConst(m_x, m_nodal_array_length, 0.0, vid);
  allocAndInitDataConst(m_vol, m_zonal_array_length, 1.0, vid);
}

void NODAL_ACCUMULATION_3D::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid].at(tune_idx) += calcChecksum(m_x, m_nodal_array_length, checksum_scale_factor );
}

void NODAL_ACCUMULATION_3D::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;

  deallocData(m_x);
  deallocData(m_vol);
}

} // end namespace apps
} // end namespace rajaperf
