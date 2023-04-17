//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
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
                  (1*sizeof(Real_type) + 1*sizeof(Real_type)) * m_domain->n_real_nodes);
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

  setVariantDefined( Base_StdPar );
  setVariantDefined( Lambda_StdPar );
}

NODAL_ACCUMULATION_3D::~NODAL_ACCUMULATION_3D()
{
  delete m_domain;
}

void NODAL_ACCUMULATION_3D::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataConst(m_x, m_nodal_array_length, 0.0, vid);
  allocAndInitDataConst(m_vol, m_zonal_array_length, 1.0, vid);
  allocAndInitDataConst(m_real_zones, m_domain->n_real_zones,
                        static_cast<Index_type>(-1), vid);

  {
    auto reset_rz = scopedMoveData(m_real_zones, m_domain->n_real_zones, vid);

    setRealZones_3d(m_real_zones, *m_domain);
  }
}

void NODAL_ACCUMULATION_3D::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid].at(tune_idx) += calcChecksum(m_x, m_nodal_array_length, checksum_scale_factor , vid);
}

void NODAL_ACCUMULATION_3D::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;

  deallocData(m_x, vid);
  deallocData(m_vol, vid);
  deallocData(m_real_zones, vid);
}

} // end namespace apps
} // end namespace rajaperf
