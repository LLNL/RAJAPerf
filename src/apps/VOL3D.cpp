//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "VOL3D.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"
#include "common/DataUtils.hpp"

#include <cmath>


namespace rajaperf
{
namespace apps
{


VOL3D::VOL3D(const RunParams& params)
  : KernelBase(rajaperf::Apps_VOL3D, params)
{
  setDefaultProblemSize(100*100*100);  // See rzmax in ADomain struct
  setDefaultReps(100);

  Index_type rzmax = std::cbrt(getTargetProblemSize()) + 1 + std::cbrt(3)-1;
  m_domain = new ADomain(rzmax, /* ndims = */ 3);

  m_array_length = m_domain->nnalls;

  setActualProblemSize( m_domain->n_real_zones );

  setItsPerRep( m_domain->lpz+1 - m_domain->fpz );
  setKernelsPerRep(1);
  // touched data size, not actual number of stores and loads
  setBytesReadPerRep( 3*sizeof(Real_type) * (getItsPerRep() + 1+m_domain->jp+m_domain->kp) );
  setBytesWrittenPerRep( 1*sizeof(Real_type) * getItsPerRep() );
  setBytesAtomicModifyWrittenPerRep( 0 );
  setFLOPsPerRep(72 * (m_domain->lpz+1 - m_domain->fpz));

  checksum_scale_factor = 0.001 *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );

  setUsesFeature(Forall);

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

  setVariantDefined( Base_SYCL );
  setVariantDefined( RAJA_SYCL );
}

VOL3D::~VOL3D()
{
  delete m_domain;
}

void VOL3D::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataConst(m_x, m_array_length, 0.0, vid);
  allocAndInitDataConst(m_y, m_array_length, 0.0, vid);
  allocAndInitDataConst(m_z, m_array_length, 0.0, vid);

  {
    auto reset_x = scopedMoveData(m_x, m_array_length, vid);
    auto reset_y = scopedMoveData(m_y, m_array_length, vid);
    auto reset_z = scopedMoveData(m_z, m_array_length, vid);

    Real_type dx = 0.3;
    Real_type dy = 0.2;
    Real_type dz = 0.1;
    setMeshPositions_3d(m_x, dx, m_y, dy, m_z, dz, *m_domain);
  }

  allocAndInitDataConst(m_vol, m_array_length, 0.0, vid);

  m_vnormq = 0.083333333333333333; /* vnormq = 1/12 */
}

void VOL3D::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_vol, m_array_length, checksum_scale_factor , vid);
}

void VOL3D::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;

  deallocData(m_x, vid);
  deallocData(m_y, vid);
  deallocData(m_z, vid);
  deallocData(m_vol, vid);
}

} // end namespace apps
} // end namespace rajaperf
