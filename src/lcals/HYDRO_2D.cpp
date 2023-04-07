//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <cmath>


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

  setDefaultProblemSize(m_kn * m_jn);
  setDefaultReps(100);

  m_jn = m_kn = std::sqrt(getTargetProblemSize());
  m_array_length = m_kn * m_jn;

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( 3 * getActualProblemSize() );
  setKernelsPerRep(3);
  setBytesPerRep( (2*sizeof(Real_type ) + 0*sizeof(Real_type )) * (m_kn-2) * (m_jn-2) +
                  (0*sizeof(Real_type ) + 4*sizeof(Real_type )) * m_array_length +
                  (2*sizeof(Real_type ) + 0*sizeof(Real_type )) * (m_kn-2) * (m_jn-2) +
                  (0*sizeof(Real_type ) + 4*sizeof(Real_type )) * m_array_length +
                  (2*sizeof(Real_type ) + 4*sizeof(Real_type )) * (m_kn-2) * (m_jn-2) );
  setFLOPsPerRep((14 +
                  26 +
                  4  ) * (m_jn-2)*(m_kn-2));

  checksum_scale_factor = 0.001 *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );

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
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Kokkos_Lambda );
}

HYDRO_2D::~HYDRO_2D()
{
}

void HYDRO_2D::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
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

void HYDRO_2D::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_zzout, m_array_length, checksum_scale_factor , vid);
  checksum[vid][tune_idx] += calcChecksum(m_zrout, m_array_length, checksum_scale_factor , vid);
}

void HYDRO_2D::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_zrout, vid);
  deallocData(m_zzout, vid);
  deallocData(m_za, vid);
  deallocData(m_zb, vid);
  deallocData(m_zm, vid);
  deallocData(m_zp, vid);
  deallocData(m_zq, vid);
  deallocData(m_zr, vid);
  deallocData(m_zu, vid);
  deallocData(m_zv, vid);
  deallocData(m_zz, vid);
}

} // end namespace lcals
} // end namespace rajaperf
