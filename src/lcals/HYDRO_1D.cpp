//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_1D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace lcals
{


HYDRO_1D::HYDRO_1D(const RunParams& params)
  : KernelBase(rajaperf::Lcals_HYDRO_1D, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(1000);

  setActualProblemSize( getTargetProblemSize() );

  m_array_length = getActualProblemSize() + 12;

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type ) + 1*sizeof(Real_type )) * getActualProblemSize() +
                  (0*sizeof(Real_type ) + 1*sizeof(Real_type )) * (getActualProblemSize()+1) );
  setFLOPsPerRep(5 * getActualProblemSize());

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

  setVariantDefined( Base_StdPar );
  setVariantDefined( Lambda_StdPar );
  setVariantDefined( RAJA_StdPar );
}

HYDRO_1D::~HYDRO_1D()
{
}

void HYDRO_1D::setUp(VariantID vid)
{
  allocAndInitDataConst(m_x, m_array_length, 0.0, vid);
  allocAndInitData(m_y, m_array_length, vid);
  allocAndInitData(m_z, m_array_length, vid);

  initData(m_q, vid);
  initData(m_r, vid);
  initData(m_t, vid);
}

void HYDRO_1D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x, getActualProblemSize(), checksum_scale_factor );
}

void HYDRO_1D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_x);
  deallocData(m_y);
  deallocData(m_z);
}

} // end namespace lcals
} // end namespace rajaperf
