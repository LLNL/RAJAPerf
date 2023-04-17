//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PLANCKIAN.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace lcals
{


PLANCKIAN::PLANCKIAN(const RunParams& params)
  : KernelBase(rajaperf::Lcals_PLANCKIAN, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(50);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (2*sizeof(Real_type ) + 3*sizeof(Real_type )) * getActualProblemSize() );
  setFLOPsPerRep(4 * getActualProblemSize()); // 1 exp

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

  setVariantDefined( Kokkos_Lambda );

  setVariantDefined( Base_StdPar );
  setVariantDefined( Lambda_StdPar );
}

PLANCKIAN::~PLANCKIAN()
{
}

void PLANCKIAN::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitData(m_x, getActualProblemSize(), vid);
  allocAndInitData(m_y, getActualProblemSize(), vid);
  allocAndInitData(m_u, getActualProblemSize(), vid);
  allocAndInitData(m_v, getActualProblemSize(), vid);
  allocAndInitDataConst(m_w, getActualProblemSize(), 0.0, vid);
}

void PLANCKIAN::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_w, getActualProblemSize(), vid);
}

void PLANCKIAN::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_x, vid);
  deallocData(m_y, vid);
  deallocData(m_u, vid);
  deallocData(m_v, vid);
  deallocData(m_w, vid);
}

} // end namespace lcals
} // end namespace rajaperf
