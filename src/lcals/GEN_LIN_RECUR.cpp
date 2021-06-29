//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace lcals
{


GEN_LIN_RECUR::GEN_LIN_RECUR(const RunParams& params)
  : KernelBase(rajaperf::Lcals_GEN_LIN_RECUR, params)
{
  setDefaultSize(1000000);
  setDefaultReps(500);

  m_N = getRunSize();

  setProblemSize( getRunSize() );

  setItsPerRep( getProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (2*sizeof(Real_type ) + 3*sizeof(Real_type )) * m_N +
                  (2*sizeof(Real_type ) + 3*sizeof(Real_type )) * m_N );
  setFLOPsPerRep((3 +
                  3 ) * getRunSize());

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


}

GEN_LIN_RECUR::~GEN_LIN_RECUR()
{
}

void GEN_LIN_RECUR::setUp(VariantID vid)
{
  m_kb5i = 0;

  allocAndInitDataConst(m_b5, m_N, 0.0, vid);
  allocAndInitData(m_stb5, m_N, vid);
  allocAndInitData(m_sa, m_N, vid);
  allocAndInitData(m_sb, m_N, vid);
}

void GEN_LIN_RECUR::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_b5, getRunSize());
}

void GEN_LIN_RECUR::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_b5);
  deallocData(m_stb5);
  deallocData(m_sa);
  deallocData(m_sb);
}

} // end namespace lcals
} // end namespace rajaperf
