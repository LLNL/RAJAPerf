//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "IF_QUAD.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


IF_QUAD::IF_QUAD(const RunParams& params)
  : KernelBase(rajaperf::Basic_IF_QUAD, params)
{
  setDefaultSize(1000000);
  setDefaultReps(180);

  setProblemSize( getRunSize() );

  setItsPerRep( getProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (2*sizeof(Real_type) + 3*sizeof(Real_type)) * getRunSize() );
  setFLOPsPerRep(11 * getRunSize()); // 1 sqrt

  setUsesFeature(Forall);


  setVariantDefined( Kokkos_Lambda );


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

IF_QUAD::~IF_QUAD()
{
}

void IF_QUAD::setUp(VariantID vid)
{
  allocAndInitDataRandSign(m_a, getRunSize(), vid);
  allocAndInitData(m_b, getRunSize(), vid);
  allocAndInitData(m_c, getRunSize(), vid);
  allocAndInitDataConst(m_x1, getRunSize(), 0.0, vid);
  allocAndInitDataConst(m_x2, getRunSize(), 0.0, vid);
}

void IF_QUAD::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_x1, getRunSize());
  checksum[vid] += calcChecksum(m_x2, getRunSize());
}

void IF_QUAD::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_b);
  deallocData(m_c);
  deallocData(m_x1);
  deallocData(m_x2);
}

} // end namespace basic
} // end namespace rajaperf
