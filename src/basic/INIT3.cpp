//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT3.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


INIT3::INIT3(const RunParams& params)
  : KernelBase(rajaperf::Basic_INIT3, params)
{
  setDefaultSize(1000000);
  setDefaultReps(500);

  setProblemSize( getRunSize() );

  setItsPerRep( getProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (3*sizeof(Real_type) + 2*sizeof(Real_type)) * getRunSize() );
  setFLOPsPerRep(1 * getRunSize());

  setUsesFeature(Forall);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );
  setVariantDefined( Kokkos_Lambda );

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

INIT3::~INIT3()
{
}

void INIT3::setUp(VariantID vid)
{
  allocAndInitDataConst(m_out1, getRunSize(), 0.0, vid);
  allocAndInitDataConst(m_out2, getRunSize(), 0.0, vid);
  allocAndInitDataConst(m_out3, getRunSize(), 0.0, vid);
  allocAndInitData(m_in1, getRunSize(), vid);
  allocAndInitData(m_in2, getRunSize(), vid);
}

void INIT3::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_out1, getRunSize());
  checksum[vid] += calcChecksum(m_out2, getRunSize());
  checksum[vid] += calcChecksum(m_out3, getRunSize());
}

void INIT3::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_out1);
  deallocData(m_out2);
  deallocData(m_out3);
  deallocData(m_in1);
  deallocData(m_in2);
}

} // end namespace basic
} // end namespace rajaperf
