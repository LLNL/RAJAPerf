//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MUL.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace stream
{


MUL::MUL(const RunParams& params)
  : KernelBase(rajaperf::Stream_MUL, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(1800);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type) + 1*sizeof(Real_type)) * 
                  getActualProblemSize() );
  setFLOPsPerRep(1 * getActualProblemSize());

  setUsesFeature( Forall );

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

  setVariantDefined(Kokkos_Lambda);
}

MUL::~MUL()
{
}

void MUL::setUp(VariantID vid)
{
  allocAndInitDataConst(m_b, getActualProblemSize(), 0.0, vid);
  allocAndInitData(m_c, getActualProblemSize(), vid);
  initData(m_alpha, vid);
}

void MUL::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_b, getActualProblemSize());
}

void MUL::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_b);
  deallocData(m_c);
}

} // end namespace stream
} // end namespace rajaperf
