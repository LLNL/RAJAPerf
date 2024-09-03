//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


DAXPY_ATOMIC::DAXPY_ATOMIC(const RunParams& params)
  : KernelBase(rajaperf::Basic_DAXPY_ATOMIC, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(500);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesReadPerRep( 1*sizeof(Real_type) * getActualProblemSize() );
  setBytesWrittenPerRep( 0 );
  setBytesAtomicModifyWrittenPerRep( 1*sizeof(Real_type) * getActualProblemSize() );
  setFLOPsPerRep(2 * getActualProblemSize());

  setComplexity(Complexity::N);

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
  setVariantDefined( Lambda_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( Lambda_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Kokkos_Lambda );
}

DAXPY_ATOMIC::~DAXPY_ATOMIC()
{
}

void DAXPY_ATOMIC::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataConst(m_y, getActualProblemSize(), 0.0, vid);
  allocAndInitData(m_x, getActualProblemSize(), vid);
  initData(m_a, vid);
}

void DAXPY_ATOMIC::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid].at(tune_idx) += calcChecksum(m_y, getActualProblemSize(), vid);
}

void DAXPY_ATOMIC::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_x, vid);
  deallocData(m_y, vid);
}

} // end namespace basic
} // end namespace rajaperf
