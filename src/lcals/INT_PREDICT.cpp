//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INT_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace lcals
{


INT_PREDICT::INT_PREDICT(const RunParams& params)
  : KernelBase(rajaperf::Lcals_INT_PREDICT, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(400);

  setActualProblemSize( getTargetProblemSize() );

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesReadPerRep( 10*sizeof(Real_type ) * getActualProblemSize() );
  setBytesWrittenPerRep( 1*sizeof(Real_type ) * getActualProblemSize() );
  setBytesAtomicModifyWrittenPerRep( 0 );
  setFLOPsPerRep(17 * getActualProblemSize());

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

  setVariantDefined( Kokkos_Lambda );
}

INT_PREDICT::~INT_PREDICT()
{
}

void INT_PREDICT::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  m_array_length = getActualProblemSize() * 13;
  m_offset = getActualProblemSize();

  m_px_initval = 1.0;
  allocAndInitDataConst(m_px, m_array_length, m_px_initval, vid);

  initData(m_dm22, vid);
  initData(m_dm23, vid);
  initData(m_dm24, vid);
  initData(m_dm25, vid);
  initData(m_dm26, vid);
  initData(m_dm27, vid);
  initData(m_dm28, vid);
  initData(m_c0, vid);
}

void INT_PREDICT::updateChecksum(VariantID vid, size_t tune_idx)
{
  {
    auto reset_px = scopedMoveData(m_px, m_array_length, vid);

    for (Index_type i = 0; i < getActualProblemSize(); ++i) {
      m_px[i] -= m_px_initval;
    }
  }

  checksum[vid][tune_idx] += calcChecksum(m_px, getActualProblemSize(), vid);
}

void INT_PREDICT::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_px, vid);
}

} // end namespace lcals
} // end namespace rajaperf
