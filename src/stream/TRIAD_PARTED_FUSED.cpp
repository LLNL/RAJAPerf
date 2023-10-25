//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD_PARTED_FUSED.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <algorithm>

namespace rajaperf
{
namespace stream
{


TRIAD_PARTED_FUSED::TRIAD_PARTED_FUSED(const RunParams& params)
  : KernelBase(rajaperf::Stream_TRIAD_PARTED_FUSED, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(1000);

  setActualProblemSize( getTargetProblemSize() );

  const Index_type num_parts = std::min(params.getNumParts(), getActualProblemSize());

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Real_type) + 2*sizeof(Real_type)) *
                  getActualProblemSize() );
  setFLOPsPerRep(2 * getActualProblemSize());

  checksum_scale_factor = 0.001 *
              ( static_cast<Checksum_type>(getDefaultProblemSize()) /
                                           getActualProblemSize() );

  m_parts.reserve(num_parts+1);
  m_parts.emplace_back(0);
  for (Index_type p = 1; p < num_parts; ++p) {
    // use evenly spaced parts for now
    m_parts.emplace_back((getActualProblemSize()/num_parts)*p +
                         (getActualProblemSize()%num_parts)*p / num_parts);
  }
  m_parts.emplace_back(getActualProblemSize());

  setUsesFeature( Workgroup );

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
}

TRIAD_PARTED_FUSED::~TRIAD_PARTED_FUSED()
{
}

void TRIAD_PARTED_FUSED::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocAndInitDataConst(m_a, getActualProblemSize(), 0.0, vid);
  allocAndInitData(m_b, getActualProblemSize(), vid);
  allocAndInitData(m_c, getActualProblemSize(), vid);
  initData(m_alpha, vid);
}

void TRIAD_PARTED_FUSED::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_a, getActualProblemSize(), checksum_scale_factor , vid);
}

void TRIAD_PARTED_FUSED::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_a, vid);
  deallocData(m_b, vid);
  deallocData(m_c, vid);
}

} // end namespace stream
} // end namespace rajaperf
