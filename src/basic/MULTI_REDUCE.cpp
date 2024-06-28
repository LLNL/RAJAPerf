//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MULTI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace basic
{


MULTI_REDUCE::MULTI_REDUCE(const RunParams& params)
  : KernelBase(rajaperf::Basic_MULTI_REDUCE, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(50);

  setActualProblemSize( getTargetProblemSize() );

  m_num_bins = 10;

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Data_type) + 1*sizeof(Data_type))*m_num_bins +
                  (1*sizeof(Data_type) + 0*sizeof(Data_type) +
                   1*sizeof(Index_type) + 0*sizeof(Index_type)) * getActualProblemSize() );
  setFLOPsPerRep(1 * getActualProblemSize());

  setUsesFeature(Forall);
  setUsesFeature(Atomic);

  setVariantDefined( Base_Seq );
  setVariantDefined( Lambda_Seq );
  setVariantDefined( RAJA_Seq );

  setVariantDefined( Base_OpenMP );
  setVariantDefined( Lambda_OpenMP );
  setVariantDefined( RAJA_OpenMP );

  setVariantDefined( Base_OpenMPTarget );

  setVariantDefined( Base_CUDA );
  setVariantDefined( Lambda_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( Lambda_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Kokkos_Lambda );
}

MULTI_REDUCE::~MULTI_REDUCE()
{
}

void MULTI_REDUCE::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocData(m_bins, getActualProblemSize(), vid);
  allocAndInitDataRandValue(m_data, getActualProblemSize(), vid);
  {
    auto reset_bins = scopedMoveData(m_bins, getActualProblemSize(), vid);
    auto reset_data = scopedMoveData(m_data, getActualProblemSize(), vid);

    for (Index_type i = 0; i < getActualProblemSize(); ++i) {
      m_bins[i] = static_cast<Index_type>(m_data[i] * m_num_bins);
      if (m_bins[i] >= m_num_bins) {
        m_bins[i] = m_num_bins - 1;
      }
      if (m_bins[i] < 0) {
        m_bins[i] = 0;
      }
    }
  }

  m_values_init.resize(m_num_bins, 0.0);
  m_values_final.resize(m_num_bins, 0.0);
}

void MULTI_REDUCE::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_values_final.data(), m_num_bins, vid);
}

void MULTI_REDUCE::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_bins, vid);
  deallocData(m_data, vid);
  m_values_init.clear(); m_values_init.shrink_to_fit();
  m_values_final.clear(); m_values_final.shrink_to_fit();
}

} // end namespace basic
} // end namespace rajaperf
