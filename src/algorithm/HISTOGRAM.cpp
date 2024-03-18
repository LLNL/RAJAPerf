//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HISTOGRAM.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

namespace rajaperf
{
namespace algorithm
{


HISTOGRAM::HISTOGRAM(const RunParams& params)
  : KernelBase(rajaperf::Algorithm_HISTOGRAM, params)
{
  setDefaultProblemSize(1000000);
  setDefaultReps(50);

  setActualProblemSize( getTargetProblemSize() );

  m_num_bins = 10;

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesPerRep( (1*sizeof(Data_type) + 1*sizeof(Data_type))*m_num_bins +
                  (1*sizeof(Index_type) + 0*sizeof(Index_type)) * getActualProblemSize() );
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
  setVariantDefined( RAJA_OpenMPTarget );

  setVariantDefined( Base_CUDA );
  setVariantDefined( Lambda_CUDA );
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
  setVariantDefined( Lambda_HIP );
  setVariantDefined( RAJA_HIP );

  setVariantDefined( Kokkos_Lambda );
}

HISTOGRAM::~HISTOGRAM()
{
}

void HISTOGRAM::setUp(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  allocData(m_bins, getActualProblemSize(), vid);
  {
    auto reset_bins = scopedMoveData(m_bins, getActualProblemSize(), vid);
    Real_ptr data;
    allocAndInitDataRandValue(data, getActualProblemSize(), Base_Seq);

    for (Index_type i = 0; i < getActualProblemSize(); ++i) {
      m_bins[i] = static_cast<Index_type>(data[i] * m_num_bins);
      if (m_bins[i] >= m_num_bins) {
        m_bins[i] = m_num_bins - 1;
      }
      if (m_bins[i] < 0) {
        m_bins[i] = 0;
      }
    }

    deallocData(data, Base_Seq);
  }

  m_counts_init.resize(m_num_bins, 0);
  m_counts_final.resize(m_num_bins, 0);
}

void HISTOGRAM::updateChecksum(VariantID vid, size_t tune_idx)
{
  checksum[vid][tune_idx] += calcChecksum(m_counts_final.data(), m_num_bins, vid);
}

void HISTOGRAM::tearDown(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
{
  (void) vid;
  deallocData(m_bins, vid);
  m_counts_init.clear(); m_counts_init.shrink_to_fit();
  m_counts_final.clear(); m_counts_final.shrink_to_fit();
}

} // end namespace algorithm
} // end namespace rajaperf
