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

#include <algorithm>

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

    bool init_even_sizes = false;
    bool init_random_sizes = true;
    bool init_random_per_iterate = false;
    if (init_even_sizes || init_random_sizes) {
      Real_ptr data = nullptr;
      if (init_even_sizes) {
        allocData(data, m_num_bins, Base_Seq);
        for (Index_type b = 0; b < m_num_bins; ++b) {
          data[b] = static_cast<Real_type>(b+1) / m_num_bins;
        }
      } else if (init_random_sizes) {
        allocAndInitDataRandValue(data, m_num_bins, Base_Seq);
        std::sort(data, data+m_num_bins);
      }

      Index_type actual_prob_size = getActualProblemSize();
      Index_type bin = 0;
      for (Index_type i = 0; i < actual_prob_size; ++i) {
        Real_type pos = static_cast<Real_type>(i) / actual_prob_size;
        while (bin+1 < m_num_bins && pos >= data[bin]) {
          bin += 1;
        }
        m_bins[i] = bin;
      }

      deallocData(data, Base_Seq);

    } else if (init_random_per_iterate) {
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
    } else {
      throw 1;
    }
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
