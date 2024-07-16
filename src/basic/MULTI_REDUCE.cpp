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

#include <algorithm>
#include <stdlib.h>

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

  const char* e_num_bins = getenv("RAJAPERF_MULTI_REDUCE_NUM_BINS");
  m_num_bins = e_num_bins ? atoi(e_num_bins) : 10;

  setItsPerRep( getActualProblemSize() );
  setKernelsPerRep(1);
  setBytesReadPerRep( 1*sizeof(Data_type) * m_num_bins +
                      1*sizeof(Data_type) * getActualProblemSize() +
                      1*sizeof(Index_type) * getActualProblemSize() );
  setBytesWrittenPerRep( 1*sizeof(Data_type) * m_num_bins );
  setBytesAtomicModifyWrittenPerRep( 0 );
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
  setVariantDefined( RAJA_CUDA );

  setVariantDefined( Base_HIP );
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

    const char* e_algorithm = getenv("RAJAPERF_MULTI_REDUCE_BIN_ASSIGNMENT");
    const int algorithm = e_algorithm ? atoi(e_algorithm) : 0;
    const bool init_random_per_iterate = algorithm == 0;
    const bool init_random_sizes = algorithm == 1;
    const bool init_even_sizes = algorithm == 2;
    const bool init_all_one = algorithm == 3;

    if (init_even_sizes || init_random_sizes || init_all_one) {
      Real_ptr data = nullptr;
      if (init_even_sizes) {
        allocData(data, m_num_bins, Base_Seq);
        for (Index_type b = 0; b < m_num_bins; ++b) {
          data[b] = static_cast<Real_type>(b+1) / m_num_bins;
        }
      } else if (init_random_sizes) {
        allocAndInitDataRandValue(data, m_num_bins, Base_Seq);
        std::sort(data, data+m_num_bins);
      } else if (init_all_one) {
        allocData(data, m_num_bins, Base_Seq);
        for (Index_type b = 0; b < m_num_bins; ++b) {
          data[b] = static_cast<Real_type>(0);
        }
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
