//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// DYANAMIC_TILE kernel reference implementation:
///
/// for each of three runtime-sized 3D boxes:
///   for (Index_type k = 0; k < nk; ++k ) {
///     for (Index_type j = 0; j < nj; ++j ) {
///       for (Index_type i = 0; i < ni; ++i ) {
///         Index_type idx = offset + i + ni * (j + nj * k);
///         output[idx] = input[idx] +
///                       0.00000001 * (i + 2 * j + 3 * k);
///       }
///     }
///   }
///
/// The three boxes use different runtime dimensions so RAJA::fornest auto
/// tiling can choose different tile shapes. On GPU those tile shapes determine
/// the thread-block/work-group dimensions.
///

#ifndef RAJAPerf_Basic_DYANAMIC_TILE_HPP
#define RAJAPerf_Basic_DYANAMIC_TILE_HPP


#define DYANAMIC_TILE_DATA_SETUP \
  Real_ptr input = m_input; \
  Real_ptr output = m_output; \
  Index_type offset0 = m_offset0; \
  Index_type offset1 = m_offset1; \
  Index_type offset2 = m_offset2; \
  Index_type ni0 = m_ni0; \
  Index_type nj0 = m_nj0; \
  Index_type nk0 = m_nk0; \
  Index_type ni1 = m_ni1; \
  Index_type nj1 = m_nj1; \
  Index_type nk1 = m_nk1; \
  Index_type ni2 = m_ni2; \
  Index_type nj2 = m_nj2; \
  Index_type nk2 = m_nk2; \
  Index_type len0 = m_len0; \
  Index_type len1 = m_len1; \
  Index_type len2 = m_len2;

#define DYANAMIC_TILE_BODY(INDEX_OFFSET, NI, NJ) \
  { \
    Index_type idx = (INDEX_OFFSET) + i + (NI) * (j + (NJ) * k); \
    output[idx] = input[idx] + \
                  static_cast<Real_type>(0.00000001) * \
                      (i + static_cast<Index_type>(2) * j + \
                       static_cast<Index_type>(3) * k); \
  }


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class DYANAMIC_TILE : public KernelBase
{
public:

  DYANAMIC_TILE(const RunParams& params);

  ~DYANAMIC_TILE();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();
  void defineSyclVariantTunings();

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);

  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);
  template < size_t work_group_size >
  void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type =
    integer::make_gpu_block_size_list_type<default_gpu_block_size,
                                           integer::MultipleOf<32>>;

  Real_ptr m_input;
  Real_ptr m_output;

  Index_type m_offset0;
  Index_type m_offset1;
  Index_type m_offset2;

  Index_type m_ni0;
  Index_type m_nj0;
  Index_type m_nk0;
  Index_type m_ni1;
  Index_type m_nj1;
  Index_type m_nk1;
  Index_type m_ni2;
  Index_type m_nj2;
  Index_type m_nk2;

  Index_type m_len0;
  Index_type m_len1;
  Index_type m_len2;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
