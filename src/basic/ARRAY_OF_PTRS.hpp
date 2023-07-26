//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// ARRAY_OF_PTRS kernel reference implementation:
///
/// // Use a runtime sized portion of an array
/// Index_type array_size;
/// Real_ptr x[ARRAY_OF_PTRS_MAX_ARRAY_SIZE];
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   y[i] = 0.0;
///   for (Index_type a = 0; a < array_size; ++a) {
///     y[i] += x[a][i] ;
///   }
/// }
///

#ifndef RAJAPerf_Basic_ARRAY_OF_PTRS_HPP
#define RAJAPerf_Basic_ARRAY_OF_PTRS_HPP

#define ARRAY_OF_PTRS_MAX_ARRAY_SIZE 26

#define ARRAY_OF_PTRS_DATA_SETUP_X_ARRAY \
  for (Index_type a = 0; a < array_size; ++a) { \
    x[a] = x_data + a * iend ; \
  }

#define ARRAY_OF_PTRS_DATA_SETUP \
  Index_type array_size = m_array_size; \
  Real_ptr y = m_y; \
  Real_ptr x_data = m_x; \
  Real_ptr x[ARRAY_OF_PTRS_MAX_ARRAY_SIZE]; \
  ARRAY_OF_PTRS_DATA_SETUP_X_ARRAY

#define ARRAY_OF_PTRS_BODY(x) \
  Real_type yi = 0.0; \
  for (Index_type a = 0; a < array_size; ++a) { \
    yi += (x)[a][i] ; \
  } \
  y[i] = yi;


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class ARRAY_OF_PTRS : public KernelBase
{
public:

  ARRAY_OF_PTRS(const RunParams& params);

  ~ARRAY_OF_PTRS();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runKokkosVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

  Real_ptr m_x;
  Real_ptr m_y;
  Index_type m_array_size;
};

struct ARRAY_OF_PTRS_Array {
  Real_ptr array[ARRAY_OF_PTRS_MAX_ARRAY_SIZE];

  template < size_t ... Indices >
  ARRAY_OF_PTRS_Array(Real_ptr (&array_)[ARRAY_OF_PTRS_MAX_ARRAY_SIZE],
                      camp::int_seq<size_t, Indices...>)
    : array{array_[Indices]...}
  { }

  ARRAY_OF_PTRS_Array(Real_ptr (&array_)[ARRAY_OF_PTRS_MAX_ARRAY_SIZE])
    : ARRAY_OF_PTRS_Array(array_, camp::make_int_seq_t<size_t, 26>{})
  { }
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
