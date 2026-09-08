//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// FIR kernel reference implementation:
///
/// #define FIR_COEFFLEN (16)
///
/// Real_type coeff[FIR_COEFFLEN] = { 3.0, -1.0, -1.0, -1.0,
///                                  -1.0, 3.0, -1.0, -1.0,
///                                  -1.0, -1.0, 3.0, -1.0,
///                                  -1.0, -1.0, -1.0, 3.0 };
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   Real_type sum = 0.0;
///   for (Index_type j = 0; j < coefflen; ++j ) {
///     sum += coeff[j]*in[i+j];
///   }
///   out[i] = sum;
/// }
///

#ifndef RAJAPerf_Apps_FIR_HPP
#define RAJAPerf_Apps_FIR_HPP


#define FIR_COEFFLEN (16)

#define FIR_DATA_SETUP \
  Real_ptr in = m_in; \
  Real_ptr out = m_out; \
\
  const Index_type coefflen = m_coefflen;

#define FIR_COEFF \
  Real_type coeff_array[FIR_COEFFLEN] = { 3.0, -1.0, -1.0, -1.0, \
                                         -1.0, 3.0, -1.0, -1.0, \
                                         -1.0, -1.0, 3.0, -1.0, \
                                         -1.0, -1.0, -1.0, 3.0 };

#define FIR_BODY \
  Real_type sum = 0.0; \
\
  for (Index_type j = 0; j < coefflen; ++j ) { \
    sum += coeff[j]*in[i+j]; \
  } \
  out[i] = sum;

#define FIR_BODY_SHD \
  Index_type nthds_last_blk = iend & (block_size - 1); /* block_size=2^int */ \
/* case where last block of grid has < FIR_COEFFLEN but > 0 threads */ \
  if ( blockIdx.x == gridDim.x - 1 && \
       nthds_last_blk < FIR_COEFFLEN && \
       nthds_last_blk > 0 ) { \
    Index_type ntrips_per_thd = (FIR_COEFFLEN - 1) / nthds_last_blk + 1; \
    for (Index_type trip = 0; trip < ntrips_per_thd; trip++) { \
      Index_type j = threadIdx.x + trip * nthds_last_blk; \
      if ( j < FIR_COEFFLEN ) { \
        coeff_shd[j] = coeff[j]; \
      } \
    } \
  } \
  else if (threadIdx.x < FIR_COEFFLEN) { \
    coeff_shd[threadIdx.x] = coeff[threadIdx.x]; \
  } \
  __syncthreads(); \
\
  Real_type sum = 0.0; \
  for (Index_type j = 0; j < coefflen; ++j ) { \
    sum += coeff_shd[j]*in[i+j]; \
  } \
  out[i] = sum;

#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace apps
{

class FIR : public KernelBase
{
public:

  FIR(const RunParams& params);

  ~FIR();

  void setSize(Index_type target_size, Index_type target_reps);
  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void defineSeqVariantTunings();
  void defineOpenMPVariantTunings();
  void defineOpenMPTargetVariantTunings();
  void defineCudaVariantTunings();
  void defineHipVariantTunings();
  void defineSyclVariantTunings();

  void runSeqVariant(VariantID vid);
  void runOpenMPVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);

  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);
  template < size_t work_group_size >
  void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;

  Real_ptr m_in;
  Real_ptr m_out;

  Index_type m_coefflen;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
