//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// FIRST_MIN kernel reference implementation:
///
/// Note: kernel implementation uses a "min-loc" reduction.
///
/// Index_type loc = 0;
/// x[N/2] = -1.0e+10;
/// for (Index_type i = 0; i < N; ++i ) {
///   if ( x[i] < x[loc] ) loc = i;
/// }
///

#ifndef RAJAPerf_Lcals_FIRST_MIN_HPP
#define RAJAPerf_Lcals_FIRST_MIN_HPP

#include "RAJA/util/macros.hpp"

#define FIRST_MIN_DATA_SETUP \
  Real_ptr x = m_x;

#define FIRST_MIN_BODY  \
  if ( x[i] < mymin.val ) { \
    mymin.val = x[i]; \
    mymin.loc = i; \
  }

#define FIRST_MIN_BODY_RAJA  \
  loc.minloc(x[i], i);


#include "common/RPTypes.hpp"

struct MyMinLoc {
  rajaperf::Real_type val;
  rajaperf::Index_type loc;
};


#define FIRST_MIN_MINLOC_INIT \
  MyMinLoc mymin; \
  mymin.val = m_xmin_init; \
  mymin.loc = m_initloc;

inline MyMinLoc MinLoc_compare(MyMinLoc a, MyMinLoc b) {
  return a.val < b.val ? a : b ;
}

#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace lcals
{

class FIRST_MIN : public KernelBase
{
public:

  FIRST_MIN(const RunParams& params);

  ~FIRST_MIN();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runSyclVariant(VariantID vid, size_t tune_idx);

  void runKokkosVariant(VariantID vid, size_t tune_idx);

  void setSeqTuningDefinitions(VariantID vid);
  void setOpenMPTuningDefinitions(VariantID vid);
  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  void setOpenMPTargetTuningDefinitions(VariantID vid);
  void setSyclTuningDefinitions(VariantID vid); 

  template < size_t block_size, typename MappingHelper >
  void runCudaVariantBase(VariantID vid);
  template < size_t block_size, typename MappingHelper >
  void runCudaVariantRAJA(VariantID vid);
  template < size_t block_size, typename MappingHelper >
  void runCudaVariantRAJANewReduce(VariantID vid);

  template < size_t block_size, typename MappingHelper >
  void runHipVariantBase(VariantID vid);
  template < size_t block_size, typename MappingHelper >
  void runHipVariantRAJA(VariantID vid);
  template < size_t block_size, typename MappingHelper >
  void runHipVariantRAJANewReduce(VariantID vid);

  template < size_t work_group_size >
  void runSyclVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = integer::make_gpu_block_size_list_type<default_gpu_block_size>;

  Real_ptr m_x;
  Real_type m_xmin_init;
  Index_type m_initloc;
  Index_type m_minloc;

  Index_type m_N;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
