//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#ifndef RAJAPerf_Basic_ADD_4GROUPS_HPP
#define RAJAPerf_Basic_ADD_4GROUPS_HPP




#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class ADD_4GROUPS : public KernelBase
{


public:

  ADD_4GROUPS(const RunParams& params);

  ~ADD_4GROUPS();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void setCudaTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);


private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

  Real_ptr m_a;
  Real_ptr m_b;
  Real_ptr m_c;
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
