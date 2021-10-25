//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Methods and classes for GPU kernel templates.
///


#ifndef RAJAPerf_GPUUtils_HPP
#define RAJAPerf_GPUUtils_HPP

#include "rajaperf_config.hpp"


namespace rajaperf
{

namespace gpu_block_size
{

namespace detail
{

// helpers to invoke f with each integer in the param pack
template < typename F >
bool invoke_or_helper(F)
{
  return false;
}
///
template < typename F, size_t I, size_t... Is>
bool invoke_or_helper(F f)
{
  return f.template operator()<I>() || invoke_or_helper<F, Is...>(f);
}

// class to get the size of a camp::int_seq
template < typename IntSeq >
struct SizeOfIntSeq;
///
template < size_t... Is >
struct SizeOfIntSeq<camp::int_seq<size_t, Is...>>
{
   static const size_t size = sizeof...(Is);
};

} // namespace detail

// call f's call operator with each integer as the template param in turn
// stopping at the first integer that returns true.
// return true if any f<I>() returns true, otherwise return false
template < typename F, size_t... Is >
bool invoke_or(F f, camp::int_seq<size_t, Is...>)
{
  return detail::invoke_or_helper<F, Is...>(f);
}

// if the given integer is the same as the template param block_size
// returns true otherwise returns false
struct Equals
{
  Equals(size_t actual_gpu_block_size)
    : m_actual_gpu_block_size(actual_gpu_block_size)
  {}

  template < size_t block_size >
  bool operator()() { return m_actual_gpu_block_size == block_size; }

private:
  size_t m_actual_gpu_block_size;
};

// if the kernel's actual block size is the same as the template param
// runs the cuda variant with the template param block_size and returns true
// otherwise returns false
template < typename Kernel >
struct RunCudaBlockSize
{
  RunCudaBlockSize(Kernel& kernel, VariantID vid)
    : m_kernel(kernel), m_vid(vid)
  {}

  template < size_t block_size >
  bool operator()() {
    if (block_size == m_kernel.getActualGPUBlockSize()) {
      m_kernel.template runCudaVariantImpl<block_size>(m_vid);
      return true;
    }
    return false;
  }

private:
  Kernel& m_kernel;
  VariantID m_vid;
};

// if the kernel's actual block size is the same as the template param
// runs the hip variant with the template param block_size and returns true
// otherwise returns false
template < typename Kernel >
struct RunHipBlockSize
{
  RunHipBlockSize(Kernel& kernel, VariantID vid)
    : m_kernel(kernel), m_vid(vid)
  {}

  template < size_t block_size >
  bool operator()() {
    if (block_size == m_kernel.getActualGPUBlockSize()) {
      m_kernel.template runHipVariantImpl<block_size>(m_vid);
      return true;
    }
    return false;
  }

private:
  Kernel& m_kernel;
  VariantID m_vid;
};

// return default_I if it is in sizes or the first integer in sizes otherwise
template < size_t I, size_t... Is >
size_t get_default_or_first(size_t default_I, camp::int_seq<size_t, I, Is...> sizes)
{
  if (invoke_or(Equals(default_I), sizes)) {
    return default_I;
  }
  return I;
}

// A camp::int_seq of size_t's that is rajaperf::configuration::gpu_block_sizes
// if rajaperf::configuration::gpu_block_sizes is not empty
// and a camp::int_seq of default_block_size otherwise
template < size_t default_block_size >
using list_type =
      typename std::conditional< (detail::SizeOfIntSeq<rajaperf::configuration::gpu_block_sizes>::size > 0),
                                 rajaperf::configuration::gpu_block_sizes,
                                 camp::int_seq<size_t, default_block_size>
                               >::type;

} // closing brace for gpu_block_size namespace

} // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
