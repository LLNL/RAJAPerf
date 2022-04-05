//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
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

// implementation of sqrt via binary search
// copied from https://stackoverflow.com/questions/8622256/in-c11-is-sqrt-defined-as-constexpr
constexpr size_t sqrt_helper(size_t n, size_t lo, size_t hi)
{
  return (lo == hi)
           ? lo // search complete
           : ((n / ((lo + hi + 1) / 2) < ((lo + hi + 1) / 2))
                ? sqrt_helper(n, lo, ((lo + hi + 1) / 2)-1) // search lower half
                : sqrt_helper(n, ((lo + hi + 1) / 2), hi)); // search upper half
}

// implementation of lesser_of_squarest_factor_pair via linear search
constexpr size_t lesser_of_squarest_factor_pair_helper(size_t n, size_t guess)
{
  return ((n / guess) * guess == n)
           ? guess // search complete, guess is a factor
           : lesser_of_squarest_factor_pair_helper(n, guess - 1); // continue searching
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

// class to help prepend integers to a list
// this is used for the false case where I is not prepended to IntSeq
template < bool B, size_t I, typename IntSeq >
struct conditional_prepend
{
  using type = IntSeq;
};
/// this is used for the true case where I is prepended to IntSeq
template < size_t I, size_t... Is >
struct conditional_prepend<true, I, camp::int_seq<size_t, Is...>>
{
  using type = camp::int_seq<size_t, I, Is...>;
};

// class to help create a sequence that is only the valid values in IntSeq
template < typename validity_checker, typename IntSeq >
struct remove_invalid;

// base case where the list is empty, use the empty list
template < typename validity_checker >
struct remove_invalid<validity_checker, camp::int_seq<size_t>>
{
  using type = camp::int_seq<size_t>;
};

// check validity of I and conditionally prepend I to a recursively generated
// list of valid values
template < typename validity_checker, size_t I, size_t... Is >
struct remove_invalid<validity_checker, camp::int_seq<size_t, I, Is...>>
{
  using type = typename conditional_prepend<
      validity_checker::template valid<I>(),
      I,
      typename remove_invalid<validity_checker, camp::int_seq<size_t, Is...>>::type
    >::type;
};

} // namespace detail

// constexpr integer sqrt
constexpr size_t sqrt(size_t n)
{
  return detail::sqrt_helper(n, 0, n/2 + 1);
}

// constexpr return the lesser of the most square pair of factors of n
// ex. 12 has pairs of factors (1, 12) (2, 6) *(3, 4)* and returns 3
constexpr size_t lesser_of_squarest_factor_pair(size_t n)
{
  return (n == 0)
      ? 0 // return 0 in the 0 case
      : detail::lesser_of_squarest_factor_pair_helper(n, sqrt(n));
}
// constexpr return the greater of the most square pair of factors of n
// ex. 12 has pairs of factors (1, 12) (2, 6) *(3, 4)* and returns 4
constexpr size_t greater_of_squarest_factor_pair(size_t n)
{
  return (n == 0)
      ? 0 // return 0 in the 0 case
      : n / detail::lesser_of_squarest_factor_pair_helper(n, sqrt(n));
}

// always true
struct AllowAny
{
  template < size_t I >
  static constexpr bool valid() { return true; }
};

// true if of I is a multiple of N, false otherwise
template < size_t N >
struct MultipleOf
{
  template < size_t I >
  static constexpr bool valid() { return (I/N)*N == I; }
};

// true if the sqrt of I is representable as a size_t, false otherwise
struct ExactSqrt
{
  template < size_t I >
  static constexpr bool valid() { return sqrt(I)*sqrt(I) == I; }
};

template < size_t... block_sizes >
using list_type = camp::int_seq<size_t, block_sizes...>;

// A camp::int_seq of size_t's that is rajaperf::configuration::gpu_block_sizes
// if rajaperf::configuration::gpu_block_sizes is not empty
// and a camp::int_seq of default_block_size otherwise
// with invalid entries removed according to validity_checker
template < size_t default_block_size, typename validity_checker = AllowAny >
using make_list_type =
      typename detail::remove_invalid<validity_checker,
        typename std::conditional< (detail::SizeOfIntSeq<rajaperf::configuration::gpu_block_sizes>::size > 0),
          rajaperf::configuration::gpu_block_sizes,
          list_type<default_block_size>
        >::type
      >::type;

} // closing brace for gpu_block_size namespace

//compile time loop over an integer sequence
//this allows for creating a loop over a compile time constant variable
template <typename Func, typename T, T... ts>
inline void seq_for(camp::int_seq<T, ts...> const&, Func&& func)
{
  // braced init lists are evaluated in order
  int seq_unused_array[] = {(func(camp::integral_constant<T,ts>{}), 0)...};
  RAJAPERF_UNUSED_VAR(seq_unused_array);
}

} // closing brace for rajaperf namespace

//
#define RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(kernel, variant)     \
  void kernel::run##variant##Variant(VariantID vid, size_t tune_idx)           \
  {                                                                            \
    size_t t = 0;                                                              \
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {                     \
      if (run_params.numValidGPUBlockSize() == 0u ||                           \
          run_params.validGPUBlockSize(block_size)) {                          \
        if (tune_idx == t) {                                                   \
          run##variant##VariantImpl<block_size>(vid);                          \
        }                                                                      \
        t += 1;                                                                \
      }                                                                        \
    });                                                                        \
  }                                                                            \
                                                                               \
  void kernel::set##variant##TuningDefinitions(VariantID vid)                  \
  {                                                                            \
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {                     \
      if (run_params.numValidGPUBlockSize() == 0u ||                           \
          run_params.validGPUBlockSize(block_size)) {                          \
        addVariantTuningName(vid, "block_"+std::to_string(block_size));        \
      }                                                                        \
    });                                                                        \
  }

#endif  // closing endif for header file include guard
