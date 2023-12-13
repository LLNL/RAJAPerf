//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
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

// class to help prepend integers to a list
// this is used for the false case where I is not prepended to List
template < bool B, typename T, typename List >
struct conditional_prepend
{
  using type = List;
};
/// this is used for the true case where I is prepended to List
template < typename T, typename... Ts >
struct conditional_prepend<true, T, camp::list<Ts...>>
{
  using type = camp::list<T, Ts...>;
};

// class to help create a sequence that is only the valid values in List
template < typename validity_checker, typename List >
struct remove_invalid;

// base case where the list is empty, use the empty list
template < typename validity_checker >
struct remove_invalid<validity_checker, camp::list<>>
{
  using type = camp::list<>;
};

// check validity of T and conditionally prepend T to a recursively generated
// list of valid values
template < typename validity_checker, typename T, typename... Ts >
struct remove_invalid<validity_checker, camp::list<T, Ts...>>
{
  using type = typename conditional_prepend<
      validity_checker::valid(T{}),
      T,
      typename remove_invalid<validity_checker, camp::list<Ts...>>::type
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
  static constexpr bool valid(size_t RAJAPERF_UNUSED_ARG(i)) { return true; }
};

// true if of i is a multiple of N, false otherwise
template < size_t N >
struct MultipleOf
{
  static constexpr bool valid(size_t i) { return (i/N)*N == i; }
};

// true if the sqrt of i is representable as a size_t, false otherwise
struct ExactSqrt
{
  static constexpr bool valid(size_t i) { return sqrt(i)*sqrt(i) == i; }
};

// A camp::list of camp::integral_constant<size_t, I> types.
// If gpu_block_sizes from the configuration is not empty it is those gpu_block_sizes,
// otherwise it is a list containing just default_block_size.
// Invalid entries are removed according to validity_checker in either case.
template < size_t default_block_size, typename validity_checker = AllowAny >
using make_list_type =
      typename detail::remove_invalid<validity_checker,
        typename std::conditional< (camp::size<rajaperf::configuration::gpu_block_sizes>::value > 0),
          rajaperf::configuration::gpu_block_sizes,
          list_type<default_block_size>
        >::type
      >::type;

} // closing brace for gpu_block_size namespace

} // closing brace for rajaperf namespace

//
#define RAJAPERF_GPU_REDUCER_SETUP_IMPL(pointer_type, device_ptr_name, host_ptr_name, length) \
  DataSpace reduction_data_space = getReductionDataSpace(vid);                 \
  DataSpace host_data_space = hostAccessibleDataSpace(reduction_data_space);   \
                                                                               \
  pointer_type device_ptr_name;                                                \
  allocData(reduction_data_space, device_ptr_name, (length));                  \
  pointer_type host_ptr_name = device_ptr_name;                                \
  if (reduction_data_space != host_data_space) {                               \
    allocData(host_data_space, host_ptr_name, (length));                       \
  }

//
#define RAJAPERF_GPU_REDUCER_TEARDOWN_IMPL(device_ptr_name, host_ptr_name)          \
  deallocData(reduction_data_space, device_ptr_name);                          \
  if (reduction_data_space != host_data_space) {                               \
    deallocData(host_data_space, host_ptr_name);                               \
  }

//
#define RAJAPERF_GPU_REDUCER_INITIALIZE_IMPL(gpu_type, init_ptr, device_ptr_name, host_ptr_name, length) \
  if (device_ptr_name != host_ptr_name) {                                      \
    for (int i = 0; i < (length); ++i) {                                       \
      host_ptr_name[i] = (init_ptr)[i];                                    \
    }                                                                          \
    gpu_type##Errchk( gpu_type##MemcpyAsync( device_ptr_name, host_ptr_name,   \
        (length)*sizeof(device_ptr_name[0]),                                   \
        gpu_type##MemcpyHostToDevice, res.get_stream() ) );                    \
  } else {                                                                     \
    for (int i = 0; i < (length); ++i) {                                       \
      device_ptr_name[i] = (init_ptr)[i];                                  \
    }                                                                          \
  }

//
#define RAJAPERF_GPU_REDUCER_COPY_BACK_IMPL(gpu_type, final_ptr, device_ptr_name, host_ptr_name, length) \
  if (device_ptr_name != host_ptr_name) {                                      \
    gpu_type##Errchk( gpu_type##MemcpyAsync( host_ptr_name, device_ptr_name,   \
        (length)*sizeof(device_ptr_name[0]),                                   \
        gpu_type##MemcpyDeviceToHost, res.get_stream() ) );                    \
  }                                                                            \
  gpu_type##Errchk( gpu_type##StreamSynchronize( res.get_stream() ) );         \
  for (int i = 0; i < (length); ++i) {                                         \
    (final_ptr)[i] = host_ptr_name[i];                                        \
  }


#define RAJAPERF_CUDA_REDUCER_SETUP(pointer_type, device_ptr_name, host_ptr_name, length) \
  RAJAPERF_GPU_REDUCER_SETUP_IMPL(pointer_type, device_ptr_name, host_ptr_name, length)
#define RAJAPERF_CUDA_REDUCER_TEARDOWN(device_ptr_name, host_ptr_name)          \
  RAJAPERF_GPU_REDUCER_TEARDOWN_IMPL(device_ptr_name, host_ptr_name)
#define RAJAPERF_CUDA_REDUCER_INITIALIZE(init_ptr, device_ptr_name, host_ptr_name, length) \
  RAJAPERF_GPU_REDUCER_INITIALIZE_IMPL(cuda, init_ptr, device_ptr_name, host_ptr_name, length)
#define RAJAPERF_CUDA_REDUCER_COPY_BACK(final_ptr, device_ptr_name, host_ptr_name, length) \
  RAJAPERF_GPU_REDUCER_COPY_BACK_IMPL(cuda, final_ptr, device_ptr_name, host_ptr_name, length)

#define RAJAPERF_HIP_REDUCER_SETUP(pointer_type, device_ptr_name, host_ptr_name, length) \
  RAJAPERF_GPU_REDUCER_SETUP_IMPL(pointer_type, device_ptr_name, host_ptr_name, length)
#define RAJAPERF_HIP_REDUCER_TEARDOWN(device_ptr_name, host_ptr_name)          \
  RAJAPERF_GPU_REDUCER_TEARDOWN_IMPL(device_ptr_name, host_ptr_name)
#define RAJAPERF_HIP_REDUCER_INITIALIZE(init_ptr, device_ptr_name, host_ptr_name, length) \
  RAJAPERF_GPU_REDUCER_INITIALIZE_IMPL(hip, init_ptr, device_ptr_name, host_ptr_name, length)
#define RAJAPERF_HIP_REDUCER_COPY_BACK(final_ptr, device_ptr_name, host_ptr_name, length) \
  RAJAPERF_GPU_REDUCER_COPY_BACK_IMPL(hip, final_ptr, device_ptr_name, host_ptr_name, length)

//
#define RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(kernel, variant)     \
  void kernel::run##variant##Variant(VariantID vid, size_t tune_idx)           \
  {                                                                            \
    size_t t = 0;                                                              \
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {                     \
      if (run_params.numValidGPUBlockSize() == 0u ||                           \
          run_params.validGPUBlockSize(block_size)) {                          \
        if (tune_idx == t) {                                                   \
          setBlockSize(block_size);                                            \
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
