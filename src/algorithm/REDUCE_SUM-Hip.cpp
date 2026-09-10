//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_SUM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#if defined(__HIPCC__)
#define ROCPRIM_HIP_API 1
#include "rocprim/device/device_reduce.hpp"
#elif defined(__CUDACC__)
#include "cub/device/device_reduce.cuh"
#include "cub/util_allocator.cuh"
#endif

#include "common/HipDataUtils.hpp"

#include <iostream>
#include <utility>
#include <type_traits>
#include <limits>


namespace camp
{

namespace experimental
{
#if defined(__HIPCC__)
template<typename R>
struct StreamInsertHelper<::rocprim::plus<R>&>
{
  ::rocprim::plus<R>& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "::rocprim::plus";
  }
};
///
template<typename R>
struct StreamInsertHelper<::rocprim::plus<R> const&>
{
  ::rocprim::plus<R> const& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "::rocprim::plus";
  }
};
#elif defined(__CUDACC__)
template<>
struct StreamInsertHelper<::cub::Sum&>
{
  ::cub::Sum& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "::cub::Sum";
  }
};
///
template<>
struct StreamInsertHelper<::cub::Sum const&>
{
  ::cub::Sum const& m_val;

  std::ostream& operator()(std::ostream& str) const
  {
    return str << "::cub::Sum";
  }
};
#endif

}  // closing brace for experimental namespace

}  // closing brace for camp namespace


namespace rajaperf
{
namespace algorithm
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum(Real_ptr x, Real_ptr sum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, psum);

  Index_type i = blockIdx.x * block_size + threadIdx.x;

  psum[ threadIdx.x ] = sum_init;
  for ( ; i < iend ; i += gridDim.x * block_size ) {
    psum[ threadIdx.x ] += x[i];
  }
  __syncthreads();

  for ( i = block_size / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      psum[ threadIdx.x ] += psum[ threadIdx.x + i ];
    }
     __syncthreads();
  }

  if ( threadIdx.x == 0 ) {
    RAJAPERF_ATOMIC_ADD_HIP( *sum, psum[ 0 ] );
  }
}


void REDUCE_SUM::runHipVariantRocprim(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    hipStream_t stream = res.get_stream();

    int len = iend - ibegin;

    RAJAPERF_HIP_REDUCER_SETUP(Real_ptr, sum, hsum, 1, 1);

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
    CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::reduce,
        d_temp_storage, temp_storage_bytes,
        x+ibegin,
        sum,
        m_sum_init,
        len,
        ::rocprim::plus<Real_type>(),
        stream);
#elif defined(__CUDACC__)
    CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceReduce::Reduce,
        d_temp_storage, temp_storage_bytes,
        x+ibegin,
        sum,
        len,
        ::cub::Sum(),
        m_sum_init,
        stream);
#endif

    // Allocate temporary storage
    unsigned char* temp_storage;
    allocData(DataSpace::HipDevice, temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;


    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("REDUCE_SUM_1");
      // Run
#if defined(__HIPCC__)
      CAMP_HIP_API_INVOKE_AND_CHECK(::rocprim::reduce,
          d_temp_storage, temp_storage_bytes,
          x+ibegin,
          sum,
          m_sum_init,
          len,
          ::rocprim::plus<Real_type>(),
          stream);
#elif defined(__CUDACC__)
      CAMP_CUDA_API_INVOKE_AND_CHECK(::cub::DeviceReduce::Reduce,
          d_temp_storage, temp_storage_bytes,
          x+ibegin,
          sum,
          len,
          ::cub::Sum(),
          m_sum_init,
          stream);
#endif

      RAJAPERF_HIP_REDUCER_COPY_BACK(sum, hsum, 1, 1);
      m_sum = hsum[0];
      RP_CALI_SUBKERNEL_END("REDUCE_SUM_1");

    }
    stopTimer();

    // Free temporary storage
    deallocData(DataSpace::HipDevice, temp_storage);
    RAJAPERF_HIP_REDUCER_TEARDOWN(sum, hsum);

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Hip variant id = " << vid << std::endl;

  }

}

template < size_t block_size, typename MappingHelper >
void REDUCE_SUM::runHipVariantBase(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    RAJAPERF_HIP_REDUCER_SETUP(Real_ptr, sum, hsum, 1, 1);

    constexpr size_t shmem = sizeof(Real_type)*block_size;
    const size_t max_grid_size = RAJAPERF_HIP_GET_MAX_BLOCKS(
        MappingHelper, (reduce_sum<block_size>), block_size, shmem);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("REDUCE_SUM_1");
      RAJAPERF_HIP_REDUCER_INITIALIZE(&m_sum_init, sum, hsum, 1, 1);

      const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      const size_t grid_size = std::min(normal_grid_size, max_grid_size);

      RPlaunchHipKernel( (reduce_sum<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         x, sum, m_sum_init, iend );

      RAJAPERF_HIP_REDUCER_COPY_BACK(sum, hsum, 1, 1);
      m_sum = hsum[0];
      RP_CALI_SUBKERNEL_END("REDUCE_SUM_1");

    }
    stopTimer();

    RAJAPERF_HIP_REDUCER_TEARDOWN(sum, hsum);

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Hip variant id = " << vid << std::endl;

  }

}

template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
void REDUCE_SUM::runHipVariantRAJA(VariantID vid)
{
  setBlockSize(block_size);

  using reduction_policy = std::conditional_t<AlgorithmHelper::atomic,
      RAJA::hip_reduce_atomic,
      RAJA::hip_reduce>;

  using exec_policy = std::conditional_t<MappingHelper::direct,
      RAJA::hip_exec<block_size, true /*async*/>,
      RAJA::hip_exec_occ_calc<block_size, true /*async*/>>;

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  REDUCE_SUM_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("REDUCE_SUM_1");
      RAJA::ReduceSum<reduction_policy, Real_type> sum(m_sum_init);

      RAJA::forall<exec_policy>( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          REDUCE_SUM_BODY;
      });

      m_sum = sum.get();
      RP_CALI_SUBKERNEL_END("REDUCE_SUM_1");

    }
    stopTimer();

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Hip variant id = " << vid << std::endl;

  }

}

template < size_t block_size, typename MappingHelper >
void REDUCE_SUM::runHipVariantRAJANewReduce(VariantID vid)
{
  setBlockSize(block_size);

  using exec_policy = std::conditional_t<MappingHelper::direct,
      RAJA::hip_exec<block_size, true /*async*/>,
      RAJA::hip_exec_occ_calc<block_size, true /*async*/>>;

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  REDUCE_SUM_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("REDUCE_SUM_1");
      Real_type tsum = m_sum_init;

      RAJA::forall<exec_policy>( res,
        RAJA::RangeSegment(ibegin, iend),
        RAJA::expt::Reduce<RAJA::operators::plus>(&tsum),
        [=] __device__ (Index_type i,
          RAJA::expt::ValOp<Real_type, RAJA::operators::plus>& sum) {
          REDUCE_SUM_BODY;
        }
      );

      m_sum = static_cast<Real_type>(tsum);
      RP_CALI_SUBKERNEL_END("REDUCE_SUM_1");

    }
    stopTimer();

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Hip variant id = " << vid << std::endl;

  }

}


void REDUCE_SUM::defineHipVariantTunings()
{
  for (VariantID vid : {Base_HIP, RAJA_HIP}) {

    if ( vid == Base_HIP ) {

#if defined(__HIPCC__)
      addVariantTuning<&REDUCE_SUM::runHipVariantRocprim>(
          vid, "rocprim");
#elif defined(__CUDACC__)
      addVariantTuning<&REDUCE_SUM::runHipVariantRocprim>(
          vid, "cub");
#endif

    }

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_mapping::reducer_helpers{}, [&](auto mapping_helper) {

          if ( vid == Base_HIP ) {

            auto algorithm_helper = gpu_algorithm::block_atomic_helper{};

            addVariantTuning<&REDUCE_SUM::runHipVariantBase<
                                 decltype(block_size){},
                                 decltype(mapping_helper)>>(
                vid, decltype(algorithm_helper)::get_name()+"_"+
                    decltype(mapping_helper)::get_name()+"_"+
                    std::to_string(block_size));

          } else if ( vid == RAJA_HIP ) {

            seq_for(gpu_algorithm::reducer_helpers{}, [&](auto algorithm_helper) {

              addVariantTuning<&REDUCE_SUM::runHipVariantRAJA<
                                   decltype(block_size){},
                                   decltype(algorithm_helper),
                                   decltype(mapping_helper)>>(
                  vid, decltype(algorithm_helper)::get_name()+"_"+
                      decltype(mapping_helper)::get_name()+"_"+
                      std::to_string(block_size));

            });

            auto algorithm_helper = gpu_algorithm::block_device_helper{};

            addVariantTuning<&REDUCE_SUM::runHipVariantRAJANewReduce<
                                 decltype(block_size){},
                                 decltype(mapping_helper)>>(
                vid, decltype(algorithm_helper)::get_name()+"_"+
                    decltype(mapping_helper)::get_name()+"_"+
                    "new_"+std::to_string(block_size));
            RAJA_UNUSED_VAR(algorithm_helper); // to quiet compiler warning

          }

        });

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
