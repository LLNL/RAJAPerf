//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define REDUCE3_INT_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(vec, m_vec, iend);

#define REDUCE3_INT_DATA_TEARDOWN_CUDA \
  deallocCudaDeviceData(vec);

#define REDUCE3_INT_BODY_CUDA(atomicAdd, atomicMin, atomicMax) \
  RAJAPERF_REDUCE_3_CUDA(Int_type, REDUCE3_INT_VALS, vsum, vsum_init, RAJAPERF_ADD_OP, atomicAdd, \
                                                     vmin, vmin_init, RAJAPERF_MIN_OP, atomicMin, \
                                                     vmax, vmax_init, RAJAPERF_MAX_OP, atomicMax)

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce3int(Int_ptr vec,
                           Int_ptr vsum, Int_type vsum_init,
                           Int_ptr vmin, Int_type vmin_init,
                           Int_ptr vmax, Int_type vmax_init,
                           Index_type iend)
{
  REDUCE3_INT_BODY_CUDA(::atomicAdd,
                        ::atomicMin,
                        ::atomicMax)
}



template < size_t block_size >
void REDUCE3_INT::runCudaVariantReduceAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  REDUCE3_INT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    REDUCE3_INT_DATA_SETUP_CUDA;

    Int_ptr vmem_init;
    allocCudaPinnedData(vmem_init, 3);

    Int_ptr vmem;
    allocCudaDeviceData(vmem, 3);
    Int_ptr vsum = vmem + 0;
    Int_ptr vmin = vmem + 1;
    Int_ptr vmax = vmem + 2;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      vmem_init[0] = vsum_init;
      vmem_init[1] = vmin_init;
      vmem_init[2] = vmax_init;
      cudaErrchk( cudaMemcpyAsync( vmem, vmem_init, 3*sizeof(Int_type),
                                   cudaMemcpyHostToDevice ) );

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      reduce3int<block_size><<<grid_size, block_size,
                   3*sizeof(Int_type)*block_size>>>(vec,
                                                    vsum, vsum_init,
                                                    vmin, vmin_init,
                                                    vmax, vmax_init,
                                                    iend );
      cudaErrchk( cudaGetLastError() );

      Int_type lmem[3];
      Int_ptr plmem = &lmem[0];
      getCudaDeviceData(plmem, vmem, 3);
      m_vsum += lmem[0];
      m_vmin = RAJA_MIN(m_vmin, lmem[1]);
      m_vmax = RAJA_MAX(m_vmax, lmem[2]);

    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_CUDA;

    deallocCudaDeviceData(vmem);
    deallocCudaPinnedData(vmem_init);

  } else if ( vid == Lambda_CUDA ) {

    REDUCE3_INT_DATA_SETUP_CUDA;

    Int_ptr vmem_init;
    allocCudaPinnedData(vmem_init, 3);

    Int_ptr vmem;
    allocCudaDeviceData(vmem, 3);
    Int_ptr vsum = vmem + 0;
    Int_ptr vmin = vmem + 1;
    Int_ptr vmax = vmem + 2;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      vmem_init[0] = vsum_init;
      vmem_init[1] = vmin_init;
      vmem_init[2] = vmax_init;
      cudaErrchk( cudaMemcpyAsync( vmem, vmem_init, 3*sizeof(Int_type),
                                   cudaMemcpyHostToDevice ) );

      auto reduce3int_lambda = [=] __device__ () {
        REDUCE3_INT_BODY_CUDA(::atomicAdd,
                              ::atomicMin,
                              ::atomicMax)
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda<block_size><<<grid_size, block_size,
                   3*sizeof(Int_type)*block_size>>>(reduce3int_lambda);
      cudaErrchk( cudaGetLastError() );

      Int_type lmem[3];
      Int_ptr plmem = &lmem[0];
      getCudaDeviceData(plmem, vmem, 3);
      m_vsum += lmem[0];
      m_vmin = RAJA_MIN(m_vmin, lmem[1]);
      m_vmax = RAJA_MAX(m_vmax, lmem[2]);

    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_CUDA;

    deallocCudaDeviceData(vmem);
    deallocCudaPinnedData(vmem_init);

  } else {
     getCout() << "\n  REDUCE3_INT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void REDUCE3_INT::runCudaVariantReduce(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE3_INT_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    REDUCE3_INT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Int_type> vsum(vsum_init);
      RAJA::ReduceMin<RAJA::cuda_reduce, Int_type> vmin(vmin_init);
      RAJA::ReduceMax<RAJA::cuda_reduce, Int_type> vmax(vmax_init);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        REDUCE3_INT_BODY_RAJA;
      });

      m_vsum += static_cast<Int_type>(vsum.get());
      m_vmin = RAJA_MIN(m_vmin, static_cast<Int_type>(vmin.get()));
      m_vmax = RAJA_MAX(m_vmax, static_cast<Int_type>(vmax.get()));

    }
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  REDUCE3_INT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

void REDUCE3_INT::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;
  if (vid == Base_CUDA || vid == Lambda_CUDA) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        if (tune_idx == t) {
          runCudaVariantReduceAtomic<block_size>(vid);
        }
        t += 1;
      }
    });
  } else if ( vid == RAJA_CUDA ) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        if (tune_idx == t) {
          runCudaVariantReduce<block_size>(vid);
        }
        t += 1;
      }
    });
  }
}

void REDUCE3_INT::setCudaTuningDefinitions(VariantID vid)
{
  if (vid == Base_CUDA || vid == Lambda_CUDA) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        addVariantTuningName(vid, "reduceAtomic_"+std::to_string(block_size));
      }
    });
  } else if ( vid == RAJA_CUDA ) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        addVariantTuningName(vid, "reduce_"+std::to_string(block_size));
      }
    });
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
