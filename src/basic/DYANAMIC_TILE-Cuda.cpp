//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DYANAMIC_TILE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void dyanamic_tile(Real_ptr input, Real_ptr output,
                              Index_type offset,
                              Index_type ni, Index_type nj, Index_type nk)
{
  Index_type flat = blockIdx.x * block_size + threadIdx.x;
  if (flat < ni * nj * nk) {
    Index_type i = flat % ni;
    Index_type j = (flat / ni) % nj;
    Index_type k = flat / (ni * nj);
    DYANAMIC_TILE_BODY(offset, ni, nj);
  }
}



template < size_t block_size >
void DYANAMIC_TILE::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  DYANAMIC_TILE_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    const size_t grid_size0 = RAJA_DIVIDE_CEILING_INT(len0, block_size);
    const size_t grid_size1 = RAJA_DIVIDE_CEILING_INT(len1, block_size);
    const size_t grid_size2 = RAJA_DIVIDE_CEILING_INT(len2, block_size);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_1");
      RPlaunchCudaKernel( (dyanamic_tile<block_size>),
                          grid_size0, block_size,
                          0, res.get_stream(),
                          input, output, offset0, ni0, nj0, nk0 );
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_1");

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_2");
      RPlaunchCudaKernel( (dyanamic_tile<block_size>),
                          grid_size1, block_size,
                          0, res.get_stream(),
                          input, output, offset1, ni1, nj1, nk1 );
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_2");

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_3");
      RPlaunchCudaKernel( (dyanamic_tile<block_size>),
                          grid_size2, block_size,
                          0, res.get_stream(),
                          input, output, offset2, ni2, nj2, nk2 );
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_3");

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    const Index_type ibegin = 0;
    const size_t grid_size0 = RAJA_DIVIDE_CEILING_INT(len0, block_size);
    const size_t grid_size1 = RAJA_DIVIDE_CEILING_INT(len1, block_size);
    const size_t grid_size2 = RAJA_DIVIDE_CEILING_INT(len2, block_size);

    auto flat_body0 = [=] __device__ (Index_type flat) {
      Index_type i = flat % ni0;
      Index_type j = (flat / ni0) % nj0;
      Index_type k = flat / (ni0 * nj0);
      DYANAMIC_TILE_BODY(offset0, ni0, nj0);
    };
    auto flat_body1 = [=] __device__ (Index_type flat) {
      Index_type i = flat % ni1;
      Index_type j = (flat / ni1) % nj1;
      Index_type k = flat / (ni1 * nj1);
      DYANAMIC_TILE_BODY(offset1, ni1, nj1);
    };
    auto flat_body2 = [=] __device__ (Index_type flat) {
      Index_type i = flat % ni2;
      Index_type j = (flat / ni2) % nj2;
      Index_type k = flat / (ni2 * nj2);
      DYANAMIC_TILE_BODY(offset2, ni2, nj2);
    };

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_1");
      RPlaunchCudaKernel( (lambda_cuda_forall<block_size,
                                              decltype(flat_body0)>),
                          grid_size0, block_size,
                          0, res.get_stream(),
                          ibegin, len0, flat_body0 );
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_1");

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_2");
      RPlaunchCudaKernel( (lambda_cuda_forall<block_size,
                                              decltype(flat_body1)>),
                          grid_size1, block_size,
                          0, res.get_stream(),
                          ibegin, len1, flat_body1 );
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_2");

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_3");
      RPlaunchCudaKernel( (lambda_cuda_forall<block_size,
                                              decltype(flat_body2)>),
                          grid_size2, block_size,
                          0, res.get_stream(),
                          ibegin, len2, flat_body2 );
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_3");

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    using EXEC_POL =
      RAJA::fornest_tiling_policy<RAJA::cuda_exec<block_size, true /*async*/>,
                                  RAJA::fornest_tile_auto,
                                  RAJA::fornest_tile_auto,
                                  RAJA::fornest_tile_auto>;

    auto body0 = [=] __device__ (Index_type k, Index_type j, Index_type i) {
      DYANAMIC_TILE_BODY(offset0, ni0, nj0);
    };
    auto body1 = [=] __device__ (Index_type k, Index_type j, Index_type i) {
      DYANAMIC_TILE_BODY(offset1, ni1, nj1);
    };
    auto body2 = [=] __device__ (Index_type k, Index_type j, Index_type i) {
      DYANAMIC_TILE_BODY(offset2, ni2, nj2);
    };

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_1");
      RAJA::fornest(res, EXEC_POL {},
                    RAJA::range(nk0), RAJA::range(nj0), RAJA::range(ni0),
                    body0);
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_1");

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_2");
      RAJA::fornest(res, EXEC_POL {},
                    RAJA::range(nk1), RAJA::range(nj1), RAJA::range(ni1),
                    body1);
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_2");

      RP_CALI_SUBKERNEL_BEGIN("DYANAMIC_TILE_3");
      RAJA::fornest(res, EXEC_POL {},
                    RAJA::range(nk2), RAJA::range(nj2), RAJA::range(ni2),
                    body2);
      RP_CALI_SUBKERNEL_END("DYANAMIC_TILE_3");

    }
    stopTimer();

  } else {
     getCout() << "\n  DYANAMIC_TILE : Unknown Cuda variant id = " << vid << std::endl;
  }
}

void DYANAMIC_TILE::defineCudaVariantTunings()
{
  for (VariantID vid : {Base_CUDA, Lambda_CUDA, RAJA_CUDA}) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (vid == RAJA_CUDA) {
          addVariantTuning<&DYANAMIC_TILE::runCudaVariantImpl<block_size>>(
              vid, "fornest-auto-tile_block_"+std::to_string(block_size));
        } else {
          addVariantTuning<&DYANAMIC_TILE::runCudaVariantImpl<block_size>>(
              vid, "block_"+std::to_string(block_size));
        }

      }

    });

  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
