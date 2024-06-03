//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MATVEC_3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void matvec_3d(Real_ptr b,
                          Real_ptr xdbl,
                          Real_ptr xdbc,
                          Real_ptr xdbr,
                          Real_ptr xdcl,
                          Real_ptr xdcc,
                          Real_ptr xdcr,
                          Real_ptr xdfl,
                          Real_ptr xdfc,
                          Real_ptr xdfr,
                          Real_ptr xcbl,
                          Real_ptr xcbc,
                          Real_ptr xcbr,
                          Real_ptr xccl,
                          Real_ptr xccc,
                          Real_ptr xccr,
                          Real_ptr xcfl,
                          Real_ptr xcfc,
                          Real_ptr xcfr,
                          Real_ptr xubl,
                          Real_ptr xubc,
                          Real_ptr xubr,
                          Real_ptr xucl,
                          Real_ptr xucc,
                          Real_ptr xucr,
                          Real_ptr xufl,
                          Real_ptr xufc,
                          Real_ptr xufr,
                          Real_ptr dbl,
                          Real_ptr dbc,
                          Real_ptr dbr,
                          Real_ptr dcl,
                          Real_ptr dcc,
                          Real_ptr dcr,
                          Real_ptr dfl,
                          Real_ptr dfc,
                          Real_ptr dfr,
                          Real_ptr cbl,
                          Real_ptr cbc,
                          Real_ptr cbr,
                          Real_ptr ccl,
                          Real_ptr ccc,
                          Real_ptr ccr,
                          Real_ptr cfl,
                          Real_ptr cfc,
                          Real_ptr cfr,
                          Real_ptr ubl,
                          Real_ptr ubc,
                          Real_ptr ubr,
                          Real_ptr ucl,
                          Real_ptr ucc,
                          Real_ptr ucr,
                          Real_ptr ufl,
                          Real_ptr ufc,
                          Real_ptr ufr,
                          Index_ptr real_zones,
                          Index_type ibegin, Index_type iend)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i = ii + ibegin;
   if (i < iend) {
     MATVEC_3D_BODY_INDEX;
     MATVEC_3D_BODY;
   }
}


template < size_t block_size >
void MATVEC_3D::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  auto res{getCudaResource()};

  MATVEC_3D_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (matvec_3d<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          b,
                          xdbl,
                          xdbc,
                          xdbr,
                          xdcl,
                          xdcc,
                          xdcr,
                          xdfl,
                          xdfc,
                          xdfr,
                          xcbl,
                          xcbc,
                          xcbr,
                          xccl,
                          xccc,
                          xccr,
                          xcfl,
                          xcfc,
                          xcfr,
                          xubl,
                          xubc,
                          xubr,
                          xucl,
                          xucc,
                          xucr,
                          xufl,
                          xufc,
                          xufr,
                          dbl,
                          dbc,
                          dbr,
                          dcl,
                          dcc,
                          dcr,
                          dfl,
                          dfc,
                          dfr,
                          cbl,
                          cbc,
                          cbr,
                          ccl,
                          ccc,
                          ccr,
                          cfl,
                          cfc,
                          cfr,
                          ubl,
                          ubc,
                          ubr,
                          ucl,
                          ucc,
                          ucr,
                          ufl,
                          ufc,
                          ufr,
                          real_zones,
                          ibegin, iend );

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    RAJA::TypedListSegment<Index_type> zones(real_zones, iend,
                                             res, RAJA::Unowned);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        zones, [=] __device__ (Index_type i) {
          MATVEC_3D_BODY;
      });

    }
    stopTimer();

  } else {
     getCout() << "\n  MATVEC_3D : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MATVEC_3D, Cuda)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
