//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAGONAL_PAR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define TRIDIAGONAL_PAR_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(Aa_global, m_Aa_global, m_N*iend); \
  allocAndInitHipDeviceData(Ab_global, m_Ab_global, m_N*iend); \
  allocAndInitHipDeviceData(Ac_global, m_Ac_global, m_N*iend); \
  allocAndInitHipDeviceData(x_global, m_x_global, m_N*iend); \
  allocAndInitHipDeviceData(b_global, m_b_global, m_N*iend);

#define TRIDIAGONAL_PAR_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_x_global, x_global, m_N*iend); \
  deallocHipDeviceData(Aa_global); \
  deallocHipDeviceData(Ab_global); \
  deallocHipDeviceData(Ac_global); \
  deallocHipDeviceData(x_global); \
  deallocHipDeviceData(b_global);

#define TRIDIAGONAL_PAR_TEMP_DATA_SETUP_HIP_GLOBAL \
  Real_ptr d_global; \
  allocHipDeviceData(d_global, m_N*iend);

#define TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_HIP_GLOBAL \
  deallocHipDeviceData(d_global);

#define TRIDIAGONAL_PAR_LOCAL_DATA_SETUP_HIP_GLOBAL \
  TRIDIAGONAL_PAR_LOCAL_DATA_SETUP; \
  Real_ptr d = d_global + TRIDIAGONAL_PAR_OFFSET(i);

#define TRIDIAGONAL_PAR_TEMP_DATA_SETUP_HIP_SHARED

#define TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_HIP_SHARED

#define TRIDIAGONAL_PAR_LOCAL_DATA_SETUP_HIP_SHARED \
  TRIDIAGONAL_PAR_LOCAL_DATA_SETUP_V2;

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void tridiagonal_par(Real_ptr Aa_global, Real_ptr Ab_global, Real_ptr Ac_global,
                                Real_ptr  x_global, Real_ptr  b_global,
                                Index_type N, Index_type iend)
{
  Index_type i = blockIdx.x;
  // if (i < iend)
  {
    TRIDIAGONAL_PAR_LOCAL_DATA_SETUP_HIP_SHARED;

    int n = threadIdx.x;

    __shared__ volatile Real_type Aa[block_size-1]; // lower diagonal of A [1:N)
    // for (int n = 1; n < N; ++n)
    if (1 <= n && n < N) { // par
      Index_type idx_n = TRIDIAGONAL_PAR_INDEX(n);
      Aa[n-1] = Aa_data[idx_n];
    }
    __shared__ volatile Real_type Ac[block_size-1]; // upper diagonal of A [0:N-1)
    // for (int n = 0; n < N-1; ++n)
    if (0 <= n && n < N-1) { // par
      Index_type idx_n = TRIDIAGONAL_PAR_INDEX(n);
      Ac[n] = Ac_data[idx_n];
    }
    __shared__ volatile Real_type Ab[block_size]; // diagonal of A [0:N)
    // for (int n = 0; n < N; ++n)
    if (0 <= n && n < N) { // par
      Index_type idx_n = TRIDIAGONAL_PAR_INDEX(n);
      Ab[n] = Ab_data[idx_n];
    }
    __shared__ volatile Real_type b[block_size]; // rhs of equation [0:N)
    // for (int n = 0; n < N; ++n)
    if (0 <= n && n < N) { // par
      Index_type idx_n = TRIDIAGONAL_PAR_INDEX(n);
      b[n] = b_data[idx_n];
    }

    __syncthreads();

    __shared__ volatile Real_type AaAc[block_size]; // holds products (-e[i]*f[i-1]) [1:N]
    // __shared__ volatile Real_type temp[block_size]; // temporary array [1:N]
    __shared__ volatile Real_type qi[block_size]; // Qi[n] [1:N]
    __shared__ volatile Real_type qim1[block_size+1]; // Qi-1[n] [0:N]
    __shared__ volatile Real_type qim2[block_size+2]; // Qi-1[n] [-1:N]

    __shared__ volatile Real_type u[block_size]; // [1:N]

    // Real_type m[N-1];  // [2:N]
    __shared__ volatile Real_type m[block_size]; // [1:N]

    __shared__ volatile Real_type x[block_size]; // [1:N]

    if (n == 0) {
      AaAc[0] = 0;
    }
    // for (int n = 1; n < N; ++n)
    if (1 <= n && n < N) { // par
      AaAc[n] = -Aa[n-1] * Ac[n-1];
    }
    // for (int n = 0; n < N+2; ++n)
    if (0 <= n && n < N+2) { // par
      qim2[n] = 1;
    }
    if (n == 0) {
      qim1[0] = 1;
    }
    // for (int n = 0; n < N; ++n)
    if (0 <= n && n < N) { // par
      qim1[n+1] = Ab[n];
    }
    if (n == 0) {
      qi[0] = Ab[0];
    }
    // for (int n = 1; n < N; ++n)
    if (1 <= n && n < N) { // par
      qi[n] = Ab[n] * Ab[n-1] + AaAc[n];
    }
    __syncthreads();
    for (int k = 2; k <= N; k *= 2) {
      Real_type qim2_nP2; // qim2[n+2]
      // for (int n = k-2; n < N; ++n)
      if (k-2 <= n && n < N) { // par
        qim2_nP2 = qim1[n+1] * qim1[n-k+2] + AaAc[n-k+2] * qim2[n+2] * qim2[n-k+2];
      }
      Real_type qim1_nP1; // qim1[n+1]
      // for (int n = k-1; n < N; ++n)
      if (k-1 <= n && n < N) { // par
        qim1_nP1 = qi[n] * qim1[n-k+1] + AaAc[n-k+1] * qim1[n+1] * qim2[n-k+1];
      }
      __syncthreads();
      // for (int n = k-2; n < N; ++n)
      if (k-2 <= n && n < N) { // par
        qim2[n+2] = qim2_nP2;
      }
      if (k-1 <= n && n < N) { // par
        qim1[n+1] = qim1_nP1;
      }
      __syncthreads();
      // for (int n = k; n < N; ++n)
      if (k <= n && n < N) { // par
        qi[n] = Ab[n] * qim1[n] + AaAc[n] * qim2[n];
      }
    }
    __syncthreads();

    if (n == 0) {
      u[0] = qi[0];
    }
    // for (int n = 1; n < N; ++n)
    if (1 <= n && n < N) { // par
      u[n] = qi[n] / qi[n-1];
    }
    __syncthreads();
    // for (int n = 1; n < N; ++n)
    if (1 <= n && n < N) { // par
      m[n] = Aa[n-1] / u[n-1];
    }
    // for (int n = 0; n < N; ++n)
    if (0 <= n && n < N) { // par
      x[n] = b[n];
    }
    __syncthreads();
    if (n == 0) {
      m[0] = 0;
    }
    // for (int n = 1; n < N; ++n)
    if (1 <= n && n < N) { // par
      m[n] = -m[n];
    }
    __syncthreads();
    for (int k = 1; k <= N; k *= 2) {
      Real_type x_n; // x[n]
      // for (int n = k; n < N; ++n)
      if (k <= n && n < N) { // par
        x_n = x[n] + x[n-k] * m[n];
      }
      Real_type m_n; // m[n]
      // for (int n = k; n < N; ++n)
      if (k <= n && n < N) { // par
        m_n = m[n] * m[n-k];
      }
      __syncthreads();
      if (k <= n && n < N) { // par
        x[n] = x_n;
      }
      if (k <= n && n < N) { // par
        m[n] = m_n;
      }
      __syncthreads();
    }

    // for (int n = 0; n < N; ++n)
    if (0 <= n && n < N) { // par
      x[n] = x[n] / u[n];
    }
    // for (int n = 0; n < N-1; ++n)
    if (0 <= n && n < N-1) { // par
      m[n] = -Ac[n] / u[n];
    }
    if (n == N-1) {
      m[N-1] = 0;
    }
    __syncthreads();
    for (int k = 1; k <= N; k *= 2) {
      Real_type x_n; // x[n]
      // for (int n = 0; n < N-k; ++n)
      if (0 <= n && n < N-k) { // par
        x_n = x[n] + x[n+k] * m[n];
      }
      Real_type m_n; // m[n]
      // for (int n = 0; n < N-k; ++n)
      if (0 <= n && n < N-k) { // par
        m_n = m[n] * m[n+k];
      }
      __syncthreads();
      if (0 <= n && n < N-k) { // par
        x[n] = x_n;
      }
      if (0 <= n && n < N-k) { // par
        m[n] = m_n;
      }
      __syncthreads();
    }

    // for (int n = 0; n < N; ++n)
    if (0 <= n && n < N) { // par
      Index_type idx_n = TRIDIAGONAL_PAR_INDEX(n);
      x_data[idx_n] = x[n];
    }
  }
}


template < size_t block_size >
void TRIDIAGONAL_PAR::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRIDIAGONAL_PAR_DATA_SETUP;

  if ( vid == Base_HIP ) {

    TRIDIAGONAL_PAR_DATA_SETUP_HIP;
    TRIDIAGONAL_PAR_TEMP_DATA_SETUP_HIP_SHARED;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      assert(N+2 <= static_cast<Index_type>(block_size));
      Index_type matrices_per_block = 1;
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, matrices_per_block);
      hipLaunchKernelGGL((tridiagonal_par<block_size>), dim3(grid_size), dim3(N+2), 0, 0,
          Aa_global, Ab_global, Ac_global,
          x_global, b_global,
          N, iend );
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_HIP_SHARED;
    TRIDIAGONAL_PAR_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    TRIDIAGONAL_PAR_DATA_SETUP_HIP;
    TRIDIAGONAL_PAR_TEMP_DATA_SETUP_HIP_GLOBAL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto tridiagonal_par_lambda = [=] __device__ (Index_type i) {
        TRIDIAGONAL_PAR_LOCAL_DATA_SETUP_HIP_GLOBAL;
        TRIDIAGONAL_PAR_BODY_FORWARD_TEMP_GLOBAL;
        TRIDIAGONAL_PAR_BODY_BACKWARD_TEMP_GLOBAL;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(tridiagonal_par_lambda)>),
        grid_size, block_size, 0, 0, ibegin, iend, tridiagonal_par_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_HIP_GLOBAL;
    TRIDIAGONAL_PAR_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    TRIDIAGONAL_PAR_DATA_SETUP_HIP;
    TRIDIAGONAL_PAR_TEMP_DATA_SETUP_HIP_GLOBAL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        TRIDIAGONAL_PAR_LOCAL_DATA_SETUP_HIP_GLOBAL;
        TRIDIAGONAL_PAR_BODY_FORWARD_TEMP_GLOBAL;
        TRIDIAGONAL_PAR_BODY_BACKWARD_TEMP_GLOBAL;
      });

    }
    stopTimer();

    TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_HIP_GLOBAL;
    TRIDIAGONAL_PAR_DATA_TEARDOWN_HIP;

  } else {
      getCout() << "\n  TRIDIAGONAL_PAR : Unknown Hip variant id = " << vid << std::endl;
  }
}

void TRIDIAGONAL_PAR::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      if (tune_idx == t) {
        runHipVariantImpl<block_size>(vid);
      }
      t += 1;
    }
  });
}

void TRIDIAGONAL_PAR::setHipTuningDefinitions(VariantID vid)
{
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      addVariantTuningName(vid, "block_"+std::to_string(block_size));
    }
  });
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
