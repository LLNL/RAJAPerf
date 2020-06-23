//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

#define MASS3DPA_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(B, m_B, m_Q1D*m_D1D); \
  allocAndInitCudaDeviceData(Bt, m_Bt, m_Q1D*m_D1D); \
  allocAndInitCudaDeviceData(D, m_D, m_Q1D*m_Q1D*m_Q1D*m_NE); \
  allocAndInitCudaDeviceData(X, m_X, m_D1D*m_D1D*m_D1D*m_NE); \
  allocAndInitCudaDeviceData(Y, m_Y, m_D1D*m_D1D*m_D1D*m_NE);

#define MASS3DPA_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_Y, Y, m_D1D*m_D1D*m_D1D*m_NE); \
  deallocCudaDeviceData(B); \
  deallocCudaDeviceData(Bt); \
  deallocCudaDeviceData(D); \
  deallocCudaDeviceData(X); \
  deallocCudaDeviceData(Y);

#define D1D 4
#define Q1D 5
#define s_B_(x, y) s_B[x + Q1D*y]
#define s_Bt_(x, y) s_Bt[x + D1D*y]
#define s_xy_(x, y) s_xy[x + M1D*y]
#define X_(dx, dy, dz, e) X[dx + D1D*dy + D1D*D1D*dz + D1D*D1D*D1D*e]
#define Y_(dx, dy, dz, e) Y[dx + D1D*dy + D1D*D1D*dz + D1D*D1D*D1D*e]
#define D_(qx, qy, qz, e) D[qx + Q1D*qy + Q1D*Q1D*qz + Q1D*Q1D*Q1D*e]

__global__ void Mass3DPA(Index_type NE, const Real_ptr B, const Real_ptr Bt,
                         const Real_ptr D,  const Real_ptr X, Real_ptr Y)
{

  constexpr int DQ1D = D1D*Q1D;
  constexpr int M1D = D1D > Q1D ? D1D : Q1D;
  constexpr int M2D = M1D*M1D;

  const int e = blockIdx.x;

  //Basis functions sampled at quadrature points in 1D
  __shared__ double s_B[DQ1D];
  __shared__ double s_Bt[DQ1D];

  //Space for solution in the xy-plane
  __shared__ double s_xy[M2D];

  //Thread private memory
  double r_z[Q1D];
  double r_z2[D1D];

  { const int y = threadIdx.y;
    { const int x = threadIdx.x;

      const int id = (y * M1D) + x;
      // Fetch Q <--> D maps
      if (id < DQ1D) {
        s_B[id]  = B[id];
        s_Bt[id]  = Bt[id];
      }
      // Initialize our Z axis
      for (int qz = 0; qz < Q1D; ++qz) {
        r_z[qz] = 0;
      }
      for (int dz = 0; dz < D1D; ++dz) {
        r_z2[dz] = 0;
      }

    }
  }

  { const int dy = threadIdx.y;
    { const int dx = threadIdx.x;

      if ((dx < D1D) && (dy < D1D)) {
        for (int dz = 0; dz < D1D; ++dz) {
          const double s = X_(dx, dy, dz, e);
          // Calculate D -> Q in the Z axis
          for (int qz = 0; qz < Q1D; ++qz) {
            r_z[qz] += s * s_B_(qz, dz);
          }
        }
      }
    }
  }

    // For each xy plane
    for (int qz = 0; qz < Q1D; ++qz) {
      // Fill xy plane at given z position
      { const int dy = threadIdx.y;
        { const int dx = threadIdx.x;
          if ((dx < D1D) && (dy < D1D)) {
            s_xy_(dx, dy) = r_z[qz];
          }
        }
      }

       // Calculate Dxyz, xDyz, xyDz in plane
      { const int qy = threadIdx.y;
        { const int qx = threadIdx.x;

          if ((qx < Q1D) && (qy < Q1D)) {
            double s = 0;
            for (int dy = 0; dy < D1D; ++dy) {
              const double wy = s_B_(qy, dy);
              for (int dx = 0; dx < D1D; ++dx) {
                const double wx = s_B_(qx, dx);
                s += wx * wy * s_xy_(dx, dy);
              }
            }

            s *= D_(qx, qy, qz, e);

            for (int dz = 0; dz < D1D; ++dz) {
              const double wz  = s_Bt_(dz, qz);
              r_z2[dz] += wz * s;
            }
          }
        }
      }
      __syncthreads();
    }

    // Iterate over xy planes to compute solution
    for (int dz = 0; dz < D1D; ++dz) {

      // Place xy plane in @shared memory
      { const int qy = threadIdx.y;
        { const int qx = threadIdx.x;
          if ((qx < Q1D) && (qy < Q1D)) {
            s_xy_(qx, qy) = r_z2[dz];
          }
        }
      }
      // Finalize solution in xy plane
      {const int dy = threadIdx.y;
        {const int dx = threadIdx.x;
          if ((dx < D1D) && (dy < D1D)) {
            double solZ = 0;
            for (int qy = 0; qy < Q1D; ++qy) {
              const double wy = s_Bt_(dy, qy);
              for (int qx = 0; qx < Q1D; ++qx) {
                const double wx = s_Bt_(dx, qx);
                solZ += wx * wy * s_xy_(qx, qy);
              }
            }
            Y_(dx, dy, dz, e) += solZ;
          }
        }
      }
      __syncthreads();
    }

}


void MASS3DPA::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  MASS3DPA_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    MASS3DPA_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      dim3 nthreads_per_block(Q1D, Q1D, 1);

      Mass3DPA<<<m_NE, nthreads_per_block>>>(m_NE, B, Bt, D, X, Y);

    }
    stopTimer();

    MASS3DPA_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    printf("TODO \n");

  } else {
     std::cout << "\n MASS3DPA : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
