//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

#define D1D 4
#define Q1D 5
#define B_(x, y) B[x + Q1D*y] 
#define Bt_(x, y) Bt[x + D1D*y]
#define s_xy_(x, y) s_xy[x + M1D*y]
#define X_(dx, dy, dz, e) X[dx + D1D*dy + D1D*D1D*dz + D1D*D1D*D1D*e]
#define Y_(dx, dy, dz, e) Y[dx + D1D*dy + D1D*D1D*dz + D1D*D1D*D1D*e]
#define D_(qx, qy, qz, e) D[qx + Q1D*qy + Q1D*Q1D*qz + Q1D*Q1D*Q1D*e]

void MASS3DPA::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  MASS3DPA_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (int e = 0; e < NE; ++e) {
        
        double sol_xyz[Q1D][Q1D][Q1D];
        for (int qz = 0; qz < Q1D; ++qz) {
          for (int qy = 0; qy < Q1D; ++qy) {
            for (int qx = 0; qx < Q1D; ++qx) {
              sol_xyz[qz][qy][qx] = 0;
            }
          }
        }
        
        for (int dz = 0; dz < D1D; ++dz) {
          double sol_xy[Q1D][Q1D];
          for (int qy = 0; qy < Q1D; ++qy) {
            for (int qx = 0; qx < Q1D; ++qx) {
              sol_xy[qy][qx] = 0;
            }
          }
          
          for (int dy = 0; dy < D1D; ++dy) {
            double sol_x[Q1D];
            for (int qx = 0; qx < Q1D; ++qx) {
              sol_x[qx] = 0;
            }
            
            for (int dx = 0; dx < D1D; ++dx) {
              const double s = X_(dx, dy, dz, e);
              for (int qx = 0; qx < Q1D; ++qx) {
                sol_x[qx] += B_(qx, dx) * s;
              }
            }
            
            for (int qy = 0; qy < Q1D; ++qy) {
              const double wy = B_(qy, dy);
              for (int qx = 0; qx < Q1D; ++qx) {
                sol_xy[qy][qx] += wy * sol_x[qx];
              }
            }
          }
          
          for (int qz = 0; qz < Q1D; ++qz) {
            const double wz = B_(qz, dz);
            for (int qy = 0; qy < Q1D; ++qy) {
              for (int qx = 0; qx < Q1D; ++qx) {
                sol_xyz[qz][qy][qx] += wz * sol_xy[qy][qx];
              }
            }
          }
        }
        
        for (int qz = 0; qz < Q1D; ++qz) {
          for (int qy = 0; qy < Q1D; ++qy) {
            for (int qx = 0; qx < Q1D; ++qx) {
              sol_xyz[qz][qy][qx] *= D_(qx, qy, qz, e);
            }
          }
        }
        
        for (int qz = 0; qz < Q1D; ++qz) {
          double sol_xy[D1D][D1D];
          for (int dy = 0; dy < D1D; ++dy) {
            for (int dx = 0; dx < D1D; ++dx) {
              sol_xy[dy][dx] = 0;
            }
          }
          
          for (int qy = 0; qy < Q1D; ++qy) {
            double sol_x[D1D];
            for (int dx = 0; dx < D1D; ++dx) {
              sol_x[dx] = 0;
            }
            
            for (int qx = 0; qx < Q1D; ++qx) {
              const double s = sol_xyz[qz][qy][qx];
              for (int dx = 0; dx < D1D; ++dx) {
                sol_x[dx] += Bt_(dx, qx) * s;
              }
            }
            
            for (int dy = 0; dy < D1D; ++dy) {
              const double wy = Bt_(dy, qy);
              for (int dx = 0; dx < D1D; ++dx) {
                sol_xy[dy][dx] += wy * sol_x[dx];
              }
            }
          }
          
          for (int dz = 0; dz < D1D; ++dz) {
            const double wz = Bt_(dz, qz);
            for (int dy = 0; dy < D1D; ++dy) {
              for (int dx = 0; dx < D1D; ++dx) {
                Y_(dx, dy, dz, e) += wz * sol_xy[dy][dx];
              }
            }
          }
        }
        
      }//element loop


    }
    stopTimer();


  } else if ( vid == RAJA_Seq ) {

    printf("TODO \n");

  } else {
     std::cout << "\n MASS3DPA : Unknown Seq variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf
