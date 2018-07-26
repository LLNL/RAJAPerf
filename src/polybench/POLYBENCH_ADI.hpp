/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for Polybench kernel 2mm .
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-xxxxxx
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For additional details, please read the file LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
/// Polybench ADI:  Alternating Direction Implicit solver Kernel reference implementation
///  DX = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
///  DY = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
///  DT = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_TSTEPS;
///  B1 = SCALAR_VAL(2.0);
///  B2 = SCALAR_VAL(1.0);
///  mul1 = B1 * DT / (DX * DX);
///  mul2 = B2 * DT / (DY * DY);
///
///  a = -mul1 /  SCALAR_VAL(2.0);
///  b = SCALAR_VAL(1.0)+mul1;
///  c = a;
///  d = -mul2 / SCALAR_VAL(2.0);
///  e = SCALAR_VAL(1.0)+mul2;
///  f = d;
///
/// for (t=1; t<=_PB_TSTEPS; t++) {
///    //Column Sweep
///    for (i=1; i<_PB_N-1; i++) {
///      v[0][i] = SCALAR_VAL(1.0);
///      p[i][0] = SCALAR_VAL(0.0);
///      q[i][0] = v[0][i];
///      for (j=1; j<_PB_N-1; j++) {
///        p[i][j] = -c / (a*p[i][j-1]+b);
///        q[i][j] = (-d*u[j][i-1]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d)*u[j][i] - f*u[j][i+1]-a*q[i][j-1])/(a*p[i][j-1]+b);
///      }
///      
///      v[_PB_N-1][i] = SCALAR_VAL(1.0);
///      for (j=_PB_N-2; j>=1; j--) {
///        v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
///      }
///    }
///    //Row Sweep
///    for (i=1; i<_PB_N-1; i++) {
///      u[i][0] = SCALAR_VAL(1.0);
///      p[i][0] = SCALAR_VAL(0.0);
///      q[i][0] = u[i][0];
///      for (j=1; j<_PB_N-1; j++) {
///        p[i][j] = -f / (d*p[i][j-1]+e);
///        q[i][j] = (-a*v[i-1][j]+(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a)*v[i][j] - c*v[i+1][j]-d*q[i][j-1])/(d*p[i][j-1]+e);
///      }
///      u[i][_PB_N-1] = SCALAR_VAL(1.0);
///      for (j=_PB_N-2; j>=1; j--) {
///        u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
///      }
///    }
///  }



#ifndef RAJAPerf_POLYBENCH_ADI_HXX
#define RAJAPerf_POLYBENCH_ADI_HXX

#include <stdio.h>
#include "common/KernelBase.hpp"

#ifndef NDEBUG
#   define myassert(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define myassert(condition, message) do { } while (false)
#endif


#define POLYBENCH_ADI_BODY1 \
  Real_type DX,DY,DT; \
  Real_type B1,B2; \
  Real_type mul1,mul2; \
  Real_type a,b,c,d,e,f; \
  DX = 1.0/(Real_type)m_n; \
  DY = 1.0/(Real_type)m_n; \
  DT = 1.0/(Real_type)m_tsteps; \
  B1 = 2.0; \
  B2 = 1.0; \
  mul1 = B1 * DT / (DX * DX); \
  mul2 = B2 * DT / (DY * DY); \
  a = -mul1 / 2.0; \
  b = 1.0 + mul1; \
  c = a; \
  d = -mul2 /2.0; \
  e = 1.0 + mul2; \
  f = d; 

#define POLYBENCH_ADI_BODY2 \
  *(V + 0 * n + i) = 1.0; \
  *(P + i * n + 0) = 0.0; \
  *(Q + i * n + 0) = *(V + 0 * n + i);

#define POLYBENCH_ADI_BODY3 \
  *(P + i * n + j) = -c / (a * *(P + i * n + j-1)+b); \
  *(Q + i * n + j) = (-d * *(U + j * n + i-1) + (1.0 + 2.0*d) * *(U + j * n + i) - f* *(U + j * n + i + 1) -a * *(Q + i * n + j-1))/(a * *(P + i * n + j -1)+b); 



//  int tid = omp_get_thread_num(); \


#define POLYBENCH_ADI_BODY4 \
  *(V + (n-1) * n + i) = 1.0;

#define POLYBENCH_ADI_BODY5 \
  int jj = n - 1 - j; \
  *(V + jj * n + i)  = *(P + i * n + jj) * *(V + (jj+1) * n + i) + *(Q + i * n + jj); 
  

#define POLYBENCH_ADI_BODY6 \
  *(U + i * n + 0) = 1.0; \
  *(P + i * n + 0) = 0.0; \
  *(Q + i * n + 0) = *(U + i * n + 0);

#define POLYBENCH_ADI_BODY7 \
  *(P + i * n + j) = -f / (d * *(P + i * n + j-1)+e); \
  *(Q + i * n + j) = (-a * *(V + (i-1) * n + j) + (1.0 + 2.0*a) * *(V + i * n + j) - c * *(V + (i + 1) * n + j) -d * *(Q + i * n + j-1))/(d * *(P + i * n + j-1)+e);

#define POLYBENCH_ADI_BODY8 \
  *(U + i * n + n-1) = 1.0;

#define POLYBENCH_ADI_BODY9 \
  int jj = n - 1 - j; \
  *(U + i * n + jj)= *(P + i * n + jj) * *(U + i * n + jj +1) + *(Q + i * n + jj); 


namespace rajaperf 
{

class RunParams;

namespace polybench
{

class POLYBENCH_ADI : public KernelBase
{
public:

  POLYBENCH_ADI(const RunParams& params);

  ~POLYBENCH_ADI();


  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);
  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);


private:
  Index_type m_n;
  Index_type m_tsteps;
  Index_type m_run_reps;
  Real_ptr m_U;
  Real_ptr m_V;
  Real_ptr m_P;
  Real_ptr m_Q;
};

} // end namespace polybench
} // end namespace rajaperf

#endif // closing endif for header file include guard
