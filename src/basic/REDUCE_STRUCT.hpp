//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// REDUCE_STRUCT kernel reference implementation:
///
/// Real_type xsum = m_sum_init; Real_type ysum = m_sum_init;
/// Real_type xmin = m_min_init; Real_type ymin = m_min_init;
/// Real_type xmax = m_max_init; Real_type ymax = m_max_init;

///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   xsum += x[i] ; ysum += y[i] ;
///   xmin = RAJA_MIN(xmin, x[i]) ; xmax = RAJA_MAX(xmax, x[i]) ;
///   ymin = RAJA_MIN(ymin, y[i]) ; ymax = RAJA_MAX(ymax, y[i]) ;
/// }
///
/// points.xcenter = xsum;
/// points.xcenter /= points.N
/// points.xmin = xmin;
/// points.xmax = xmax;
/// points.ycenter = ysum;
/// points.ycenter /= points.N
/// points.ymin = ymin;
/// points.ymax = ymax;

///
/// RAJA_MIN/MAX are macros that do what you would expect.
///

#ifndef RAJAPerf_Basic_REDUCE_STRUCT_HPP
#define RAJAPerf_Basic_REDUCE_STRUCT_HPP


#define REDUCE_STRUCT_DATA_SETUP \
  points points; \
  points.N = getActualProblemSize(); \
  points.x = m_x; \
  points.y = m_y; \

#define REDUCE_STRUCT_BODY  \
  xsum += points.x[i] ; \
  xmin = RAJA_MIN(xmin, points.x[i]) ; \
  xmax = RAJA_MAX(xmax, points.x[i]) ; \
  ysum += points.y[i] ; \
  ymin = RAJA_MIN(ymin, points.y[i]) ; \
  ymax = RAJA_MAX(ymax, points.y[i]) ;

#define REDUCE_STRUCT_BODY_RAJA  \
  xsum += points.x[i] ; \
  xmin.min(points.x[i]) ; \
  xmax.max(points.x[i]) ; \
  ysum += points.y[i] ; \
  ymin.min(points.y[i]) ; \
  ymax.max(points.y[i]) ;


#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace basic
{

class REDUCE_STRUCT : public KernelBase
{
public:

  REDUCE_STRUCT(const RunParams& params);

  ~REDUCE_STRUCT();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);
  void runStdParVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

  struct points{
    Int_type N;
    Real_ptr x, y;

    Real_ptr GetCenter(){return &center[0];};
    Real_type GetXMax(){return xmax;};
    Real_type GetXMin(){return xmin;};
    Real_type GetYMax(){return ymax;};
    Real_type GetYMin(){return ymin;};
    void SetCenter(Real_type xval, Real_type yval){this->center[0]=xval, this->center[1]=yval;};
    void SetXMin(Real_type val){this->xmin=val;};
    void SetXMax(Real_type val){this->xmax=val;};
    void SetYMin(Real_type val){this->ymin=val;};
    void SetYMax(Real_type val){this->ymax=val;};              
        
    //results
    private:
    Real_type center[2] = {0.0,0.0};
    Real_type xmin, xmax;
    Real_type ymin, ymax;
    }; 
private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;
  Real_ptr m_x; Real_ptr m_y;
  Real_type	m_init_sum; 
  Real_type	m_init_min; 
  Real_type	m_init_max; 
  points m_points;
  Real_type X_MIN = 0.0, X_MAX = 100.0; 
  Real_type Y_MIN = 0.0, Y_MAX = 50.0; 
  Real_type Lx = (X_MAX) - (X_MIN); 
  Real_type Ly = (Y_MAX) - (Y_MIN);
 
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
