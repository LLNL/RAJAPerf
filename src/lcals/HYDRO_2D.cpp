//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{


#define HYDRO_2D_DATA_SETUP_CPU \
  ResReal_ptr zadat = m_za; \
  ResReal_ptr zbdat = m_zb; \
  ResReal_ptr zmdat = m_zm; \
  ResReal_ptr zpdat = m_zp; \
  ResReal_ptr zqdat = m_zq; \
  ResReal_ptr zrdat = m_zr; \
  ResReal_ptr zudat = m_zu; \
  ResReal_ptr zvdat = m_zv; \
  ResReal_ptr zzdat = m_zz; \
\
  ResReal_ptr zroutdat = m_zrout; \
  ResReal_ptr zzoutdat = m_zzout; \
\
  const Real_type s = m_s; \
  const Real_type t = m_t; \
\
  const Index_type kn = m_kn; \
  const Index_type jn = m_jn;


HYDRO_2D::HYDRO_2D(const RunParams& params)
  : KernelBase(rajaperf::Lcals_HYDRO_2D, params)
{
   m_jn = 1000;
   m_kn = 1000;

   m_s = 0.0041;
   m_t = 0.0037;

   setDefaultSize(m_jn);
   setDefaultReps(200);
}

HYDRO_2D::~HYDRO_2D() 
{
}

void HYDRO_2D::setUp(VariantID vid)
{
  m_kn = getRunSize();
  m_array_length = m_kn * m_jn;

  allocAndInitDataConst(m_zrout, m_array_length, 0.0, vid);
  allocAndInitDataConst(m_zzout, m_array_length, 0.0, vid);
  allocAndInitData(m_za, m_array_length, vid);
  allocAndInitData(m_zb, m_array_length, vid);
  allocAndInitData(m_zm, m_array_length, vid);
  allocAndInitData(m_zp, m_array_length, vid);
  allocAndInitData(m_zq, m_array_length, vid);
  allocAndInitData(m_zr, m_array_length, vid);
  allocAndInitData(m_zu, m_array_length, vid);
  allocAndInitData(m_zv, m_array_length, vid);
  allocAndInitData(m_zz, m_array_length, vid);
}

void HYDRO_2D::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  HYDRO_2D_DATA_SETUP_CPU;

  HYDRO_2D_VIEWS_RAJA;

  auto hydro2d1_lam = [=] (Index_type k, Index_type j) {
                        HYDRO_2D_BODY1_RAJA;
                      };
  auto hydro2d2_lam = [=] (Index_type k, Index_type j) {
                        HYDRO_2D_BODY2_RAJA;
                      };
  auto hydro2d3_lam = [=] (Index_type k, Index_type j) {
                        HYDRO_2D_BODY3_RAJA;
                      };

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            HYDRO_2D_BODY1;
          }
        }

        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            HYDRO_2D_BODY2;
          }
        }

        for (Index_type k = kbeg; k < kend; ++k ) {
          for (Index_type j = jbeg; j < jend; ++j ) {
            HYDRO_2D_BODY3;
          }
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)     
    case RAJA_Seq : {

      using EXECPOL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::loop_exec,  // k
            RAJA::statement::For<1, RAJA::loop_exec,  // j
              RAJA::statement::Lambda<0>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel<EXECPOL>(
                     RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                       RAJA::RangeSegment(jbeg, jend)),
                     hydro2d1_lam); 

        RAJA::kernel<EXECPOL>(
                     RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                       RAJA::RangeSegment(jbeg, jend)),
                     hydro2d2_lam); 

        RAJA::kernel<EXECPOL>(
                     RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                       RAJA::RangeSegment(jbeg, jend)),
                     hydro2d3_lam); 

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel
        {

          #pragma omp for nowait
	  for (Index_type k = kbeg; k < kend; ++k ) {
	    for (Index_type j = jbeg; j < jend; ++j ) {
	      HYDRO_2D_BODY1;
	    }
	  }

          #pragma omp for nowait
	  for (Index_type k = kbeg; k < kend; ++k ) {
	    for (Index_type j = jbeg; j < jend; ++j ) {
	      HYDRO_2D_BODY2;
	    }
	  }

          #pragma omp for nowait
	  for (Index_type k = kbeg; k < kend; ++k ) {
	    for (Index_type j = jbeg; j < jend; ++j ) {
	      HYDRO_2D_BODY3;
	    }
	  }

        } // end omp parallel region

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      using EXECPOL =
        RAJA::KernelPolicy<
          RAJA::statement::For<0, RAJA::omp_for_nowait_exec,  // k
            RAJA::statement::For<1, RAJA::loop_exec,  // j
              RAJA::statement::Lambda<0>
            >
          >
        >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {

          RAJA::kernel<EXECPOL>(
                       RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                         RAJA::RangeSegment(jbeg, jend)),
                       hydro2d1_lam); 

          RAJA::kernel<EXECPOL>(
                       RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                         RAJA::RangeSegment(jbeg, jend)),
                       hydro2d2_lam); 

          RAJA::kernel<EXECPOL>(
                       RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                                         RAJA::RangeSegment(jbeg, jend)),
                       hydro2d3_lam); 

        }); // end omp parallel region 

      }
      stopTimer();

      break;
    }
#endif

#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
    {
      runOpenMPTargetVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA :
    case RAJA_CUDA :
    {
      runCudaVariant(vid);
      break;
    }
#endif

    default : {
      std::cout << "\n  HYDRO_2D : Unknown variant id = " << vid << std::endl;
    }

  }

}

void HYDRO_2D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_zzout, m_array_length);
  checksum[vid] += calcChecksum(m_zrout, m_array_length);
}

void HYDRO_2D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_zrout);
  deallocData(m_zzout);
  deallocData(m_za);
  deallocData(m_zb);
  deallocData(m_zm);
  deallocData(m_zp);
  deallocData(m_zq);
  deallocData(m_zr);
  deallocData(m_zu);
  deallocData(m_zv);
  deallocData(m_zz);
}

} // end namespace lcals
} // end namespace rajaperf
