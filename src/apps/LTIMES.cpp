//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read raja-perfsuite/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// LTIMES kernel reference implementation:
///
/// for (Index_type d = 0; d < num_d; ++d ) {
///   for (Index_type z = 0; z < num_z; ++z ) {
///     for (Index_type g = 0; g < num_g; ++g ) {
///       for (Index_type m = 0; z < num_m; ++m ) {
///         phi[m+ (g * num_g) + (z * num_z * num_g)] +=  
///           ell[d+ (m * num_m)] * psi[d+ (g * num_g) + (z * num_z * num_g];
///          
///       }
///     }
///   }
/// }
///
/// RAJA variants of kernel use multi-dimensional layouts and views to 
/// do the same thing.
///

#include "LTIMES.hpp"

#include "common/DataUtils.hpp"

#include "RAJA/RAJA.hpp"
#include "camp/camp.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{

//
// These index value types cannot be defined in function scope for
// RAJA CUDA variant to work.
//
namespace ltimes_idx {
  RAJA_INDEX_VALUE(ID, "ID");
  RAJA_INDEX_VALUE(IZ, "IZ");
  RAJA_INDEX_VALUE(IG, "IG");
  RAJA_INDEX_VALUE(IM, "IM");
}

#define LTIMES_DATA \
  ResReal_ptr phidat = m_phidat; \
  ResReal_ptr elldat = m_elldat; \
  ResReal_ptr psidat = m_psidat; \
\
  Index_type num_d = m_num_d; \
  Index_type num_z = m_num_z; \
  Index_type num_g = m_num_g; \
  Index_type num_m = m_num_m;

#define LTIMES_VIEWS_RAJA \
  using namespace ltimes_idx; \
\
  using PSI_VIEW = RAJA::TypedView<Real_type, RAJA::Layout<3>, IZ, IG, ID>; \
  using ELL_VIEW = RAJA::TypedView<Real_type, RAJA::Layout<2>, IM, ID>; \
  using PHI_VIEW = RAJA::TypedView<Real_type, RAJA::Layout<3>, IZ, IG, IM>; \
\
  PSI_VIEW psi(psidat, \
               RAJA::make_permuted_layout( {num_z, num_g, num_d}, \
                     RAJA::as_array<camp::idx_seq<0, 1, 2> >::get() ) ); \
  ELL_VIEW ell(elldat, \
               RAJA::make_permuted_layout( {num_m, num_d}, \
                     RAJA::as_array<camp::idx_seq<0, 1> >::get() ) ); \
  PHI_VIEW phi(phidat, \
               RAJA::make_permuted_layout( {num_z, num_g, num_m}, \
                     RAJA::as_array<camp::idx_seq<0, 1, 2> >::get() ) ); \
\
      using IDRange = RAJA::TypedRangeSegment<ID>; \
      using IZRange = RAJA::TypedRangeSegment<IZ>; \
      using IGRange = RAJA::TypedRangeSegment<IG>; \
      using IMRange = RAJA::TypedRangeSegment<IM>;

#define LTIMES_BODY \
  phidat[m+ (g * num_m) + (z * num_m * num_g)] += \
    elldat[d+ (m * num_d)] * psidat[d+ (g * num_d) + (z * num_d * num_g)];

#define LTIMES_BODY_RAJA \
  phi(z, g, m) +=  ell(m, d) * psi(z, g, d);


#if defined(RAJA_ENABLE_CUDA)

#define LTIMES_DATA_SETUP_CUDA \
  Real_ptr phidat; \
  Real_ptr elldat; \
  Real_ptr psidat; \
\
  Index_type num_d = m_num_d; \
  Index_type num_z = m_num_z; \
  Index_type num_g = m_num_g; \
  Index_type num_m = m_num_m; \
\
  allocAndInitCudaDeviceData(phidat, m_phidat, m_philen); \
  allocAndInitCudaDeviceData(elldat, m_elldat, m_elllen); \
  allocAndInitCudaDeviceData(psidat, m_psidat, m_psilen);

#define LTIMES_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_phidat, phidat, m_philen); \
  deallocCudaDeviceData(phidat); \
  deallocCudaDeviceData(elldat); \
  deallocCudaDeviceData(psidat);

__global__ void ltimes(Real_ptr phidat, Real_ptr elldat, Real_ptr psidat,
                       Index_type d, 
                       Index_type num_d, Index_type num_g, Index_type num_m)
{
   Index_type m = threadIdx.x;
   Index_type g = blockIdx.y;
   Index_type z = blockIdx.z;

   LTIMES_BODY;
}

#endif // if defined(RAJA_ENABLE_CUDA)


LTIMES::LTIMES(const RunParams& params)
  : KernelBase(rajaperf::Apps_LTIMES, params)
{
#if 1
  m_num_z_default = 64;
  m_num_g_default = 8;
  m_num_m_default = 256;
  m_num_d_default = 16;
#else
  m_num_z_default = 4;
  m_num_g_default = 1;
  m_num_m_default = 8;
  m_num_d_default = 2;
#endif

  setDefaultSize(m_num_d_default * m_num_m_default * 
                 m_num_g_default * m_num_z_default);
  setDefaultReps(500);
}

LTIMES::~LTIMES() 
{
}

void LTIMES::setUp(VariantID vid)
{
  m_num_z = m_num_z_default;
  m_num_g = m_num_g_default;  
  m_num_m = run_params.getSizeFactor() * m_num_m_default;  
  m_num_d = m_num_d_default;  

  m_philen = m_num_m * m_num_g * m_num_z;
  m_elllen = m_num_d * m_num_m;
  m_psilen = m_num_d * m_num_g * m_num_z;

  allocAndInitDataConst(m_phidat, int(m_philen), Real_type(0.0), vid);
  allocAndInitData(m_elldat, int(m_elllen), vid);
  allocAndInitData(m_psidat, int(m_psilen), vid);
}

void LTIMES::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  switch ( vid ) {

    case Base_Seq : {

      LTIMES_DATA;
  
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type d = 0; d < num_d; ++d ) {
          for (Index_type z = 0; z < num_z; ++z ) {
            for (Index_type g = 0; g < num_g; ++g ) {
              for (Index_type m = 0; m < num_m; ++m ) {
                LTIMES_BODY;
              }
            }
          }
        }

      }
      stopTimer();

      break;
    } 

    case RAJA_Seq : {

#if defined(USE_FORALLN_FOR_SEQ)

      LTIMES_DATA;

      LTIMES_VIEWS_RAJA;
 
      using EXEC_POL = 
        RAJA::NestedPolicy<RAJA::ExecList< RAJA::seq_exec, 
                                           RAJA::seq_exec, 
                                           RAJA::seq_exec, 
                                           RAJA::seq_exec> >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forallN< EXEC_POL, ID, IZ, IG, IM >(
              RAJA::RangeSegment(0, num_d),
              RAJA::RangeSegment(0, num_z),
              RAJA::RangeSegment(0, num_g),
              RAJA::RangeSegment(0, num_m),
          [=](ID d, IZ z, IG g, IM m) {
          LTIMES_BODY_RAJA;
        });

      }
      stopTimer(); 

#else // use RAJA::nested

      LTIMES_DATA;

      LTIMES_VIEWS_RAJA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      
        RAJA::nested::forall(RAJA::nested::Policy<
                             RAJA::nested::For<3, RAJA::seq_exec>,
                             RAJA::nested::For<2, RAJA::seq_exec>,
                             RAJA::nested::For<1, RAJA::seq_exec>,
                             RAJA::nested::For<0, RAJA::seq_exec> >{},
                             camp::make_tuple(IDRange(0, num_d),
                                              IZRange(0, num_z),
                                              IGRange(0, num_g),
                                              IMRange(0, num_m)), 
          [=](ID d, IZ z, IG g, IM m) {
          LTIMES_BODY_RAJA;
        });

      }
      stopTimer(); 

#endif

      break;
    }

#if defined(RAJA_ENABLE_OPENMP)      
    case Base_OpenMP : {

      LTIMES_DATA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type d = 0; d < num_d; ++d ) {
          #pragma omp parallel for
          for (Index_type z = 0; z < num_z; ++z ) {
            for (Index_type g = 0; g < num_g; ++g ) {
              for (Index_type m = 0; m < num_m; ++m ) {
                LTIMES_BODY;
              }
            }
          }
        }  

      }
      stopTimer();

      break;
    }

    case RAJALike_OpenMP : {
      // case is not defined...
      break;
    }

    case RAJA_OpenMP : {

#if defined(USE_FORALLN_FOR_OPENMP)

      LTIMES_DATA;

      LTIMES_VIEWS_RAJA;

      using EXEC_POL = 
        RAJA::NestedPolicy<RAJA::ExecList< RAJA::seq_exec,
                                           RAJA::omp_parallel_for_exec,
                                           RAJA::seq_exec,
                                           RAJA::seq_exec> >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forallN< EXEC_POL, ID, IZ, IG, IM >(
              RAJA::RangeSegment(0, num_d),
              RAJA::RangeSegment(0, num_z),
              RAJA::RangeSegment(0, num_g),
              RAJA::RangeSegment(0, num_m),
          [=](ID d, IZ z, IG g, IM m) {
          LTIMES_BODY_RAJA;
        });

      }
      stopTimer();

#else // use RAJA::nested

      LTIMES_DATA;

      LTIMES_VIEWS_RAJA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::nested::forall(RAJA::nested::Policy<
                             RAJA::nested::For<3, RAJA::seq_exec>,
                             RAJA::nested::For<2, RAJA::omp_parallel_for_exec>,
                             RAJA::nested::For<1, RAJA::seq_exec>,
                             RAJA::nested::For<0, RAJA::seq_exec> >{},
                             camp::make_tuple(IDRange(0, num_d),
                                              IZRange(0, num_z),
                                              IGRange(0, num_g),
                                              IMRange(0, num_m)),
          [=](ID d, IZ z, IG g, IM m) {
          LTIMES_BODY_RAJA;
        });

      }
      stopTimer();

#endif

      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA : {

      LTIMES_DATA_SETUP_CUDA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        dim3 nthreads_per_block(num_m, 1, 1);
        dim3 nblocks(1, num_g, num_z);

        for (Index_type d = 0; d < num_d; ++d ) {
          ltimes<<<nblocks, nthreads_per_block>>>(phidat, elldat, psidat,
                                                  d,
                                                  num_d, num_g, num_m);  
        }  

      }
      stopTimer();

      LTIMES_DATA_TEARDOWN_CUDA;

      break;
    }

    case RAJA_CUDA : {

#if defined(USE_FORALLN_FOR_CUDA)

      LTIMES_DATA_SETUP_CUDA;

      LTIMES_VIEWS_RAJA;

      using EXEC_POL =
        RAJA::NestedPolicy<RAJA::ExecList< RAJA::seq_exec,
                                           RAJA::cuda_block_z_exec,
                                           RAJA::cuda_block_y_exec,
                                           RAJA::cuda_thread_x_exec > >;
              
      startTimer(); 
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        
        RAJA::forallN< EXEC_POL, ID, IZ, IG, IM >(
              RAJA::RangeSegment(0, num_d),
              RAJA::RangeSegment(0, num_z),
              RAJA::RangeSegment(0, num_g),
              RAJA::RangeSegment(0, num_m),
          [=] __device__ (ID d, IZ z, IG g, IM m) {
          LTIMES_BODY_RAJA;
        });
      
      }
      stopTimer();

      LTIMES_DATA_TEARDOWN_CUDA;

#else // use RAJA::nested

      LTIMES_DATA_SETUP_CUDA;

      LTIMES_VIEWS_RAJA;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::nested::forall(RAJA::nested::Policy<
                             RAJA::nested::For<3, RAJA::seq_exec>,
                             RAJA::nested::For<2, RAJA::cuda_block_z_exec>,
                             RAJA::nested::For<1, RAJA::cuda_block_y_exec>,
                             RAJA::nested::For<0, RAJA::cuda_thread_x_exec> >{},
                             camp::make_tuple(IDRange(0, num_d),
                                              IZRange(0, num_z),
                                              IGRange(0, num_g),
                                              IMRange(0, num_m)),
          [=](ID d, IZ z, IG g, IM m) {
          LTIMES_BODY_RAJA;
        });

      }
      stopTimer();

#endif

      break;
    }
#endif

#if 0
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget : {
      // Fill these in later...you get the idea...
      break;
    }
#endif

    default : {
      std::cout << "\n  Unknown variant id = " << vid << std::endl;
    }

  }
}

void LTIMES::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_phidat, m_philen);
}

void LTIMES::tearDown(VariantID vid)
{
  (void) vid;
 
  deallocData(m_phidat);
  deallocData(m_elldat);
  deallocData(m_psidat);
}

} // end namespace apps
} // end namespace rajaperf
