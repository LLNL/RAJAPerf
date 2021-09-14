//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "VOL3D.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf 
{
namespace apps
{


struct arrayOffSetStruct3D {
  using ViewType = Kokkos::View<Real_ptr>;

  // v's are offsets of indices
  ViewType v, v0, v1, v2, v3, v4, v5, v6, v7;

  // constructor
    arrayOffSetStruct3D(const std::string& name,
                      Index_type num_elements,
                      Index_type jp,
                      Index_type kp,
                      Real_ptr head
                    ):
            // ":" = list of things to initialize
            // Initialize v 
            v (getViewFromPointer(head, num_elements)), 
            v0(v),
            v1(Kokkos::subview(v0, std::make_pair(static_cast<unsigned long>(1), v0.extent(0)))),
            v2(Kokkos::subview(v0, std::make_pair(static_cast<unsigned long>(jp), v0.extent(0)))),
            v3(Kokkos::subview(v1, std::make_pair(static_cast<unsigned long>(jp), v1.extent(0)))),
            v4(Kokkos::subview(v0, std::make_pair(static_cast<unsigned long>(kp), v0.extent(0)))),
            v5(Kokkos::subview(v1, std::make_pair(static_cast<unsigned long>(kp), v1.extent(0)))),
            v6(Kokkos::subview(v2, std::make_pair(static_cast<unsigned long>(kp), v2.extent(0)))),
            v7(Kokkos::subview(v3, std::make_pair(static_cast<unsigned long>(kp), v3.extent(0)))) {
                    }
};


void VOL3D::runKokkosVariant(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = m_domain->fpz;
  const Index_type iend = m_domain->lpz+1;

  VOL3D_DATA_SETUP;

  NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
  NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ;
  NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;

  // not sure about the 'ibegin, iend' here:
  auto vol_view = getViewFromPointer(vol, m_domain->nnalls);

  arrayOffSetStruct3D x_offsets("x_offsets", m_domain->nnalls, m_domain->jp, m_domain->kp, x); 
  arrayOffSetStruct3D y_offsets("y_offsets", m_domain->nnalls, m_domain->jp, m_domain->kp, y); 
  arrayOffSetStruct3D z_offsets("z_offsets", m_domain->nnalls, m_domain->jp, m_domain->kp, z);

  auto& x_view = x_offsets.v;
  auto& x0_view = x_offsets.v0;
  auto& x1_view = x_offsets.v1;
  auto& x2_view = x_offsets.v2;
  auto& x3_view = x_offsets.v3;
  auto& x4_view = x_offsets.v4;
  auto& x5_view = x_offsets.v5;
  auto& x6_view = x_offsets.v6;
  auto& x7_view = x_offsets.v7;

  auto& y_view = y_offsets.v;
  auto& y0_view = y_offsets.v0;
  auto& y1_view = y_offsets.v1;
  auto& y2_view = y_offsets.v2;
  auto& y3_view = y_offsets.v3;
  auto& y4_view = y_offsets.v4;
  auto& y5_view = y_offsets.v5;
  auto& y6_view = y_offsets.v6;
  auto& y7_view = y_offsets.v7;

  auto& z_view = z_offsets.v;
  auto& z0_view = z_offsets.v0;
  auto& z1_view = z_offsets.v1;
  auto& z2_view = z_offsets.v2;
  auto& z3_view = z_offsets.v3;
  auto& z4_view = z_offsets.v4;
  auto& z5_view = z_offsets.v5;
  auto& z6_view = z_offsets.v6;
  auto& z7_view = z_offsets.v7;


  auto vol3d_lam = [=](Index_type i) {
                     VOL3D_BODY;
                   };

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin ; i < iend ; ++i ) {
          VOL3D_BODY;
        }

      }
      stopTimer();

      break;
    } 

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin ; i < iend ; ++i ) {
          vol3d_lam(i);
        }

      }
      stopTimer();

      break;
    }
/*
    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::loop_exec>(
          RAJA::RangeSegment(ibegin, iend), vol3d_lam);

      }
      stopTimer(); 

      break;
    }
*/
    case Kokkos_Lambda : {

      startTimer();

      //auto index_list = getViewFromPointer(m_domain->real_zones, m_domain->n_real_zones);

      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

              Kokkos::parallel_for(
                              "VOL3D Kokkos_Lambda",
                              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin,iend),
                              KOKKOS_LAMBDA(Index_type i ) {

                              // #define VOL3D_BODY
                              //int i = index_list[ii];

                                  Real_type x71 = x7_view[i] - x1_view[i] ; \
                                  Real_type x72 = x7_view[i] - x2_view[i] ; \
                                  Real_type x74 = x7_view[i] - x4_view[i] ; \
                                  Real_type x30 = x3_view[i] - x0_view[i] ; \
                                  Real_type x50 = x5_view[i] - x0_view[i] ; \
                                  Real_type x60 = x6_view[i] - x0_view[i] ; \
                                 
                                  Real_type y71 = y7_view[i] - y1_view[i] ; \
                                  Real_type y72 = y7_view[i] - y2_view[i] ; \
                                  Real_type y74 = y7_view[i] - y4_view[i] ; \
                                  Real_type y30 = y3_view[i] - y0_view[i] ; \
                                  Real_type y50 = y5_view[i] - y0_view[i] ; \
                                  Real_type y60 = y6_view[i] - y0_view[i] ; \
                                 
                                  Real_type z71 = z7_view[i] - z1_view[i] ; \
                                  Real_type z72 = z7_view[i] - z2_view[i] ; \
                                  Real_type z74 = z7_view[i] - z4_view[i] ; \
                                  Real_type z30 = z3_view[i] - z0_view[i] ; \
                                  Real_type z50 = z5_view[i] - z0_view[i] ; \
                                  Real_type z60 = z6_view[i] - z0_view[i] ; \
                                 
                                  Real_type xps = x71 + x60 ; \
                                  Real_type yps = y71 + y60 ; \
                                  Real_type zps = z71 + z60 ; \
                                 
                                  Real_type cyz = y72 * z30 - z72 * y30 ; \
                                  Real_type czx = z72 * x30 - x72 * z30 ; \
                                  Real_type cxy = x72 * y30 - y72 * x30 ; \
                                  vol_view[i] = xps * cyz + yps * czx + zps * cxy ; \
                                 
                                  xps = x72 + x50 ; \
                                  yps = y72 + y50 ; \
                                  zps = z72 + z50 ; \
                                 
                                  cyz = y74 * z60 - z74 * y60 ; \
                                  czx = z74 * x60 - x74 * z60 ; \
                                  cxy = x74 * y60 - y74 * x60 ; \
                                  vol_view[i] += xps * cyz + yps * czx + zps * cxy ; \
                                 
                                  xps = x74 + x30 ; \
                                  yps = y74 + y30 ; \
                                  zps = z74 + z30 ; \
                                 
                                  cyz = y74 * z60 - z74 * y60 ; \
                                  czx = z74 * x60 - x74 * z60 ; \
                                  cxy = x74 * y60 - y74 * x60 ; \
                                  vol_view[i] += xps * cyz + yps * czx + zps * cxy ; \
                                 
                                  xps = x74 + x30 ; \
                                  yps = y74 + y30 ; \
                                  zps = z74 + z30 ; \
                                 
                                  cyz = y71 * z50 - z71 * y50 ; \
                                  czx = z71 * x50 - x71 * z50 ; \
                                  cxy = x71 * y50 - y71 * x50 ; \
                                  vol_view[i] += xps * cyz + yps * czx + zps * cxy ; \
                                 
                                  vol_view[i] *= vnormq ;
                              }
              );

      }
      stopTimer(); 

      break;
    }

    default : {
      std::cout << "\n  VOL3D : Unknown variant id = " << vid << std::endl;
    }

  }

#endif // RUN_KOKKOS

  moveDataToHostFromKokkosView(x, x_view, m_domain->nnalls);
  moveDataToHostFromKokkosView(y, y_view, m_domain->nnalls);
  moveDataToHostFromKokkosView(z, z_view, m_domain->nnalls);
  moveDataToHostFromKokkosView(vol, vol_view, m_domain->nnalls);


}

} // end namespace apps
} // end namespace rajaperf
