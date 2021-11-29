//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DEL_DOT_VEC_2D.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"

#include "camp/resource.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

struct arrayOffSetStruct {
  using ViewType = Kokkos::View<Real_ptr>; // Real_ptr is equivalent to float*

  // v's represent different offsets in different Kokkos views;
  ViewType v, v4, v1, v2, v3;

  // constructor
    arrayOffSetStruct(const std::string& name, // we needed a name, for future efforts 
                      Index_type num_elements, // alloc size of head;
                      Index_type jp, // their macro took in jp, so we're using it
                      Real_ptr head // v, approximately;
                    ):
            // ":" = list of things to initialize
            v (getViewFromPointer(head, num_elements)),
            // Initializing v4 with v 
            v4(v),
            v1(Kokkos::subview(v4, std::make_pair(static_cast<unsigned long>(1), v4.extent(0)))),
            v2(Kokkos::subview(v1, std::make_pair(static_cast<unsigned long>(jp), v1.extent(0)))),
            v3(Kokkos::subview(v4, std::make_pair(static_cast<unsigned long>(jp), v4.extent(0)))){
                    }
};

void DEL_DOT_VEC_2D::runKokkosVariant(VariantID vid) {

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  DEL_DOT_VEC_2D_DATA_SETUP;

  NDSET2D(m_domain->jp, x, x1, x2, x3, x4);
  NDSET2D(m_domain->jp, y, y1, y2, y3, y4);
  NDSET2D(m_domain->jp, xdot, fx1, fx2, fx3, fx4);
  NDSET2D(m_domain->jp, ydot, fy1, fy2, fy3, fy4);

  // Instantiating Kokkos view
  auto div_view = getViewFromPointer(div, m_domain->nnalls);

  arrayOffSetStruct x_offsets("x_offsets", m_domain->nnalls, m_domain->jp, x );
  arrayOffSetStruct y_offsets("y_offsets", m_domain->nnalls, m_domain->jp, y );
  arrayOffSetStruct xdot_offsets("xdot_offsets", m_domain->nnalls, m_domain->jp, xdot );
  arrayOffSetStruct ydot_offsets("ydot_offsets", m_domain->nnalls, m_domain->jp, ydot );

 auto& x_view = x_offsets.v;
 auto& x1_view = x_offsets.v1;
 auto& x2_view = x_offsets.v2;
 auto& x3_view = x_offsets.v3;
 auto& x4_view = x_offsets.v4;


 auto& y_view = y_offsets.v;
 auto& y1_view = y_offsets.v1;
 auto& y2_view = y_offsets.v2;
 auto& y3_view = y_offsets.v3;
 auto& y4_view = y_offsets.v4;


 auto& xdot_view = xdot_offsets.v;
 auto& fx1_view = xdot_offsets.v1;
 auto& fx2_view = xdot_offsets.v2;
 auto& fx3_view = xdot_offsets.v3;
 auto& fx4_view = xdot_offsets.v4;


 auto& ydot_view = ydot_offsets.v;
 auto& fy1_view = ydot_offsets.v1;
 auto& fy2_view = ydot_offsets.v2;
 auto& fy3_view = ydot_offsets.v3;
 auto& fy4_view = ydot_offsets.v4;


#if defined(RUN_KOKKOS)
  switch (vid) {

  case Kokkos_Lambda: {

    // Translation from RAJAPerf Suite to Kokkos notes:
    // Host resource will be used for loop execution
    // camp::resources::Resource working_res{camp::resources::Host()};

    // List segment = indices you're iterating over are contained in lists;

    /*      RAJA::TypedListSegment<Index_type> zones(m_domain->real_zones,
                                                   m_domain->n_real_zones,
                                                   working_res);
    */

    auto index_list =
        getViewFromPointer(m_domain->real_zones, m_domain->n_real_zones);

    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // RAJA::forall<RAJA::loop_exec>(zones, deldotvec2d_lam);
      Kokkos::parallel_for(
          "DEL_DOT_VEC_2D Kokkos_Lambda",
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(ibegin, iend),
          KOKKOS_LAMBDA(Index_type ii) {
            // #define DEL_DOT_VEC_2D_BODY
            int i = index_list[ii];

            // Real_type xi  = half * ( x1[i]  + x2[i]  - x3[i]  - x4[i]  ) ;
            Real_type xi =
                half * (x1_view[i] + x2_view[i] - x3_view[i] -
                        x4_view[i]); // Real_type xj  = half * ( x2[i]  + x3[i]
                                     // - x4[i]  - x1[i]  ) ;
            Real_type xj =
                half * (x2_view[i] + x3_view[i] - x4_view[i] -
                        x1_view[i]); // Real_type yi  = half * ( y1[i]  + y2[i]
                                     // - y3[i]  - y4[i]  ) ;
            Real_type yi =
                half * (y1_view[i] + y2_view[i] - y3_view[i] -
                        y4_view[i]); // Real_type yj  = half * ( y2[i]  + y3[i]
                                     // - y4[i]  - y1[i]  ) ;
            Real_type yj =
                half * (y2_view[i] + y3_view[i] - y4_view[i] -
                        y1_view[i]); // Real_type fxi = half * ( fx1[i] + fx2[i]
                                     // - fx3[i] - fx4[i] ) ;
            Real_type fxi =
                half * (fx1_view[i] + fx2_view[i] - fx3_view[i] -
                        fx4_view[i]); // Real_type fxj = half * ( fx2[i] +
                                      // fx3[i] - fx4[i] - fx1[i] ) ;
            Real_type fxj =
                half * (fx2_view[i] + fx3_view[i] - fx4_view[i] -
                        fx1_view[i]); // Real_type fyi = half * ( fy1[i] +
                                      // fy2[i] - fy3[i] - fy4[i] ) ;
            Real_type fyi =
                half * (fy1_view[i] + fy2_view[i] - fy3_view[i] -
                        fy4_view[i]); // Real_type fyj = half * ( fy2[i] +
                                      // fy3[i] - fy4[i] - fy1[i] ) ;
            Real_type fyj =
                half * (fy2_view[i] + fy3_view[i] - fy4_view[i] -
                        fy1_view[i]); // Real_type rarea  = 1.0 / ( xi * yj - xj
                                      // * yi + ptiny ) ;
            Real_type rarea =
                1.0 /
                (xi * yj - xj * yi +
                 ptiny); // Real_type dfxdx  = rarea * ( fxi * yj - fxj * yi ) ;
            Real_type dfxdx =
                rarea * (fxi * yj - fxj * yi); // Real_type dfydy  = rarea * (
                                               // fyj * xi - fyi * xj ) ;
            Real_type dfydy =
                rarea * (fyj * xi - fyi * xj); /* Real_type affine = ( fy1[i] +
                                                  fy2[i] + fy3[i] + fy4[i] ) / \
                                                                  ( y1[i]  +
                                                  y2[i]  + y3[i]  + y4[i]  ) ; \
                                                                  */
            Real_type affine =
                (fy1_view[i] + fy2_view[i] + fy3_view[i] + fy4_view[i]) /
                (y1_view[i] + y2_view[i] + y3_view[i] +
                 y4_view[i]); //  div[i] = dfxdx + dfydy + affine ;
            div_view[i] = dfxdx + dfydy + affine;
          }

      );
    }
    stopTimer();

    break;
  }

  default: {
    std::cout << "\n  DEL_DOT_VEC_2D : Unknown variant id = " << vid
              << std::endl;
  }
  
}

#endif // RUN_KOKKOS


  moveDataToHostFromKokkosView(x, x_view, m_domain->nnalls);
  moveDataToHostFromKokkosView(y, y_view, m_domain->nnalls);
  moveDataToHostFromKokkosView(xdot, xdot_view, m_domain->nnalls);
  moveDataToHostFromKokkosView(ydot, ydot_view, m_domain->nnalls);
  moveDataToHostFromKokkosView(div, div_view, m_domain->nnalls);

}

} // end namespace apps
} // end namespace rajaperf
