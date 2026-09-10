//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NODAL_ACCUMULATION_3D.hpp"

#if defined(RUN_KOKKOS)
#include "common/KokkosViewUtils.hpp"
#include <iostream>
#include "AppsData.hpp"

namespace rajaperf
{
namespace apps
{

void NODAL_ACCUMULATION_3D::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize(); //m_domain->n_real_zones;

  NODAL_ACCUMULATION_3D_DATA_SETUP;

  if ( vid == Kokkos_Lambda ) {
    int jp = m_domain->jp;
    int kp = m_domain->kp;
    auto real_zones_v = getViewFromPointer(real_zones, iend);
    auto vol_v = getViewFromPointer(vol, m_nodal_array_length);
    auto x_v = getViewFromPointer(x, m_nodal_array_length);
   
    using view_t = decltype(x_v);
    view_t x0_v = x_v;
    view_t x1_v(x_v.data() + 1, m_nodal_array_length - 1); 
    view_t x2_v(x_v.data() + jp, m_nodal_array_length - jp); 
    view_t x3_v(x_v.data() + jp, m_nodal_array_length - jp); 
    view_t x4_v(x_v.data() + jp, m_nodal_array_length - jp); 
    view_t x5_v(x_v.data() + kp, m_nodal_array_length - kp); 
    view_t x6_v(x_v.data() + kp, m_nodal_array_length - kp); 
    view_t x7_v(x_v.data() + kp, m_nodal_array_length - kp); 

    Kokkos::fence();
    startTimer();
    // Awkward expression for loop counter quiets C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; ((irep = irep + 1), 0)) {
      RP_CALI_SUBKERNEL_BEGIN("NODAL_ACCUMULATION_3D_1");
      Kokkos::parallel_for("NODAL_ACCUMULATION_3D", iend, KOKKOS_LAMBDA(Index_type ii) {
        Index_type i = real_zones_v(ii);
        Real_type val = 0.125 * vol_v(i);
        Kokkos::atomic_add(&x0_v(i), val);
        Kokkos::atomic_add(&x1_v(i), val);
        Kokkos::atomic_add(&x2_v(i), val);
        Kokkos::atomic_add(&x3_v(i), val);
        Kokkos::atomic_add(&x4_v(i), val);
        Kokkos::atomic_add(&x5_v(i), val);
        Kokkos::atomic_add(&x6_v(i), val);
        Kokkos::atomic_add(&x7_v(i), val);
      });
      RP_CALI_SUBKERNEL_END("NODAL_ACCUMULATION_3D_1");
    }
    Kokkos::fence();
    stopTimer();

    moveDataToHostFromKokkosView(x, x_v, m_nodal_array_length);

  } else {
     getCout() << "\n  NODAL_ACCUMULATION_3D : Unknown Kokkos variant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(NODAL_ACCUMULATION_3D, Kokkos, Kokkos_Lambda)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
