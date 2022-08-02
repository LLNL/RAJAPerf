//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Types and methods for managing Suite kernels, variants, features, etc..
///

#ifndef KokkosViewUtils_HPP
#define KokkosViewUtils_HPP

#include "Kokkos_Core.hpp"

#include <ostream>
#include <string>

namespace rajaperf {
template <class PointedAt, size_t NumBoundaries> struct PointerOfNdimensions;

template <class PointedAt> struct PointerOfNdimensions<PointedAt, 0> {
  using type = PointedAt;
};

template <class PointedAt, size_t NumBoundaries> struct PointerOfNdimensions {
  using type =
      typename PointerOfNdimensions<PointedAt, NumBoundaries - 1>::type *;
};

// This templated function is used to wrap pointers
// (declared and defined in RAJAPerf Suite kernels) in Kokkos Views
//
template <class PointedAt, class... Boundaries>
auto getViewFromPointer(PointedAt *kokkos_ptr, Boundaries... boundaries) ->
     Kokkos::View<
        typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
        typename Kokkos::DefaultExecutionSpace::memory_space>

{

  using host_view_type = typename Kokkos::View<
      typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
      typename Kokkos::DefaultHostExecutionSpace::memory_space>;

  using device_view_type = typename Kokkos::View<
      typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
      typename Kokkos::DefaultExecutionSpace::memory_space>;

  using mirror_view_type = typename device_view_type::HostMirror;

  host_view_type pointer_holder(kokkos_ptr, boundaries...);

  // The boundaries parameter pack contains the array dimenions;
  // An allocation is implicitly made here
  device_view_type device_data_copy("StringName", boundaries...);

  mirror_view_type cpu_to_gpu_mirror =
      Kokkos::create_mirror_view(device_data_copy);

  Kokkos::deep_copy(cpu_to_gpu_mirror, pointer_holder);

  Kokkos::deep_copy(device_data_copy, cpu_to_gpu_mirror);

  // Kokkos::View return type

  return device_data_copy;
}

// This function will move data in a Kokkos::View back to host from device,
// and will be stored in the existing pointer(s)
template <class PointedAt, class ExistingView, class... Boundaries>
void moveDataToHostFromKokkosView(PointedAt *kokkos_ptr, ExistingView my_view,
                                  Boundaries... boundaries) {

  using host_view_type = typename Kokkos::View<
      typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
      typename Kokkos::DefaultHostExecutionSpace::memory_space>;

  using device_view_type = typename Kokkos::View<
      typename PointerOfNdimensions<PointedAt, sizeof...(Boundaries)>::type,
      typename Kokkos::DefaultExecutionSpace::memory_space>;

  using mirror_view_type = typename device_view_type::HostMirror;

  host_view_type pointer_holder(kokkos_ptr, boundaries...);

  // Layout is optimal for gpu, but data are actually located on CPU
  mirror_view_type cpu_to_gpu_mirror = Kokkos::create_mirror_view(my_view);

  // Actual copying of the data from the gpu (my_view) back to the cpu
  Kokkos::deep_copy(cpu_to_gpu_mirror, my_view);

  // This copies from the mirror on the host cpu back to the existing
  // pointer(s)
  Kokkos::deep_copy(pointer_holder, cpu_to_gpu_mirror);
}

} // namespace rajaperf

#endif // closing endif for header file include guard
