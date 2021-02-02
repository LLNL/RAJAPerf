//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

//Kokkos Design Spirit:
//WE NEED:
//1) Use KokkosViews --> a wrapper around pointers for host and device memory
//management
//2) Use default execution space
//
//
//
//
// NEW FUNCTION WILL:
// 1) Take in a raw pointer (e.g., float*, int*, etc.) 
// 2) From this pointer, return a Kokkos::View
//
// Return type :  Kokkos::View
// Kokkos::View takes tempalted arguments
// To write "generically" implies templated arguments
// https://eli.thegreenplace.net/2014/variadic-templates-in-c/
//
template<class PointedAt, size_t NumBoundaries>


// This is a TEMPLATED STRUCT.  This struct will contain the type of a pointer of n dimensions 
// This struct is templated on the template<class PointedAt, size_t
// NumBoundaries> that immediately precedes the struct declaration.
struct PointerOfNdimensions;

// This template block declares a specialization, which means that you say the
// template arguments that you're NOT specializing
template<class PointedAt>

// Here, we are specialising a template according to the type of argument that
// is passed.  In this example, we've specialized the PointedAt template
// argument for the case that the number of dimensions is 0.  All we will do in
// this struct is to define a type.

// This struct is a specialization of :
// template<class PointedAt, size_t NumBoundaries>
struct PointerOfNdimensions <PointedAt, 0>{
	// "using" is a type alias
	// if you derefernce a pointer, you're just left with an object, the value
	// of that pointer
	using type = PointedAt;
};

// NO SPECIALIZATION, i.e., we fix no templated arguments
template<class PointedAt, size_t NumBoundaries>

struct PointerOfNdimensions {
	// PointerOfNdimensions is a type
	// My type is a pointer to the type of myself, decremented
	using type = typename PointerOfNdimensions<PointedAt, NumBoundaries - 1>::type*;	

};


template<class PointedAt, class... Boundaries>

// FUNCTION THAT GETS A VIEW FROM A POINTER WITH RETURN TYPE KOKKOS::VIEW
//
auto getViewFromPointer(PointedAt* kokkos_ptr, Boundaries... boundaries)
	// Recall:  PointerOfNdimensions is struct that exists solely to hold a
	// type
	// -> connotes "return type after the arrow"
	-> typename Kokkos::View<
					typename PointerOfNdimensions <PointedAt, sizeof...(Boundaries)>::type,
					//typename Kokkos::DefaultHostExecutionSpace::memory_space>
					//This more generic expression allow moving the
					//View-wrapped pointer b/w
					//Host and GPU
					typename Kokkos::DefaultExecutionSpace::memory_space>


{
	// This says construct the pointer_holder variable from arguments passed to
	// the template block
	//
	using host_view_type = typename Kokkos::View<
					typename PointerOfNdimensions <PointedAt, sizeof...(Boundaries)>::type,
					typename Kokkos::DefaultHostExecutionSpace::memory_space>;

	// FYI - Device can be GPU, OpenMPTarget, HIP (for targeting an AMD GPU), SYCL (library in Intel
	// Compiler)
	//
	using device_view_type = typename Kokkos::View<
					typename PointerOfNdimensions <PointedAt, sizeof...(Boundaries)>::type,
					typename Kokkos::DefaultExecutionSpace::memory_space>;



	// When copying data, we can either change the Layout or the memory_space
	// (host or device), but we cannot change both!
	// Here, we are mirroring data on the host to the device, i.e., Layout is
	// as if on the device, but the data is actually on the host.  The host
	// mirror will be Layout Left (optimal for the device), but data are
	// actually on the HOST!

	// Here, "using" is type alias; in this example,its our gpu Layout on cpu
	using mirror_view_type = typename device_view_type::HostMirror;

	// Assignment statement; we are constructing a host_view_type with the name  pointer_holder.  The value of kokkos_ptr
	// is the pointer we're wrapping on the Host, and the Boundaries parameter
	// pack values, boundaries, will also be part of this this host_view_type
	// object.
	
	host_view_type pointer_holder (kokkos_ptr, boundaries...);
	
	// boundaries will contain the array dimenions; an allocation is implicitly made here
	device_view_type device_data_copy ( "StringName", boundaries...);

	mirror_view_type cpu_to_gpu_mirror = Kokkos::create_mirror_view(device_data_copy);

	// We need to deep_copy our existing data, the contents of
	// pointer_holder, into the mirror_view;
	// Copying from Host to Device has two steps:  1) Change the layout, 2)
	// change the memory_space (host or device).  Step 1 is to change the
	// layout to enable sending data from CPU to GPU. Step 2 is actually
	// sending the optimal data layout to the GPU
	
	// This step changes the Layout to be optimal for the gpu
	Kokkos::deep_copy(cpu_to_gpu_mirror, pointer_holder);


	// The mirror view data layout on the HOST is like the layout for the GPU.  GPU-optimized layouts are LayoutLeft,
	// i.e., column-major
	// This deep_copy copy GPU-layout data on the HOST to the Device

	// Actual copying of the data from the host to the gpu
	Kokkos::deep_copy(device_data_copy, cpu_to_gpu_mirror);


	// Kokkos::View return type

	return device_data_copy;

}

///////////////////////////////////////////////////////////////////////////////
//THIS FUNCTION WILL MOVE DATA IN A KOKKOS::VIEW BACK TO HOST FROM DEVICE, AND
//STORE IN AN EXISTING POINTER
///////////////////////////////////////////////////////////////////////////////



template<class PointedAt, class ExistingView, class... Boundaries>

// DEFINING FUNCTION THAT GETS A VIEW FROM A POINTER WITH RETURN TYPE KOKKOS::VIEW
//"my_view" parameter is equivalent to device_data_copy
//
void moveDataToHostFromKokkosView(PointedAt* kokkos_ptr, ExistingView my_view, Boundaries... boundaries)

{
	// This says construct the pointer_holder variable from arguments passed to
	// the template block
	//
	using host_view_type = typename Kokkos::View<
					typename PointerOfNdimensions <PointedAt, sizeof...(Boundaries)>::type,
					typename Kokkos::DefaultHostExecutionSpace::memory_space>;

	// FYI - Device can be GPU, OpenMPTarget, HIP (for targeting an AMD GPU), SYCL (library in Intel
	// Compiler)
	//
	using device_view_type = typename Kokkos::View<
					typename PointerOfNdimensions <PointedAt, sizeof...(Boundaries)>::type,
					typename Kokkos::DefaultExecutionSpace::memory_space>;



	// When copying data, we can either change the Layout or the memory_space
	// (host or device), but we cannot change both!
	// Here, we are mirroring data on the host to the device, i.e., Layout is
	// as if on the device, but the data is actually on the host.  The host
	// mirror will be Layout Left (optimal for the device), but data are
	// actually on the HOST!

	// Here, "using" is type alias; in this example,its our gpu Layout on cpu
	using mirror_view_type = typename device_view_type::HostMirror;

	// Assignment statement; we are constructing a host_view_type with the name  pointer_holder.  The value of kokkos_ptr
	// is the pointer we're wrapping on the Host, and the Boundaries parameter
	// pack values, boundaries, will also be part of this this host_view_type
	// object.
	
	host_view_type pointer_holder (kokkos_ptr, boundaries...);

	// Layout is optimal for gpu, but located on CPU
	mirror_view_type cpu_to_gpu_mirror = Kokkos::create_mirror_view(my_view);

	

	// We need to deep_copy our existing data, the contents of
	// pointer_holder, into the mirror_view;
	// Copying from Host to Device has two steps:  1) Change the layout, 2)
	// change the memory_space (host or device).  Step 1 is to change the
	// layout to enable sending data from CPU to GPU. Step 2 is actually
	// sending the optimal data layout to the GPU
	
	// This step changes the Layout to be optimal for the gpu


	// The mirror view data layout on the HOST is like the layout for the GPU.  GPU-optimized layouts are LayoutLeft,
	// i.e., column-major
	// This deep_copy copy GPU-layout data on the HOST to the Device

	// Actual copying of the data from the gpu to the cpu
	Kokkos::deep_copy(cpu_to_gpu_mirror, my_view);
	
	//This copies from the mirror on the cpu 
	Kokkos::deep_copy(pointer_holder, cpu_to_gpu_mirror);

}


//////////////////////////////////////////////////////////////////////////////

void NESTED_INIT::runKokkosVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();


  NESTED_INIT_DATA_SETUP;

  auto nestedinit_lam = [=](Index_type i, Index_type j, Index_type k) {
                          NESTED_INIT_BODY;
                        };

#if defined RUN_KOKKOS

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = 0; k < nk; ++k ) {
          for (Index_type j = 0; j < nj; ++j ) {
            for (Index_type i = 0; i < ni; ++i ) {
              NESTED_INIT_BODY;
            }
          }
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

          for (Index_type k = 0; k < nk; ++k ) {
            for (Index_type j = 0; j < nj; ++j ) {
              for (Index_type i = 0; i < ni; ++i ) {
                nestedinit_lam(i, j, k);
              }
            }
          }

      }
      stopTimer();

      break;
    }

// Kokkos_Lambda variant

    case Kokkos_Lambda : {

  	  // Wrap the nested init array pointer in a Kokkos View
      // In  a Kokkos View, array arguments for array boundaries go from outmost
  	  // to innermost dimension sizes
      // See the basic NESTED_INIT.hpp file for defnition of NESTED_INIT
 
      auto array_kokkos_view = getViewFromPointer(array, nk, nj, ni);
	
      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {


	// MDRange can be optimized
	 
	Kokkos::parallel_for("NESTED_INIT KokkosSeq",
						 // Range policy
						 Kokkos::MDRangePolicy<Kokkos::Rank<3>,
						 // Execution space
                         Kokkos::DefaultExecutionSpace>({0,0,0}, {nk,nj,ni}),
						 // Loop body
                         KOKKOS_LAMBDA(Index_type k, Index_type j, Index_type i) {
						 //NESTED_INIT_BODY no longer useful, because we're not
						 //operating on the array, but on the Kokkos::View
						// array_kokkos_view created to hold value for
						// getViewFromPointer(array, nk, nj, ni)
						// MD Views are index'ed via "()"
						//
						// KOKKOS-FIED translation of NESTED_INIT_BODY:
						// #define NESTED_INIT_BODY
						// array[i+ni*(j+nj*k)] = 0.00000001 * i * j * k ;
						//
						array_kokkos_view(k, j, i) = 0.00000001 * i * j * k;
                        }
);
 	
      }
      stopTimer();
	  // "Moves" mirror data from GPU to CPU (void, i.e., no retrun type).  In
	  // this moving of data back to Host, the layout is changed back to Layout
	  // Right, vs. the LayoutLeft of the GPU
      moveDataToHostFromKokkosView(array, array_kokkos_view, nk, nj, ni);

      break;
    }
#endif // RUN_RAJA_SEQ

    default : {
      std::cout << "\n  NESTED_INIT : Unknown variant id = " << vid << std::endl;
    }

  }
#endif //RUN_KOKKOS
}

} // end namespace basic
} // end namespace rajaperf
