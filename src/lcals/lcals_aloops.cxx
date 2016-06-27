#include "gtest/gtest.h"

#include "RAJA/RAJA.hxx"

template <typename T>
class LcalsTest : public ::testing::Test {
    virtual void SetUp() {
    }

    virtual void TearDown() {
    }
};

TYPED_TEST_CASE_P(LcalsTest);

TYPED_TEST_P(LcalsTest, PressureCalc) {
  ASSERT_EQ(true, true);

  // loopInit(iloop, stat);

  // Real_ptr compression = // loop_data.array_1D_Real[0];
  // Real_ptr bvc = // loop_data.array_1D_Real[1];
  // Real_ptr p_new = // loop_data.array_1D_Real[2];
  // Real_ptr e_old = // loop_data.array_1D_Real[3];
  // Real_ptr vnewc = // loop_data.array_1D_Real[4];

  // const Real_type cls = 1.0; // loop_data.scalar_Real[0];
  // const Real_type p_cut = 0.5; // loop_data.scalar_Real[1];
  // const Real_type pmin = 0.0001; // loop_data.scalar_Real[2];
  // const Real_type eosvmax = 11.0; // loop_data.scalar_Real[3];

  // for (SampIndex_type isamp = 0; isamp < num_samples; ++isamp) {

  //    forall<exec_policy>(0, len,
  //    [&] (Index_type i) {
  //      bvc[i] = cls * (compression[i] + 1.0);
  //    } );

  //    forall<exec_policy>(0, len,
  //    [&] (Index_type i) {
  //       p_new[i] = bvc[i] * e_old[i] ;

  //       if ( fabs(p_new[i]) <  p_cut )  p_new[i] = 0.0 ;

  //       if ( vnewc[i] >= eosvmax )  p_new[i] = 0.0 ;

  //       if ( p_new[i]  <  pmin )  p_new[i] = pmin ;
  //    } );

  // }
}

REGISTER_TYPED_TEST_CASE_P(LcalsTest, PressureCalc);

typedef ::testing::Types<RAJA::seq_exec, RAJA::simd_exec> SequentialTypes; 
INSTANTIATE_TYPED_TEST_CASE_P(Sequential, LcalsTest, SequentialTypes);

#if defined(RAJA_ENABLE_OPENMP)
typedef ::testing::Types<RAJA::omp_parallel_for_exec> OpenMPTypes;
INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, LcalsTest, OpenMPTypes);
#endif
 
#if defined(RAJA_ENABLE_CILK)
typedef ::testing::Types<RAJA::cilk_for_exec> CilkTypes;
INSTANTIATE_TYPED_TEST_CASE_P(Cilk, LcalsTest, CilkTypes);
#endif
