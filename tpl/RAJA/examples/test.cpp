#include "RAJA/RAJA.hpp"
#include <iostream>

using namespace RAJA;
using Real_type = double;

RAJA_INDEX_VALUE_T(NI, int, "NI");
RAJA_INDEX_VALUE_T(NJ, int, "NJ");
RAJA_INDEX_VALUE_T(NK, int, "NK");
RAJA_INDEX_VALUE_T(NL, int, "NL");

int main()
{
    const int ni=16; 
    const int nj=18; 
    const int nk=22; 
    const int nl=24;
    const Real_type alpha = 1.5;
    const Real_type beta = 1.2;

    const int tmpSize = ni * nj;
    const int ASize = ni * nk;
    const int BSize = nk * nj;
    const int CSize = nj * nl;
    const int DSize = nl * ni;

    std::vector<Real_type> tmp_vec(ni * nj);
    std::vector<Real_type> A_vec(ni * nk);
    std::vector<Real_type> B_vec(nk * nj);
    std::vector<Real_type> C_vec(nj * nl);
    std::vector<Real_type> D_vec(ni * nl, 0);

    Real_type* tmpData = &tmp_vec[0];
    Real_type* AData = &A_vec[0];
    Real_type* BData = &B_vec[0];
    Real_type* CData = &C_vec[0];
    Real_type* DData = &D_vec[0];

    for(int i = 0; i < tmpSize; i++)
	tmpData[i] = i + 1;

    for(int i = 0; i < ASize; i++)
	AData[i] = 2 * i + 1;

    for(int i = 0; i < BSize; i++)
	BData[i] = 3 * i + 1;

    for(int i = 0; i < CSize; i++)
	CData[i] = 4 * i + 1;

    //using TMP_VIEW = RAJA::TypedView<Real_type, RAJA::Layout<2, int, 1>, NI, NJ>;
    //std::array<RAJA::idx_t, 2> tmp_perm {{0, 1}};
    //TMP_VIEW tmp(tmpData, RAJA::make_permuted_layout({{ni, nj}}, tmp_perm));

    //using A_VIEW = RAJA::TypedView<Real_type, RAJA::Layout<2, int, 1>, NI, NK>;
    //std::array<RAJA::idx_t, 2> a_perm {{0, 1}};
    //A_VIEW A(AData, RAJA::make_permuted_layout({{ni, nk}}, a_perm));

    //using B_VIEW = RAJA::TypedView<Real_type, RAJA::Layout<2, int, 1>, NK, NJ>;
    //std::array<RAJA::idx_t, 2> b_perm {{0, 1}};
    //B_VIEW B(BData, RAJA::make_permuted_layout({{nk, nj}}, b_perm));

    //using C_VIEW = RAJA::TypedView<Real_type, RAJA::Layout<2, int, 1>, NJ, NL>;
    //std::array<RAJA::idx_t, 2> c_perm {{0, 1}};
    //C_VIEW C(CData, RAJA::make_permuted_layout({{nj, nl}}, c_perm));

    //using D_VIEW = RAJA::TypedView<Real_type, RAJA::Layout<2, int, 1>, NI, NL>;
    //std::array<RAJA::idx_t, 2> d_perm {{0, 1}};
    //D_VIEW D(DData, RAJA::make_permuted_layout({{ni, nl}}, d_perm));
    //
    using TMP_VIEW = RAJA::View<Real_type, RAJA::Layout<2, int, 1>>;
    TMP_VIEW tmp(tmpData, RAJA::Layout<2>(ni, nj));
    TMP_VIEW A(AData, RAJA::Layout<2>(ni, nk));
    TMP_VIEW B(BData, RAJA::Layout<2>(nk, nj));
    TMP_VIEW C(CData, RAJA::Layout<2>(nj, nl));
    TMP_VIEW D(DData, RAJA::Layout<2>(ni, nl));

    using EXEC_POL =
    RAJA::KernelPolicy<
     RAJA::statement::For<0, RAJA::loop_exec,
      RAJA::statement::For<1, RAJA::loop_exec,
       RAJA::statement::Lambda<0, RAJA::Segs<0,0,0>, Params<0>>,
        RAJA::statement::For<2, RAJA::loop_exec,
         RAJA::statement::Lambda<1, RAJA::Segs<0,1,2>, Params<0>>,
          RAJA::statement::Lambda<2, RAJA::Segs<0,1>, Params<0>>
           //RAJA::statement::Lambda<3, Params<0>>
        >
         //RAJA::statement::For<3, RAJA::loop_exec,
         // RAJA::statement::Lambda<4, RAJA::Segs<0,1,3>, Params<0>>,
         //  RAJA::statement::Lambda<5, RAJA::Segs<0,3>, Params<0>>
         //>
      >
     >
    >;

    auto segments = RAJA::make_tuple(RAJA::RangeSegment(0, ni),
                                     RAJA::RangeSegment(0, nj),
                                     RAJA::RangeSegment(0, nk));

    RAJA::kernel_param<EXEC_POL>(segments, RAJA::tuple<Real_type>{0.0},
        [=] (RAJA::Index_type, RAJA::Index_type, RAJA::Index_type, Real_type& dot) {
            dot = 0;
        },

        [=](RAJA::Index_type i, RAJA::Index_type j, RAJA::Index_type k, Real_type& dot) {
            dot += alpha * A(i,k) * B(k,j);
        },

        [=](RAJA::Index_type i, RAJA::Index_type j, Real_type& dot) {
            tmp(i,j) += dot;
        }
    );

    //auto segment1 = RAJA::make_tuple(RAJA::TypedRangeSegment<NI>(0, ni),
    //                                 RAJA::TypedRangeSegment<NL>(0, nl),
    //                                 RAJA::TypedRangeSegment<NJ>(0, nj));

    //RAJA::kernel_param<EXEC_POL>(segment1, RAJA::tuple<Real_type>{0.0},

    //    [=](Real_type& dot) {
    //        dot = beta;
    //    },

    //    [=](NI i, NL l, NJ j, Real_type& dot) {
    //        dot += tmp(i,j) * C(j,l);
    //    },

    //    [=](NI i, NL l, Real_type& dot) {
    //        D(i,l) = dot;
    //    }
    //);


    //auto segments = RAJA::make_tuple(RAJA::TypedRangeSegment<NI>(0, ni),
    //                                 RAJA::TypedRangeSegment<NJ>(0, nj),
    //                                 RAJA::TypedRangeSegment<NK>(0, nk),
    //                                 RAJA::TypedRangeSegment<NL>(0, nl));
    //RAJA::ReduceSum<RAJA::seq_reduce, Real_type> seqdot(0.0);

    //RAJA::kernel_param<EXEC_POL>(segments, RAJA::tuple<Real_type>{0.0},
    //    [=] (Real_type& dot) {
    //        dot = 0;
    //    },

    //    [=](NI i, NJ j, NK k, Real_type& dot) {
    //        dot += alpha * A(i,k) * B(k,j);
    //    },

    //    [=](NI i, NJ j, Real_type& dot) {
    //        tmp(i,j) += dot;
    //    },

    //    [=](Real_type& dot) {
    //        dot = beta;
    //    },

    //    [=](NI i, NJ j, NL l, Real_type& dot) {
    //        dot += tmp(i,j) * C(j,l);
    //    },

    //    [=](NI i, NL l, Real_type& dot) {
    //        D(i,l) = dot;
    //    }
    //);

    //for(NI i(0); i < ni; i++)
    //{
    //    for(NJ j(0); j < nj; j++)
    //        std::cout << tmp(i,j) << std::endl;
    //}

    return 0;
}
