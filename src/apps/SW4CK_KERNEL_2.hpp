//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// SW4CK_KERNEL_2 kernel reference implementation:
///
///
///
int main()
{
  for( int k= kstart; k <= klast-2 ; k++ )
    for( int j=jfirst+2; j <= jlast-2 ; j++ )
      for( int i=ifirst+2; i <= ilast-2 ; i++ )
        {

          // 5 ops
          float_sw4 ijac = strx(i) * stry(j) / jac(i, j, k);
          float_sw4 istry = 1 / (stry(j));
          float_sw4 istrx = 1 / (strx(i));
          float_sw4 istrxy = istry * istrx;

          float_sw4 r1 = 0;
          
          // pp derivative (u)
          // 53 ops, tot=58
          float_sw4 cof1 = (2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
                           met(1, i - 2, j, k) * met(1, i - 2, j, k) *
                           strx(i - 2);
          float_sw4 cof2 = (2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
                           met(1, i - 1, j, k) * met(1, i - 1, j, k) *
                           strx(i - 1);
          float_sw4 cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(1, i, j, k) *
                           met(1, i, j, k) * strx(i);
          float_sw4 cof4 = (2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
                           met(1, i + 1, j, k) * met(1, i + 1, j, k) *
                           strx(i + 1);
          float_sw4 cof5 = (2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
                           met(1, i + 2, j, k) * met(1, i + 2, j, k) *
                           strx(i + 2);
          float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
          float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
          float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
          float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

          r1 += i6 *
                (mux1 * (u(1, i - 2, j, k) - u(1, i, j, k)) +
                 mux2 * (u(1, i - 1, j, k) - u(1, i, j, k)) +
                 mux3 * (u(1, i + 1, j, k) - u(1, i, j, k)) +
                 mux4 * (u(1, i + 2, j, k) - u(1, i, j, k))) *
                istry;
          // qq derivative (u)
          // 43 ops, tot=101
          {
            float_sw4 cof1 = (mu(i, j - 2, k)) * met(1, i, j - 2, k) *
                             met(1, i, j - 2, k) * stry(j - 2);
            float_sw4 cof2 = (mu(i, j - 1, k)) * met(1, i, j - 1, k) *
                             met(1, i, j - 1, k) * stry(j - 1);
            float_sw4 cof3 =
                (mu(i, j, k)) * met(1, i, j, k) * met(1, i, j, k) * stry(j);
            float_sw4 cof4 = (mu(i, j + 1, k)) * met(1, i, j + 1, k) *
                             met(1, i, j + 1, k) * stry(j + 1);
            float_sw4 cof5 = (mu(i, j + 2, k)) * met(1, i, j + 2, k) *
                             met(1, i, j + 2, k) * stry(j + 2);
            float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
            float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
            float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
            float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

            r1 += i6 *
                  (mux1 * (u(1, i, j - 2, k) - u(1, i, j, k)) +
                   mux2 * (u(1, i, j - 1, k) - u(1, i, j, k)) +
                   mux3 * (u(1, i, j + 1, k) - u(1, i, j, k)) +
                   mux4 * (u(1, i, j + 2, k) - u(1, i, j, k))) *
                  istrx;
          }
#ifdef MAGIC_SYNC
	__syncthreads();
#endif
          // rr derivative (u)
          // 5*11+14+14=83 ops, tot=184
          {
            float_sw4 cof1 =
                (2 * mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
                    strx(i) * met(2, i, j, k - 2) * strx(i) +
                mu(i, j, k - 2) * (met(3, i, j, k - 2) * stry(j) *
                                       met(3, i, j, k - 2) * stry(j) +
                                   met(4, i, j, k - 2) * met(4, i, j, k - 2));
            float_sw4 cof2 =
                (2 * mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
                    strx(i) * met(2, i, j, k - 1) * strx(i) +
                mu(i, j, k - 1) * (met(3, i, j, k - 1) * stry(j) *
                                       met(3, i, j, k - 1) * stry(j) +
                                   met(4, i, j, k - 1) * met(4, i, j, k - 1));
            float_sw4 cof3 = (2 * mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) *
                                 strx(i) * met(2, i, j, k) * strx(i) +
                             mu(i, j, k) * (met(3, i, j, k) * stry(j) *
                                                met(3, i, j, k) * stry(j) +
                                            met(4, i, j, k) * met(4, i, j, k));
            float_sw4 cof4 =
                (2 * mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
                    strx(i) * met(2, i, j, k + 1) * strx(i) +
                mu(i, j, k + 1) * (met(3, i, j, k + 1) * stry(j) *
                                       met(3, i, j, k + 1) * stry(j) +
                                   met(4, i, j, k + 1) * met(4, i, j, k + 1));
            float_sw4 cof5 =
                (2 * mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
                    strx(i) * met(2, i, j, k + 2) * strx(i) +
                mu(i, j, k + 2) * (met(3, i, j, k + 2) * stry(j) *
                                       met(3, i, j, k + 2) * stry(j) +
                                   met(4, i, j, k + 2) * met(4, i, j, k + 2));

            float_sw4 mux1 = cof2 - tf * (cof3 + cof1);
            float_sw4 mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
            float_sw4 mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
            float_sw4 mux4 = cof4 - tf * (cof3 + cof5);

            r1 += i6 *
                  (mux1 * (u(1, i, j, k - 2) - u(1, i, j, k)) +
                   mux2 * (u(1, i, j, k - 1) - u(1, i, j, k)) +
                   mux3 * (u(1, i, j, k + 1) - u(1, i, j, k)) +
                   mux4 * (u(1, i, j, k + 2) - u(1, i, j, k))) *
                  istrxy;
          }
          // rr derivative (v)
          // 42 ops, tot=226
          cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
                 met(3, i, j, k - 2);
          cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
                 met(3, i, j, k - 1);
          cof3 =
              (mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) * met(3, i, j, k);
          cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
                 met(3, i, j, k + 1);
          cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
                 met(3, i, j, k + 2);
          mux1 = cof2 - tf * (cof3 + cof1);
          mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
          mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
          mux4 = cof4 - tf * (cof3 + cof5);

          r1 += i6 * (mux1 * (u(2, i, j, k - 2) - u(2, i, j, k)) +
                      mux2 * (u(2, i, j, k - 1) - u(2, i, j, k)) +
                      mux3 * (u(2, i, j, k + 1) - u(2, i, j, k)) +
                      mux4 * (u(2, i, j, k + 2) - u(2, i, j, k)));

          // rr derivative (w)
          // 43 ops, tot=269
          cof1 = (mu(i, j, k - 2) + la(i, j, k - 2)) * met(2, i, j, k - 2) *
                 met(4, i, j, k - 2);
          cof2 = (mu(i, j, k - 1) + la(i, j, k - 1)) * met(2, i, j, k - 1) *
                 met(4, i, j, k - 1);
          cof3 =
              (mu(i, j, k) + la(i, j, k)) * met(2, i, j, k) * met(4, i, j, k);
          cof4 = (mu(i, j, k + 1) + la(i, j, k + 1)) * met(2, i, j, k + 1) *
                 met(4, i, j, k + 1);
          cof5 = (mu(i, j, k + 2) + la(i, j, k + 2)) * met(2, i, j, k + 2) *
                 met(4, i, j, k + 2);
          mux1 = cof2 - tf * (cof3 + cof1);
          mux2 = cof1 + cof4 + 3 * (cof3 + cof2);
          mux3 = cof2 + cof5 + 3 * (cof4 + cof3);
          mux4 = cof4 - tf * (cof3 + cof5);

          r1 += i6 *
                (mux1 * (u(3, i, j, k - 2) - u(3, i, j, k)) +
                 mux2 * (u(3, i, j, k - 1) - u(3, i, j, k)) +
                 mux3 * (u(3, i, j, k + 1) - u(3, i, j, k)) +
                 mux4 * (u(3, i, j, k + 2) - u(3, i, j, k))) *
                istry;

          // pq-derivatives
          // 38 ops, tot=307
          r1 +=
              c2 *
                  (mu(i, j + 2, k) * met(1, i, j + 2, k) * met(1, i, j + 2, k) *
                       (c2 * (u(2, i + 2, j + 2, k) - u(2, i - 2, j + 2, k)) +
                        c1 * (u(2, i + 1, j + 2, k) - u(2, i - 1, j + 2, k))) -
                   mu(i, j - 2, k) * met(1, i, j - 2, k) * met(1, i, j - 2, k) *
                       (c2 * (u(2, i + 2, j - 2, k) - u(2, i - 2, j - 2, k)) +
                        c1 * (u(2, i + 1, j - 2, k) - u(2, i - 1, j - 2, k)))) +
              c1 *
                  (mu(i, j + 1, k) * met(1, i, j + 1, k) * met(1, i, j + 1, k) *
                       (c2 * (u(2, i + 2, j + 1, k) - u(2, i - 2, j + 1, k)) +
                        c1 * (u(2, i + 1, j + 1, k) - u(2, i - 1, j + 1, k))) -
                   mu(i, j - 1, k) * met(1, i, j - 1, k) * met(1, i, j - 1, k) *
                       (c2 * (u(2, i + 2, j - 1, k) - u(2, i - 2, j - 1, k)) +
                        c1 * (u(2, i + 1, j - 1, k) - u(2, i - 1, j - 1, k))));

          // qp-derivatives
          // 38 ops, tot=345
          r1 +=
              c2 *
                  (la(i + 2, j, k) * met(1, i + 2, j, k) * met(1, i + 2, j, k) *
                       (c2 * (u(2, i + 2, j + 2, k) - u(2, i + 2, j - 2, k)) +
                        c1 * (u(2, i + 2, j + 1, k) - u(2, i + 2, j - 1, k))) -
                   la(i - 2, j, k) * met(1, i - 2, j, k) * met(1, i - 2, j, k) *
                       (c2 * (u(2, i - 2, j + 2, k) - u(2, i - 2, j - 2, k)) +
                        c1 * (u(2, i - 2, j + 1, k) - u(2, i - 2, j - 1, k)))) +
              c1 *
                  (la(i + 1, j, k) * met(1, i + 1, j, k) * met(1, i + 1, j, k) *
                       (c2 * (u(2, i + 1, j + 2, k) - u(2, i + 1, j - 2, k)) +
                        c1 * (u(2, i + 1, j + 1, k) - u(2, i + 1, j - 1, k))) -
                   la(i - 1, j, k) * met(1, i - 1, j, k) * met(1, i - 1, j, k) *
                       (c2 * (u(2, i - 1, j + 2, k) - u(2, i - 1, j - 2, k)) +
                        c1 * (u(2, i - 1, j + 1, k) - u(2, i - 1, j - 1, k))));

          // pr-derivatives
          // 130 ops., tot=475
          r1 +=
              c2 *
                  ((2 * mu(i, j, k + 2) + la(i, j, k + 2)) *
                       met(2, i, j, k + 2) * met(1, i, j, k + 2) *
                       (c2 * (u(1, i + 2, j, k + 2) - u(1, i - 2, j, k + 2)) +
                        c1 * (u(1, i + 1, j, k + 2) - u(1, i - 1, j, k + 2))) *
                       strx(i) * istry +
                   mu(i, j, k + 2) * met(3, i, j, k + 2) * met(1, i, j, k + 2) *
                       (c2 * (u(2, i + 2, j, k + 2) - u(2, i - 2, j, k + 2)) +
                        c1 * (u(2, i + 1, j, k + 2) - u(2, i - 1, j, k + 2))) +
                   mu(i, j, k + 2) * met(4, i, j, k + 2) * met(1, i, j, k + 2) *
                       (c2 * (u(3, i + 2, j, k + 2) - u(3, i - 2, j, k + 2)) +
                        c1 * (u(3, i + 1, j, k + 2) - u(3, i - 1, j, k + 2))) *
                       istry -
                   ((2 * mu(i, j, k - 2) + la(i, j, k - 2)) *
                        met(2, i, j, k - 2) * met(1, i, j, k - 2) *
                        (c2 * (u(1, i + 2, j, k - 2) - u(1, i - 2, j, k - 2)) +
                         c1 * (u(1, i + 1, j, k - 2) - u(1, i - 1, j, k - 2))) *
                        strx(i) * istry +
                    mu(i, j, k - 2) * met(3, i, j, k - 2) *
                        met(1, i, j, k - 2) *
                        (c2 * (u(2, i + 2, j, k - 2) - u(2, i - 2, j, k - 2)) +
                         c1 * (u(2, i + 1, j, k - 2) - u(2, i - 1, j, k - 2))) +
                    mu(i, j, k - 2) * met(4, i, j, k - 2) *
                        met(1, i, j, k - 2) *
                        (c2 * (u(3, i + 2, j, k - 2) - u(3, i - 2, j, k - 2)) +
                         c1 * (u(3, i + 1, j, k - 2) - u(3, i - 1, j, k - 2))) *
                        istry)) +
              c1 *
                  ((2 * mu(i, j, k + 1) + la(i, j, k + 1)) *
                       met(2, i, j, k + 1) * met(1, i, j, k + 1) *
                       (c2 * (u(1, i + 2, j, k + 1) - u(1, i - 2, j, k + 1)) +
                        c1 * (u(1, i + 1, j, k + 1) - u(1, i - 1, j, k + 1))) *
                       strx(i) * istry +
                   mu(i, j, k + 1) * met(3, i, j, k + 1) * met(1, i, j, k + 1) *
                       (c2 * (u(2, i + 2, j, k + 1) - u(2, i - 2, j, k + 1)) +
                        c1 * (u(2, i + 1, j, k + 1) - u(2, i - 1, j, k + 1))) +
                   mu(i, j, k + 1) * met(4, i, j, k + 1) * met(1, i, j, k + 1) *
                       (c2 * (u(3, i + 2, j, k + 1) - u(3, i - 2, j, k + 1)) +
                        c1 * (u(3, i + 1, j, k + 1) - u(3, i - 1, j, k + 1))) *
                       istry -
                   ((2 * mu(i, j, k - 1) + la(i, j, k - 1)) *
                        met(2, i, j, k - 1) * met(1, i, j, k - 1) *
                        (c2 * (u(1, i + 2, j, k - 1) - u(1, i - 2, j, k - 1)) +
                         c1 * (u(1, i + 1, j, k - 1) - u(1, i - 1, j, k - 1))) *
                        strx(i) * istry +
                    mu(i, j, k - 1) * met(3, i, j, k - 1) *
                        met(1, i, j, k - 1) *
                        (c2 * (u(2, i + 2, j, k - 1) - u(2, i - 2, j, k - 1)) +
                         c1 * (u(2, i + 1, j, k - 1) - u(2, i - 1, j, k - 1))) +
                    mu(i, j, k - 1) * met(4, i, j, k - 1) *
                        met(1, i, j, k - 1) *
                        (c2 * (u(3, i + 2, j, k - 1) - u(3, i - 2, j, k - 1)) +
                         c1 * (u(3, i + 1, j, k - 1) - u(3, i - 1, j, k - 1))) *
                        istry));

          // rp derivatives
          // 130 ops, tot=605
          r1 +=
              (c2 *
                   ((2 * mu(i + 2, j, k) + la(i + 2, j, k)) *
                        met(2, i + 2, j, k) * met(1, i + 2, j, k) *
                        (c2 * (u(1, i + 2, j, k + 2) - u(1, i + 2, j, k - 2)) +
                         c1 * (u(1, i + 2, j, k + 1) - u(1, i + 2, j, k - 1))) *
                        strx(i + 2) +
                    la(i + 2, j, k) * met(3, i + 2, j, k) *
                        met(1, i + 2, j, k) *
                        (c2 * (u(2, i + 2, j, k + 2) - u(2, i + 2, j, k - 2)) +
                         c1 * (u(2, i + 2, j, k + 1) - u(2, i + 2, j, k - 1))) *
                        stry(j) +
                    la(i + 2, j, k) * met(4, i + 2, j, k) *
                        met(1, i + 2, j, k) *
                        (c2 * (u(3, i + 2, j, k + 2) - u(3, i + 2, j, k - 2)) +
                         c1 * (u(3, i + 2, j, k + 1) - u(3, i + 2, j, k - 1))) -
                    ((2 * mu(i - 2, j, k) + la(i - 2, j, k)) *
                         met(2, i - 2, j, k) * met(1, i - 2, j, k) *
                         (c2 * (u(1, i - 2, j, k + 2) - u(1, i - 2, j, k - 2)) +
                          c1 *
                              (u(1, i - 2, j, k + 1) - u(1, i - 2, j, k - 1))) *
                         strx(i - 2) +
                     la(i - 2, j, k) * met(3, i - 2, j, k) *
                         met(1, i - 2, j, k) *
                         (c2 * (u(2, i - 2, j, k + 2) - u(2, i - 2, j, k - 2)) +
                          c1 *
                              (u(2, i - 2, j, k + 1) - u(2, i - 2, j, k - 1))) *
                         stry(j) +
                     la(i - 2, j, k) * met(4, i - 2, j, k) *
                         met(1, i - 2, j, k) *
                         (c2 * (u(3, i - 2, j, k + 2) - u(3, i - 2, j, k - 2)) +
                          c1 * (u(3, i - 2, j, k + 1) -
                                u(3, i - 2, j, k - 1))))) +
               c1 *
                   ((2 * mu(i + 1, j, k) + la(i + 1, j, k)) *
                        met(2, i + 1, j, k) * met(1, i + 1, j, k) *
                        (c2 * (u(1, i + 1, j, k + 2) - u(1, i + 1, j, k - 2)) +
                         c1 * (u(1, i + 1, j, k + 1) - u(1, i + 1, j, k - 1))) *
                        strx(i + 1) +
                    la(i + 1, j, k) * met(3, i + 1, j, k) *
                        met(1, i + 1, j, k) *
                        (c2 * (u(2, i + 1, j, k + 2) - u(2, i + 1, j, k - 2)) +
                         c1 * (u(2, i + 1, j, k + 1) - u(2, i + 1, j, k - 1))) *
                        stry(j) +
                    la(i + 1, j, k) * met(4, i + 1, j, k) *
                        met(1, i + 1, j, k) *
                        (c2 * (u(3, i + 1, j, k + 2) - u(3, i + 1, j, k - 2)) +
                         c1 * (u(3, i + 1, j, k + 1) - u(3, i + 1, j, k - 1))) -
                    ((2 * mu(i - 1, j, k) + la(i - 1, j, k)) *
                         met(2, i - 1, j, k) * met(1, i - 1, j, k) *
                         (c2 * (u(1, i - 1, j, k + 2) - u(1, i - 1, j, k - 2)) +
                          c1 *
                              (u(1, i - 1, j, k + 1) - u(1, i - 1, j, k - 1))) *
                         strx(i - 1) +
                     la(i - 1, j, k) * met(3, i - 1, j, k) *
                         met(1, i - 1, j, k) *
                         (c2 * (u(2, i - 1, j, k + 2) - u(2, i - 1, j, k - 2)) +
                          c1 *
                              (u(2, i - 1, j, k + 1) - u(2, i - 1, j, k - 1))) *
                         stry(j) +
                     la(i - 1, j, k) * met(4, i - 1, j, k) *
                         met(1, i - 1, j, k) *
                         (c2 * (u(3, i - 1, j, k + 2) - u(3, i - 1, j, k - 2)) +
                          c1 * (u(3, i - 1, j, k + 1) -
                                u(3, i - 1, j, k - 1)))))) *
              istry;

          // qr derivatives
          // 82 ops, tot=687
          r1 +=
              c2 *
                  (mu(i, j, k + 2) * met(3, i, j, k + 2) * met(1, i, j, k + 2) *
                       (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j - 2, k + 2)) +
                        c1 * (u(1, i, j + 1, k + 2) - u(1, i, j - 1, k + 2))) *
                       stry(j) * istrx +
                   la(i, j, k + 2) * met(2, i, j, k + 2) * met(1, i, j, k + 2) *
                       (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j - 2, k + 2)) +
                        c1 * (u(2, i, j + 1, k + 2) - u(2, i, j - 1, k + 2))) -
                   (mu(i, j, k - 2) * met(3, i, j, k - 2) *
                        met(1, i, j, k - 2) *
                        (c2 * (u(1, i, j + 2, k - 2) - u(1, i, j - 2, k - 2)) +
                         c1 * (u(1, i, j + 1, k - 2) - u(1, i, j - 1, k - 2))) *
                        stry(j) * istrx +
                    la(i, j, k - 2) * met(2, i, j, k - 2) *
                        met(1, i, j, k - 2) *
                        (c2 * (u(2, i, j + 2, k - 2) - u(2, i, j - 2, k - 2)) +
                         c1 * (u(2, i, j + 1, k - 2) -
                               u(2, i, j - 1, k - 2))))) +
              c1 *
                  (mu(i, j, k + 1) * met(3, i, j, k + 1) * met(1, i, j, k + 1) *
                       (c2 * (u(1, i, j + 2, k + 1) - u(1, i, j - 2, k + 1)) +
                        c1 * (u(1, i, j + 1, k + 1) - u(1, i, j - 1, k + 1))) *
                       stry(j) * istrx +
                   la(i, j, k + 1) * met(2, i, j, k + 1) * met(1, i, j, k + 1) *
                       (c2 * (u(2, i, j + 2, k + 1) - u(2, i, j - 2, k + 1)) +
                        c1 * (u(2, i, j + 1, k + 1) - u(2, i, j - 1, k + 1))) -
                   (mu(i, j, k - 1) * met(3, i, j, k - 1) *
                        met(1, i, j, k - 1) *
                        (c2 * (u(1, i, j + 2, k - 1) - u(1, i, j - 2, k - 1)) +
                         c1 * (u(1, i, j + 1, k - 1) - u(1, i, j - 1, k - 1))) *
                        stry(j) * istrx +
                    la(i, j, k - 1) * met(2, i, j, k - 1) *
                        met(1, i, j, k - 1) *
                        (c2 * (u(2, i, j + 2, k - 1) - u(2, i, j - 2, k - 1)) +
                         c1 *
                             (u(2, i, j + 1, k - 1) - u(2, i, j - 1, k - 1)))));

          // rq derivatives
          // 82 ops, tot=769
          r1 +=
              c2 *
                  (mu(i, j + 2, k) * met(3, i, j + 2, k) * met(1, i, j + 2, k) *
                       (c2 * (u(1, i, j + 2, k + 2) - u(1, i, j + 2, k - 2)) +
                        c1 * (u(1, i, j + 2, k + 1) - u(1, i, j + 2, k - 1))) *
                       stry(j + 2) * istrx +
                   mu(i, j + 2, k) * met(2, i, j + 2, k) * met(1, i, j + 2, k) *
                       (c2 * (u(2, i, j + 2, k + 2) - u(2, i, j + 2, k - 2)) +
                        c1 * (u(2, i, j + 2, k + 1) - u(2, i, j + 2, k - 1))) -
                   (mu(i, j - 2, k) * met(3, i, j - 2, k) *
                        met(1, i, j - 2, k) *
                        (c2 * (u(1, i, j - 2, k + 2) - u(1, i, j - 2, k - 2)) +
                         c1 * (u(1, i, j - 2, k + 1) - u(1, i, j - 2, k - 1))) *
                        stry(j - 2) * istrx +
                    mu(i, j - 2, k) * met(2, i, j - 2, k) *
                        met(1, i, j - 2, k) *
                        (c2 * (u(2, i, j - 2, k + 2) - u(2, i, j - 2, k - 2)) +
                         c1 * (u(2, i, j - 2, k + 1) -
                               u(2, i, j - 2, k - 1))))) +
              c1 *
                  (mu(i, j + 1, k) * met(3, i, j + 1, k) * met(1, i, j + 1, k) *
                       (c2 * (u(1, i, j + 1, k + 2) - u(1, i, j + 1, k - 2)) +
                        c1 * (u(1, i, j + 1, k + 1) - u(1, i, j + 1, k - 1))) *
                       stry(j + 1) * istrx +
                   mu(i, j + 1, k) * met(2, i, j + 1, k) * met(1, i, j + 1, k) *
                       (c2 * (u(2, i, j + 1, k + 2) - u(2, i, j + 1, k - 2)) +
                        c1 * (u(2, i, j + 1, k + 1) - u(2, i, j + 1, k - 1))) -
                   (mu(i, j - 1, k) * met(3, i, j - 1, k) *
                        met(1, i, j - 1, k) *
                        (c2 * (u(1, i, j - 1, k + 2) - u(1, i, j - 1, k - 2)) +
                         c1 * (u(1, i, j - 1, k + 1) - u(1, i, j - 1, k - 1))) *
                        stry(j - 1) * istrx +
                    mu(i, j - 1, k) * met(2, i, j - 1, k) *
                        met(1, i, j - 1, k) *
                        (c2 * (u(2, i, j - 1, k + 2) - u(2, i, j - 1, k - 2)) +
                         c1 *
                             (u(2, i, j - 1, k + 1) - u(2, i, j - 1, k - 1)))));

          // 4 ops, tot=773
          lu(1, i, j, k) = a1 * lu(1, i, j, k) + sgn * r1 * ijac;





        }

///

#ifndef RAJAPerf_Apps_SW4CK_KERNEL_2_HPP
#define RAJAPerf_Apps_SW4CK_KERNEL_2_HPP

#define SW4CK_KERNEL_2_DATA_SETUP \

#define SW4CK_KERNEL_2_BODY \



#include "common/KernelBase.hpp"

namespace rajaperf
{
class RunParams;

namespace apps
{
class ADomain;

class SW4CK_KERNEL_2 : public KernelBase
{
public:

  SW4CK_KERNEL_2(const RunParams& params);

  ~SW4CK_KERNEL_2();

  void setUp(VariantID vid, size_t tune_idx);
  void updateChecksum(VariantID vid, size_t tune_idx);
  void tearDown(VariantID vid, size_t tune_idx);

  void runSeqVariant(VariantID vid, size_t tune_idx);
  void runOpenMPVariant(VariantID vid, size_t tune_idx);
  void runCudaVariant(VariantID vid, size_t tune_idx);
  void runHipVariant(VariantID vid, size_t tune_idx);
  void runOpenMPTargetVariant(VariantID vid, size_t tune_idx);

  void setCudaTuningDefinitions(VariantID vid);
  void setHipTuningDefinitions(VariantID vid);
  template < size_t block_size >
  void runCudaVariantImpl(VariantID vid);
  template < size_t block_size >
  void runHipVariantImpl(VariantID vid);

private:
  static const size_t default_gpu_block_size = 256;
  using gpu_block_sizes_type = gpu_block_size::make_list_type<default_gpu_block_size>;

  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_z;
  Real_ptr m_vol;

  Real_type m_vnormq;

  ADomain* m_domain;
  Index_type m_array_length;
};

} // end namespace apps
} // end namespace rajaperf

#endif // closing endif for header file include guard
