
#ifndef __E2K_QS_I
# define __E2K_QN   5
# define __E2K_QS_I 1
#endif

#define __E2K_FUNCT(a,n,i,b) /* ..... */ a##n##_##i##b##i
#define __E2K_CONST(a,n,i)   /* ..... */ a##n##_##i
#define __E2K_TEMPL(a,n,i,b) __E2K_FUNCT(a, n, i, b)
#define __E2K_TYPEM(a,n,i)   __E2K_CONST(a, n, i)
#define __E2K_QN_T           __E2K_TYPEM(vd_q, __E2K_QN, __E2K_QS_I)
#define __E2K_Q8_T           __E2K_TYPEM(vd_q, /*  */ 8, __E2K_QS_I)

/*
   Tamplate function for vec_dot_[ q4_0_q8_0, q4_1_q8_1,
                                   q5_0_q8_0, q5_1_q8_1 ]
*/
__E2K_INLINE float __E2K_TEMPL(__e2k_vec_dot_q, __E2K_QN, __E2K_QS_I, _q8_)(
    const int nb,
    const __E2K_QN_T * restrict x,
    const __E2K_Q8_T * restrict y
) {
/*
    0x1DE2 - is max posible value for `q4` line (0xF*0xFF + 0xF*0xFF),
             we can safely sum of 16b vec 4 or 8 times (0x1DE2 * 8 => 0xEF10)

    0x3DC2 - is max for `q5` line (0x1F*0xFF + 0x1F*0xFF)
             and also can be safely 16b vec sum 4 times (0x3DC2 * 4 => 0xF708)

    We also don't need to keep the correct order of the sum,
    we only need to take care of the correct multiplication of bytes in vec.
*/

    float sumf = 0.0;
    int i;

#if __E2K_QN == 5
# define _BAIS_ 0x1010101010101010LL
#else
# define _BAIS_ 0x0808080808080808LL
#endif

#if __e2k_v__ >= 5
    const __vd mas4 = __builtin_e2k_qppackdl(_MAS4_, _MAS4_),
               bais = __builtin_e2k_qppackdl(_BAIS_, _BAIS_);
#else
    const __vd mas4 = _MAS4_,
               bais = _BAIS_;
#endif
#undef _BAIS_

#pragma loop count(1000)
    for (i = 0; i < nb; i++)
    {
#if __E2K_QS_I == 1
        const float ms = __e2k_cvt_f16_f32(x[i].m) * y[i].s;
        const float md = __e2k_cvt_f16_f32(x[i].d) * y[i].d;
#else
        const float md = __e2k_cvt_f16_f32(x[i].d) * __e2k_cvt_f16_f32(y[i].d);
#endif
        int sumi, j, k;

        __vd vs[QK4_L];

#if __E2K_QN == 5
        type_umsk_256 qh = unpack_msk32(x[i].qh[0], bais);
#endif
#pragma unroll
        for (j = 0, k = QK4_L; j < QK4_L; j++, k++)
        {
            __vd y_l = y[i].qs[j],
                 y_h = y[i].qs[k],
                 x_l = x[i].qs[j],
                 x_h = __e2k_vshift(rlh, x_l, 4);
            // Unpack nibbles into individual bytes
            x_l = __e2k_vbitw(and, x_l, mas4); // 0x1E   -> 0x0E
            x_h = __e2k_vbitw(and, x_h, mas4); // 0xF1F2 -> 0x0F1F -> 0x0F0F

#if __E2K_QN == 5
            x_l = __e2k_vbitw(or, x_l, qh.ml[j]);
            x_h = __e2k_vbitw(or, x_h, qh.mh[j]);
#endif
#if __E2K_QS_I == 0
            // Move each one in [ -8 .. +7, -16 .. +15 ] interval:
            x_l = __e2k_varith(subsb, x_l, bais);
            x_h = __e2k_varith(subsb, x_h, bais);
#endif
            // Perform multiplication and create 16-bit values
            y_l = __e2k_vmadd_i8_i16(x_l, y_l);
            y_h = __e2k_vmadd_i8_i16(x_h, y_h);
            // Sum overflow is impossible: 0xF * 0xFF * 4 => 0x7788
            vs[j] = __e2k_varith(addh, y_l, y_h);
        }

#if   __e2k_v__ >= 5
        type_union_128 sat = {
            .__v2di = __e2k_vhsat_i16_i32(vs[0])
        };
        sumi = sat.i.i0 + sat.i.i1 + sat.i.i2 + sat.i.i3;
#else
        type_union_64 sat = {
            .l0 = __e2k_vhsat_i16_i32(__e2k_varith(addh, vs[0], vs[1]))
        };
        sumi = sat.i.i0 + sat.i.i1;
#endif
        sumf += md * sumi;
#if __E2K_QS_I == 1
        sumf += ms, bais; // just fo fix warning
#endif
    }
    return sumf;
}

#undef __E2K_FUNCT
#undef __E2K_CONST
#undef __E2K_TEMPL
#undef __E2K_TYPEM
#undef __E2K_QN_T
#undef __E2K_Q8_T
